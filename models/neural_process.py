# ------------------------------------------------------------------------
# Jie Liu
# University of Amsterdam
# ------------------------------------------------------------------------

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
from collections import OrderedDict

from torch.distributions import kl_divergence, Normal


def build_mlp(dim_in, dim_hid, dim_out, depth,last_bias=True):
    modules = [nn.Linear(dim_in, dim_hid), nn.ReLU(True)]
    for _ in range(depth-2):
        modules.append(nn.Linear(dim_hid, dim_hid))
        modules.append(nn.ReLU(True))
    modules.append(nn.Linear(dim_hid, dim_out, bias=last_bias))
    return nn.Sequential(*modules)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class TaskTokenTransformer(nn.Module):
    def __init__(self, input_dim: int, layers: int, heads: int, output_dim: int, taskpres=True):  # 512, 2, 8, 512
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual = taskpres

        self.ln_pre = LayerNorm(output_dim)
        self.ln_post = LayerNorm(output_dim)

        self.class_embedding = nn.Parameter(torch.empty((1, output_dim), dtype=torch.float32).normal_(0., 0.1), requires_grad=True)

        self.transformer = Transformer(output_dim, layers, heads)

        self.mu_layer = build_mlp(input_dim, input_dim, output_dim, 2)
        self.sigma_layer = build_mlp(input_dim, input_dim, output_dim, 2)

    def forward(self, input_context):

        class_embedding = input_context.mean(dim=0).unsqueeze(0)

        x = torch.cat([class_embedding, input_context], dim=0)

        x = self.ln_pre(x)
        x = self.transformer(x)
        x = self.ln_post(x[0, :].unsqueeze(0))

        dist = self.normal_distribution(x) # [1, 128]

        return dist
    
    def normal_distribution(self, input):
        mu = self.mu_layer(input)
        sigma = F.softplus(self.sigma_layer(input), beta=1, threshold=20)
        dist = Normal(mu, sigma)
        return dist

class NPModulationHead(nn.Module):
    def __init__(self, dim=128, num_sampling=10, scene_weight=0.2):
        super().__init__()
        
        # instance-level distribution functions
        self.mu_layer = build_mlp(dim_in=dim, dim_hid=dim, dim_out=dim, depth=2)
        self.sigma_layer = build_mlp(dim_in=dim, dim_hid=dim, dim_out=dim, depth=2)
        self.num_sampling = num_sampling

        # scene-level distribution functions
        self.scene_dist_funcs = TaskTokenTransformer(input_dim=dim, layers=1, heads=8, output_dim=dim)
        self.num_scene_samples = 1

        # modulation functions
        self.film_generator = build_mlp(dim_in=dim, dim_hid=dim, dim_out=dim*2, depth=2)

        self.scene_weight = scene_weight
           
    def forward(self, fg_click_query, fg_click_query_num_split, bg_query, sample_feats, GT_query_feat, GT_query_split):

        # generate bg, fg instance-level features
        bg_instance_feat = bg_query.mean(dim=0).unsqueeze(0)  # [1, 128]
        fg_click_queries = torch.split(fg_click_query, fg_click_query_num_split, dim=0)  # [N_click, 128]
        fg_instance_feat = torch.cat([query.mean(dim=0).unsqueeze(0) for query in fg_click_queries], dim=0)  # [N_instance, 128]

        # context scene-level distribution
        scene_context_dist = self.scene_dist_funcs(torch.cat([bg_instance_feat, fg_instance_feat], dim=0)) # [1, 128]
        scene_context_samples = scene_context_dist.rsample((self.num_scene_samples,)).squeeze(1)  # [1, 128]

        # bg context instance-level distribution
        bg_instance_dist_input = self.scene_weight*scene_context_samples*bg_instance_feat + (1-self.scene_weight)*bg_instance_feat
        bg_context_instance_dist = self.instance_dist_funcs(bg_instance_dist_input)  # [1, 128]
        bg_context_instance_samples = bg_context_instance_dist.rsample((self.num_sampling,)).squeeze(1) # [n_samples, 128]

        # modulate the bg query features
        gamma_beta = self.film_generator(bg_context_instance_samples)  # [n_samples, 256]
        bg_query_modulated = self.film_modulation(bg_query, gamma_beta)  # [n_samples, n_bg, 128]

        # fg context distribution
        fg_instance_dist_input = self.scene_weight*scene_context_samples.repeat(fg_instance_feat.size(0), 1)*fg_instance_feat + (1-self.scene_weight)*fg_instance_feat
        fg_context_instance_dist = self.instance_dist_funcs(fg_instance_dist_input)  # [N_instance, 128]
        fg_context_instance_samples = fg_context_instance_dist.rsample((self.num_sampling,)) # [n_samples, N_instance, 128]

        # modulate the fg query features
        gamma_beta = self.film_generator(fg_context_instance_samples)  # [n_samples, N_instance, 256]

        fg_query_list = []
        for idx, click_query in enumerate(fg_click_queries):
            fg_query_modulated = self.film_modulation(click_query, gamma_beta[ :, idx, :])  # [n_sample, n_click, 128]
            fg_query_list.append(fg_query_modulated)
        
        fg_query_modulated = torch.cat(fg_query_list, dim=1)  # [n_samples, N_click, 128]

        """building target distribution during training"""
        kl_loss = 0
        if GT_query_feat is not None:
            # generate instance-level features
            gt_queries = torch.split(GT_query_feat, GT_query_split, dim=0)
            gt_instance_feat = torch.cat([query.mean(dim=0).unsqueeze(0) for query in gt_queries], dim=0)  # [N_instance, 128]
            
            # target scene-level distribution
            scene_target_dist = self.scene_dist_funcs(gt_instance_feat) # [1, 128]
            scene_target_samples = scene_target_dist.rsample((self.num_scene_samples,)).squeeze(1)  # [1, 128]

            # split instance-level features into bg and fg
            bg_instance_feat = gt_instance_feat[0, :].unsqueeze(0)  # [1, 128]
            fg_instance_feat = gt_instance_feat[1:, :]
            #  bg instance-level distribution
            bg_instance_dist_input = self.scene_weight*scene_target_samples*bg_instance_feat + (1-self.scene_weight)*bg_instance_feat
            bg_target_instance_dist = self.instance_dist_funcs(bg_instance_dist_input)  # [1, 128]
            bg_target_instance_samples = bg_target_instance_dist.rsample((self.num_sampling,)).squeeze(1) # [n_samples, 128]

            # modulate the bg query features
            bg_gamma_beta = self.film_generator(bg_target_instance_samples)  # [n_samples, 1, 256]
            bg_gt_query_modulated = self.film_modulation(gt_queries[0], bg_gamma_beta)  # [n_samples, 1, 128]

            # fg instance-level distribution
            fg_instance_dist_input = self.scene_weight*scene_target_samples.repeat(fg_instance_feat.size(0), 1)*fg_instance_feat + (1-self.scene_weight)*fg_instance_feat
            fg_target_instance_dist = self.instance_dist_funcs(fg_instance_dist_input)  # [N_instance, 128]
            fg_target_instance_samples = fg_target_instance_dist.rsample((self.num_sampling,)) # [n_samples, N_instance, 128]

            # modulate the fg query features
            fg_gamma_beta = self.film_generator(fg_target_instance_samples)  # [n_samples, N_instance, 256]
            fg_gt_query_list = []
            for idx, gt_query in enumerate(gt_queries[1:]):
                fg_gt_query_modulated = self.film_modulation(gt_query, fg_gamma_beta[ :, idx, :])  # [n_sample, n_gt_query, 128]
                fg_gt_query_list.append(fg_gt_query_modulated)
            fg_gt_query_modulated = torch.cat(fg_gt_query_list, dim=1)  # [n_samples, N_gt_query, 128]
   
            # KL divergence term
            kl_loss += kl_divergence(bg_context_instance_dist, bg_target_instance_dist).sum() # bg instance-level kl loss
            kl_loss += kl_divergence(fg_context_instance_dist, fg_target_instance_dist).sum() # fg instance-level kl loss
            kl_loss += kl_divergence(scene_context_dist, scene_target_dist).sum() # scene-level kl loss

            return kl_loss, fg_query_modulated, bg_query_modulated, fg_gt_query_modulated, bg_gt_query_modulated
        else:
            return kl_loss, fg_query_modulated, bg_query_modulated, None, None
    
    def instance_dist_funcs(self, input):
        mu = self.mu_layer(input)
        sigma = F.softplus(self.sigma_layer(input), beta=1, threshold=20)
        dist = Normal(mu, sigma)
        return dist

    def film_modulation(self, input, gamma_beta):
        """input: [N, 128]  gamma_beta: [M, 256]"""
        gamma, beta = torch.split(gamma_beta, input.size(-1), dim=-1)

        gamma = gamma.unsqueeze(1)  # [M, 1, 128]
        beta = beta.unsqueeze(1)  # [M, 1, 128]

        input = input.unsqueeze(0)  # [1, N, 128]

        output = gamma * input + beta # [M, N, 128]

        return output
        
        

class NPModel_baseline(nn.Module):
    def __init__(self, dim=128, num_sampling=10):
        super().__init__()
        #TaskTokenTransformer(input_dim=dim, layers=1, heads=8, output_dim=dim)
        self.mu_layer = build_mlp(dim_in=dim, dim_hid=dim, dim_out=dim, depth=2)
        self.sigma_layer = build_mlp(dim_in=dim, dim_hid=dim, dim_out=dim, depth=2)
        self.num_sampling = num_sampling

    def forward(self, fg_click_query, fg_click_query_num_split, bg_query, sample_feats, GT_query_feat, GT_query_split, with_bg=True):
        """
        fg_click_query: [N, 128]  fg_click_query_num_split: e.g., [2, 1, 1, 1]  pcd_feats: [N_pcd, 128]  pcd_mask: [n_pcd]"""
        
        """building context distribution"""
        bg_context_dist = self.normal_distribution(bg_query.mean(dim=0).unsqueeze(0))  # [1, 128]
        bg_context_samples = bg_context_dist.rsample((self.num_sampling,)) # [n_samples, 1, 128]
        
        # fg context distribution
        fg_quries = torch.split(fg_click_query, fg_click_query_num_split, dim=0)
        obj_query_list = []
        for query in fg_quries:
            obj_query_list.append(query.mean(dim=0).unsqueeze(0))  # using mean to aggregate the query features
        obj_query = torch.cat(obj_query_list, dim=0)  # [N_obj, 128]

        fg_context_dist = self.normal_distribution(obj_query)  # [N_obj, 128]

        # sampling n latent variables from prior
        bg_context_samples = bg_context_dist.rsample((self.num_sampling,))  # [n_samples, 1, 128]
        fg_context_samples = fg_context_dist.rsample((self.num_sampling,))  # [n_samples, N_obj, 128]

        """building target distribution during training"""
        kl_loss = 0
        if GT_query_feat is not None:
            gt_queries = torch.split(GT_query_feat, GT_query_split, dim=0)
            gt_query_list = []
            for query in gt_queries:
                gt_query_list.append(query.mean(dim=0).unsqueeze(0))
            merged_gt_query = torch.cat(gt_query_list, dim=0)  # [N_gt, 128]

            if not with_bg:
                fg_target_dist = self.normal_distribution(merged_gt_query)  # [N_obj, 128]
                kl_loss += kl_divergence(fg_context_dist, fg_target_dist).sum(-1).mean()
                fg_target_samples = fg_target_dist.rsample((self.num_sampling,)) # [n_samples, N_obj, 128]
                fg_samples = torch.cat([fg_context_samples, fg_target_samples], dim=0)
                return kl_loss, fg_samples, bg_context_samples
            else:     
                # building target distribution, including fb and fg
                bg_target_dist = self.normal_distribution(merged_gt_query[0, :].unsqueeze(0))  # [1, 128]
                fg_target_dist = self.normal_distribution(merged_gt_query[1:, :])  # [N_obj, 128]
                # KL divergence for latent variables
                kl_loss += kl_divergence(bg_context_dist, bg_target_dist).sum(-1).mean()
                kl_loss += kl_divergence(fg_context_dist, fg_target_dist).sum(-1).mean()
                # sampling n latent variables from posterior
                bg_target_samples = bg_target_dist.rsample((self.num_sampling,))  # [n_samples, 1, 128]
                fg_target_samples = fg_target_dist.rsample((self.num_sampling,))  # [n_samples, N_obj, 128]
                # merge context and target samples
                bg_samples = torch.cat([bg_context_samples, bg_target_samples], dim=0)
                fg_samples = torch.cat([fg_context_samples, fg_target_samples], dim=0)

                return kl_loss, fg_samples, bg_samples
        else:  # for inference
            return kl_loss, fg_context_samples, bg_context_samples
    
    def normal_distribution(self, input):
        mu = self.mu_layer(input)
        sigma = F.softplus(self.sigma_layer(input), beta=1, threshold=20)
        dist = Normal(mu, sigma)
        return dist