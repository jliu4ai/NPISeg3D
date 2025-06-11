# ------------------------------------------------------------------------
# Jie Liu
# University of Amsterdam
# The project is modified from AGILE3D，thanks to auhtors!
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import MinkowskiEngine.MinkowskiOps as me
from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling
import numpy as np
from torch.nn import functional as F
from models.modules.common import conv
from models.modules.attention_block import *
from models.position_embedding import PositionEmbeddingCoordsSine, PositionalEncoding3D, PositionalEncoding1D
from torch.cuda.amp import autocast
from .backbone import build_backbone
from models.neural_process import NPModulationHead

import itertools


def build_mlp(dim_in, dim_hid, dim_out, depth,last_bias=True):
    modules = [nn.Linear(dim_in, dim_hid), nn.ReLU(True)]
    for _ in range(depth-2):
        modules.append(nn.Linear(dim_hid, dim_hid))
        modules.append(nn.ReLU(True))
    modules.append(nn.Linear(dim_hid, dim_out, bias=last_bias))
    return nn.Sequential(*modules)

class NPISeg3D(nn.Module):
    def __init__(self, backbone, hidden_dim, num_heads, dim_feedforward,
                 shared_decoder, num_decoders, num_bg_queries, dropout, pre_norm,
                 positional_encoding_type, normalize_pos_enc, hlevels,
                 voxel_size, gauss_scale, aux, np_sampling_num, num_gt_sampling,
                 np_start_layer=-1, scene_weight=0.2
                 ):
        super().__init__()
        # hyperparameters
        self.gauss_scale = gauss_scale
        self.voxel_size = voxel_size
        self.hlevels = hlevels
        self.normalize_pos_enc = normalize_pos_enc
        self.num_decoders = num_decoders
        self.num_bg_queries = num_bg_queries
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.shared_decoder = shared_decoder
        self.mask_dim = hidden_dim
        self.num_heads = num_heads
        self.pos_enc_type = positional_encoding_type
        self.aux = aux

        self.backbone = backbone

        self.lin_squeeze_head = conv(
            self.backbone.PLANES[7], self.mask_dim, kernel_size=1, stride=1, bias=True, D=3
        )

        # background query
        self.bg_query_feat = nn.Embedding(num_bg_queries, hidden_dim)  # [10, 128]
        self.bg_query_pos = nn.Embedding(num_bg_queries, hidden_dim)  # [10, 128]

        # mask embedding in decoding
        self.mask_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # encoding previous masks
        self.mask_encoder = build_mlp(dim_in=15, dim_hid=hidden_dim, dim_out=hidden_dim, depth=4)

        # latent & query fusion
        self.fusion_mlp = build_mlp(dim_in=2*hidden_dim, dim_hid=hidden_dim, dim_out=hidden_dim, depth=2)

        # position encoding type
        if self.pos_enc_type == "legacy":
            self.pos_enc = PositionalEncoding3D(channels=self.mask_dim)
        elif self.pos_enc_type == "fourier":
            self.pos_enc = PositionEmbeddingCoordsSine(pos_type="fourier",
                                                       d_pos=self.mask_dim,
                                                       gauss_scale=self.gauss_scale,
                                                       normalize=self.normalize_pos_enc)
        elif self.pos_enc_type == "sine":
            self.pos_enc = PositionEmbeddingCoordsSine(pos_type="sine",
                                                       d_pos=self.mask_dim,
                                                       normalize=self.normalize_pos_enc)
        else:
            assert False, 'pos enc type not known'

        self.pooling = MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)

        self.masked_transformer_decoder = nn.ModuleList()

        # Click-to-scene attention
        self.c2s_attention = nn.ModuleList()

        # FFN
        self.ffn_attention = nn.ModuleList()

        # Scene-to-click attention
        self.s2c_attention = nn.ModuleList()

        num_shared = self.num_decoders if not self.shared_decoder else 1

        for _ in range(num_shared):
            tmp_c2s_attention = nn.ModuleList()
            tmp_s2c_attention = nn.ModuleList()
            tmp_ffn_attention = nn.ModuleList()

            for i, hlevel in enumerate(self.hlevels):
                tmp_c2s_attention.append(
                    CrossAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm
                    )
                )

                tmp_s2c_attention.append(
                    CrossAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm
                    )
                )

                tmp_ffn_attention.append(
                    FFNLayer(
                        d_model=self.mask_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm
                    )
                )

            self.c2s_attention.append(tmp_c2s_attention)
            self.s2c_attention.append(tmp_s2c_attention)
            self.ffn_attention.append(tmp_ffn_attention)

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # click-to-click attention
        self.c2c_attention = SelfAttentionLayer(
                            d_model=self.mask_dim,
                            nhead=self.num_heads,
                            dropout=self.dropout,
                            normalize_before=self.pre_norm)

        # neural process model here
        self.NPModel = NPModulationHead(num_sampling=np_sampling_num, scene_weight=scene_weight)

        self.num_gt_sampling = num_gt_sampling

        self.np_start_layer = np_start_layer 

    def get_pos_encs(self, coords):
        pos_encodings_pcd = []

        for i in range(len(coords)):
            pos_encodings_pcd.append([[]])
            ### this is a trick to bypass a bug in Minkowski Engine cpu version
            if coords[i].F.is_cuda:
                coords_batches = coords[i].decomposed_features
            else:
                coords_batches = [coords[i].F]
            for coords_batch in coords_batches:
                scene_min = coords_batch.min(dim=0)[0][None, ...]
                scene_max = coords_batch.max(dim=0)[0][None, ...]

                with autocast(enabled=False):
                    tmp = self.pos_enc(coords_batch[None, ...].float(),
                                       input_range=[scene_min, scene_max])

                pos_encodings_pcd[-1][0].append(tmp.squeeze(0).permute((1, 0)))

        return pos_encodings_pcd

    def forward_backbone(self, x, raw_coordinates=None):
        """Get point features and position encoding"""
        pcd_features, aux = self.backbone(x)

        with torch.no_grad():
            coordinates = me.SparseTensor(features=raw_coordinates,
                                          coordinate_manager=aux[-1].coordinate_manager,
                                          coordinate_map_key=aux[-1].coordinate_map_key,
                                          device=aux[-1].device)
            coords = [coordinates]   # 5 times down pooling
            for _ in reversed(range(len(aux)-1)):
                coords.append(self.pooling(coords[-1]))

            coords.reverse()

        pos_encodings_pcd = self.get_pos_encs(coords)

        pcd_features = self.lin_squeeze_head(pcd_features)  # [N,96]--->[N,128]

        return pcd_features, aux, coordinates, pos_encodings_pcd

    def forward_mask(self, pcd_features, aux, coordinates, pos_encodings_pcd, click_idx=None, prev_mask=None, GT_click=None):
        """predict mask
        pcd_features: [N, C], aux:5 aux features list, coordinates: [N, C], pos_encodings: [5, 1, 5], click_idx: [B, N_objs]
        """
        batch_size = pcd_features.C[:, 0].max() + 1   # C denotes coordinates

        predictions_mask = [[] for i in range(batch_size)]  # list [B, N]
        np_loss_list = [[] for i in range(batch_size)]
        target_pred_masks = [[] for i in range(batch_size)]

        bg_learn_queries = self.bg_query_feat.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 10, 128]
        bg_learn_query_pos = self.bg_query_pos.weight.unsqueeze(0).repeat(batch_size, 1, 1)   # [B, 10, 128]

        # deal with each sample seperately, min: [1, 3], max: [1, 3]
        for b in range(batch_size):
            # calculate min and max of coordinate
            if coordinates.F.is_cuda:
                mins = coordinates.decomposed_features[b].min(dim=0)[0].unsqueeze(0)
                maxs = coordinates.decomposed_features[b].max(dim=0)[0].unsqueeze(0)
            else:
                mins = coordinates.F.min(dim=0)[0].unsqueeze(0)
                maxs = coordinates.F.max(dim=0)[0].unsqueeze(0)

            # click idx & order for a specific sample
            click_idx_sample = click_idx[b]

            """foreground click query encoding"""
            fg_obj_num = len(click_idx_sample.keys()) - 1
            # each click has a query
            fg_query_num_split = [len(click_idx_sample[str(i)]) for i in range(1, fg_obj_num+1)]
            fg_query_num = sum(fg_query_num_split)

            # all foreground click coords [1, fg_click_num, 3]
            if coordinates.F.is_cuda:
                fg_clicks_coords = torch.vstack([coordinates.decomposed_features[b][click_idx_sample[str(i)], :]
                                        for i in range(1, fg_obj_num+1)]).unsqueeze(0)
            else:
                fg_clicks_coords = torch.vstack([coordinates.F[click_idx_sample[str(i)], :]
                                        for i in range(1, fg_obj_num+1)]).unsqueeze(0)

            # [1, 128, fg_click_num]
            fg_query_pos = self.pos_enc(fg_clicks_coords.float(), input_range=[mins, maxs])
            fg_query_pos = fg_query_pos.permute((2, 0, 1))[:, 0, :]  # [num_queries, 128]

            # fg_queries: [num_click, 128]
            if pcd_features.F.is_cuda:
                fg_queries = torch.vstack([pcd_features.decomposed_features[b][click_idx_sample[str(i)], :]
                                           for i in range(1, fg_obj_num+1)])
            else:
                fg_queries = torch.vstack([pcd_features.F[click_idx_sample[str(i)], :] for i in range(1, fg_obj_num+1)])
            
            """background click query encoding"""
            # position encoding of background clicks, [1, 128, 10]
            bg_click_idx = click_idx_sample['0']
            if len(bg_click_idx) != 0:
                if coordinates.F.is_cuda:
                    bg_click_coords = coordinates.decomposed_features[b][bg_click_idx].unsqueeze(0)
                else:
                    bg_click_coords = coordinates.F[bg_click_idx].unsqueeze(0)
                bg_query_pos = self.pos_enc(bg_click_coords.float(), input_range=[mins, maxs])  # [num_queries, 128]
                bg_query_pos = torch.cat([bg_learn_query_pos[b].T.unsqueeze(0), bg_query_pos], dim=-1)
            else:
                bg_query_pos = bg_learn_query_pos[b].T.unsqueeze(0)
            
            bg_query_pos = bg_query_pos.permute((2, 0, 1))[:, 0, :]  # [num_queries, 128]
            
            # 10 learnable bg query + background click query
            bg_query_num = bg_query_pos.shape[0]
            if len(bg_click_idx)!=0:
                if pcd_features.F.is_cuda:
                    bg_queries = pcd_features.decomposed_features[b][bg_click_idx, :]
                else:
                    bg_queries = pcd_features.F[bg_click_idx, :]
                bg_queries = torch.cat([bg_learn_queries[b], bg_queries], dim=0)
            else:  # 10 learnable bg query
                bg_queries = bg_learn_queries[b]
            
            """sampled click encoding, only for training"""
            gt_queries, gt_query_pos, gt_query_split, gt_query_num = None, None, None, None
            GT_click_sample =None
            if GT_click is not None:
                GT_click_sample = GT_click[b]
            
                split_list = [len(GT_click_sample[str(i)]) for i in range(len(GT_click_sample))]
                gt_query_split = list(filter(lambda x: x != 0, split_list)) # remove 0
                gt_query_num = sum(gt_query_split)  # both bg and fg
                
                # position encoding of gt clicks
                if coordinates.F.is_cuda:
                    gt_click_coords = torch.vstack([coordinates.decomposed_features[b][GT_click_sample[str(i)], :] for i in range(len(GT_click_sample))]).unsqueeze(0)
                else:
                    gt_click_coords = torch.vstack([coordinates.F[GT_click_sample[str(i)], :] for i in range(len(GT_click_sample))]).unsqueeze(0)
                
                gt_query_pos = self.pos_enc(gt_click_coords.float(), input_range=[mins, maxs]) 
                gt_query_pos = gt_query_pos.permute((2, 0, 1))[:, 0, :]  # [num_queries, 128]

                # gt_queries: [num_click, 128]
                if pcd_features.F.is_cuda:
                    gt_queries = torch.vstack([pcd_features.decomposed_features[b][GT_click_sample[str(i)], :] for i in range(len(GT_click_sample))])
                else:
                    gt_queries = torch.vstack([pcd_features.F[GT_click_sample[str(i)], :] for i in range(len(GT_click_sample))])
                
            """pcd feature """
            if pcd_features.F.is_cuda:
                src_pcd = pcd_features.decomposed_features[b]
            else:
                src_pcd = pcd_features.F
            # mask embedding of previous mask
            if not torch.all(prev_mask[b] == 0):
                mask_embedding = self.mask_encoder(prev_mask[b])
                src_pcd += mask_embedding

            refine_time = 0
            target_src_pcd = None

            """ mask decoder"""
            for decoder_counter in range(self.num_decoders):
                if self.shared_decoder:
                    decoder_counter = 0
                for i, hlevel in enumerate(self.hlevels):
                    # position encoding of pcd feature
                    pos_enc = pos_encodings_pcd[hlevel][0][b]  # [num_points, 128]

                    if refine_time == 0:
                        attn_mask = None

                    if GT_click_sample is not None:
                        # click to scene attention, return query, [num_queries, 128]
                        output = self.c2s_attention[decoder_counter][i](
                            torch.cat([fg_queries, bg_queries, gt_queries], dim=0),  # [num_queries, 128]
                            src_pcd,  # [num_points, 128]
                            memory_mask=attn_mask,
                            memory_key_padding_mask=None,
                            pos=pos_enc,  # [num_points, 128]
                            query_pos=torch.cat([fg_query_pos, bg_query_pos, gt_query_pos], dim=0)  # [num_queries, 128]
                        )

                        # FFN, return query, [num_queries, 128]
                        queries = self.ffn_attention[decoder_counter][i](
                            output
                        )

                        # split query into fg & bg & gt for mask prediction
                        fg_queries, bg_queries, gt_queries = queries.split([fg_query_num, bg_query_num, gt_query_num], 0)

                        # self attention for context clicks
                        context_queries = self.c2c_attention(torch.cat([fg_queries, bg_queries], dim=0),
                                                               tgt_mask=None,
                                                               tgt_key_padding_mask=None,
                                                               query_pos=torch.cat([fg_query_pos, bg_query_pos], dim=0))
                        fg_queries, bg_queries = context_queries.split([fg_query_num, bg_query_num], 0)
                        
                        # self attention for gt clicks
                        gt_queries = self.c2c_attention(gt_queries,
                                                        tgt_mask=None,
                                                        tgt_key_padding_mask=None,
                                                        query_pos=gt_query_pos)
                        # scene to query attention, return pcd feature, [num_points, 128]
                        target_src_pcd = self.s2c_attention[decoder_counter][i](
                        src_pcd,
                        gt_queries,
                        memory_mask=None,
                        memory_key_padding_mask=None,
                        pos=gt_query_pos,
                        query_pos=pos_enc)
                    
                    else:
                        # click to scene attention, return query, [num_queries, 128]
                        output = self.c2s_attention[decoder_counter][i](
                            torch.cat([fg_queries, bg_queries], dim=0),  # [num_queries, 128]
                            src_pcd,  # [num_points, 128]
                            memory_mask=attn_mask,
                            memory_key_padding_mask=None,
                            pos=pos_enc,  # [num_points, 128]
                            query_pos=torch.cat([fg_query_pos, bg_query_pos], dim=0)  # [num_queries, 128]
                        )

                        # FFN, return query, [num_queries, 128]
                        queries = self.ffn_attention[decoder_counter][i](
                            output
                        )

                        # self attention for context clicks
                        queries = self.c2c_attention(queries,
                                                     tgt_mask=None,
                                                     tgt_key_padding_mask=None,
                                                     query_pos=torch.cat([fg_query_pos, bg_query_pos], dim=0))

                        # split query into fg & bg & gt for mask prediction
                        fg_queries, bg_queries = queries.split([fg_query_num, bg_query_num], 0)

                    # scene to query attention, return pcd feature, [num_points, 128]
                    src_pcd = self.s2c_attention[decoder_counter][i](
                        src_pcd,
                        torch.cat([fg_queries, bg_queries], dim=0),
                        memory_mask=None,
                        memory_key_padding_mask=None,
                        pos=torch.cat([fg_query_pos, bg_query_pos], dim=0),
                        query_pos=pos_enc
                    )

                    # predict mask for each point, outputs_mask: [num_points, num_objs+1], attn_mask: [num_queries, num_points]
                    if decoder_counter >= self.np_start_layer:
                        attn_mask, np_loss, outputs_mask, target_mask = self.NP_mask_module(
                                    fg_queries,
                                    bg_queries,
                                    src_pcd,
                                    fg_query_num_split=fg_query_num_split,
                                    GT_query_feat=gt_queries,
                                    GT_query_split=gt_query_split)
                        np_loss_list[b].append(np_loss)
                    else:
                        outputs_mask, target_mask, attn_mask = self.mask_module(
                                                                        fg_queries,
                                                                        bg_queries,
                                                                        src_pcd,
                                                                        fg_query_num_split=fg_query_num_split,
                                                                        gt_query=gt_queries,
                                                                        gt_query_split=gt_query_split,
                                                                        target_src_pcd=target_src_pcd)

                    # record prediction of each pcd
                    predictions_mask[b].append(outputs_mask)
                    if target_mask is not None:
                        target_pred_masks[b].append(target_mask)
                    refine_time += 1

        # output masks, 0, 1, 2; 2 aux masks
        predictions_mask = [list(i) for i in zip(*predictions_mask)]
        target_pred_masks = [list(i) for i in zip(*target_pred_masks)]
        np_loss_mean = sum([sum(loss) for loss in np_loss_list])/len(np_loss_list)
        out = {
            'pred_masks': predictions_mask[-1],
            'backbone_features': pcd_features,
            'np_loss': np_loss_mean if self.training else None
        }

        if self.aux:
            out['aux_outputs'] = self._set_aux_loss(predictions_mask)
            out['aux_target_masks'] = self._set_aux_loss(target_pred_masks)
            if self.training:
                out['aux_prob_target_mask'] = [{"pred_masks": target_pred_masks[-1]}]

        return out

    def mask_module(self, fg_query_feat, bg_query_feat, mask_features, fg_query_num_split=None, gt_query=None, gt_query_split=None, target_src_pcd=None):
        # pre norm & embedding
        fg_mask_embed = self.mask_embed_head(self.decoder_norm(fg_query_feat)) # [N_fg, 128]
        bg_mask_embed = self.mask_embed_head(self.decoder_norm(bg_query_feat)) # [N_bg, 128]
        if gt_query is not None:
            gt_mask_embed = self.mask_embed_head(self.decoder_norm(gt_query)) # [N_gt_click, 128]

        # context query mask prediction
        fg_prods = mask_features @ fg_mask_embed.T
        fg_prods = fg_prods.split(fg_query_num_split, dim=1)
        fg_masks = []
        for fg_prod in fg_prods:
            fg_masks.append(fg_prod.max(dim=-1, keepdim=True)[0])
        fg_masks = torch.cat(fg_masks, dim=-1)
        bg_masks = (mask_features @ bg_mask_embed.T).max(dim=-1, keepdim=True)[0]
        output_masks = torch.cat([bg_masks, fg_masks], dim=-1) # [N_points, obj+1]

        output_labels = output_masks.argmax(1) # [N_points]
        
        # attention mask for context query in transformer
        bg_attn_mask = ~(output_labels == 0)
        bg_attn_mask = bg_attn_mask.unsqueeze(0).repeat(bg_query_feat.shape[0], 1)
        bg_attn_mask[torch.where(bg_attn_mask.sum(-1) == bg_attn_mask.shape[-1])] = False
        fg_attn_mask = []
        for fg_obj_id in range(1, fg_masks.size(-1)+1):
            fg_obj_mask = ~(output_labels == fg_obj_id)
            fg_obj_mask = fg_obj_mask.unsqueeze(0).repeat(fg_query_num_split[fg_obj_id-1], 1)
            fg_obj_mask[torch.where(fg_obj_mask.sum(-1) == fg_obj_mask.shape[-1])] = False
            fg_attn_mask.append(fg_obj_mask)
        fg_attn_mask = torch.cat(fg_attn_mask, dim=0)

        # target query mask prediction
        if gt_query is not None:
            all_prods = target_src_pcd @ gt_mask_embed.T  # [N_points, N_gt_click]
            all_prods = all_prods.split(gt_query_split, dim=1)
            pred_target_masks = []
            for prod in all_prods:
                pred_target_masks.append(prod.max(dim=-1, keepdim=True)[0])
            pred_target_masks = torch.cat(pred_target_masks, dim=-1)  # [N_points,  obj]
            pred_target_labels = pred_target_masks.argmax(1)  # [N_points]

        # attention mask for target query in transformer
        if gt_query is not None:
            gt_fg_attn_mask = []
            for gt_obj_id in range(1, fg_masks.size(-1)+1): # gt click query attention mask
                gt_obj_mask = ~(pred_target_labels == gt_obj_id)
                num_gt_points = self.num_gt_sampling
                gt_obj_mask = gt_obj_mask.unsqueeze(0).repeat(num_gt_points, 1)
                gt_obj_mask[torch.where(gt_obj_mask.sum(-1) == gt_obj_mask.shape[-1])] = False
                gt_fg_attn_mask.append(gt_obj_mask)
            gt_fg_attn_mask = torch.cat(gt_fg_attn_mask, dim=0)
            gt_bg_attn_mask = ~(pred_target_labels == 0)
            gt_bg_attn_mask = gt_bg_attn_mask.unsqueeze(0).repeat(gt_query_split[0], 1)
            gt_bg_attn_mask[torch.where(gt_bg_attn_mask.sum(-1) == gt_bg_attn_mask.shape[-1])] = False

        # return output masks and attention mask
        if gt_query is not None:
            return output_masks, pred_target_masks, torch.cat([fg_attn_mask, bg_attn_mask, gt_bg_attn_mask, gt_fg_attn_mask], dim=0)
        else:
            attn_mask = torch.cat([fg_attn_mask, bg_attn_mask], dim=0)
            return output_masks, None, attn_mask
    
    def NP_mask_module(self, fg_query_feat, bg_query_feat, mask_features, fg_query_num_split=None, GT_query_feat=None, GT_query_split=None):
        
        # decoder norm
        fg_query_feat = self.mask_embed_head(self.decoder_norm(fg_query_feat))  # [N_fg, 128]
        bg_query_feat = self.mask_embed_head(self.decoder_norm(bg_query_feat))  # [N_bg, 128]
        if GT_query_feat is not None:
            GT_query_feat = self.mask_embed_head(self.decoder_norm(GT_query_feat))  # [N_gt_click, 128]
        
        # set neural process here  
        np_loss, fg_context, bg_context, fg_target, bg_target = self.NPModel(fg_query_feat, fg_query_num_split, bg_query_feat, mask_features, GT_query_feat, GT_query_split)
       
        # context query mask prediction
        fg_context_preds =  torch.einsum("kd, mnd->kmn", mask_features, fg_context) # [N_points, N_samples, N_query]
        bg_context_preds =  torch.einsum("kd, mnd->kmn", mask_features, bg_context) # [N_points, N_samples, N_bg_query]
        
        fg_context_preds = fg_context_preds.split(fg_query_num_split, dim=-1)
        fg_context_mask = []
        for fg_context_pred in fg_context_preds:
            fg_context_mask.append(fg_context_pred.max(dim=-1, keepdim=True)[0])
        fg_context_mask = torch.cat(fg_context_mask, dim=-1) # [N_points, N_samples, N_objs]

        bg_context_mask = bg_context_preds.max(dim=-1, keepdim=True)[0]  # [N_points, N_samples, 1]

        context_masks = torch.cat([bg_context_mask, fg_context_mask], dim=-1).permute(0, 2, 1)  # [N_points, N_objs+1, N_samples]

        context_labels = context_masks.mean(dim=-1).argmax(1)

        # attention mask for transformer
        bg_attn_mask = ~(context_labels == 0)  
        bg_attn_mask = bg_attn_mask.unsqueeze(0).repeat(bg_query_feat.shape[0], 1)
        bg_attn_mask[torch.where(bg_attn_mask.sum(-1) == bg_attn_mask.shape[-1])] = False

        fg_attn_mask = []
        for fg_obj_id in range(1, fg_context_mask.size(-1)+1):
            fg_obj_mask = ~(context_labels == fg_obj_id)
            fg_obj_mask = fg_obj_mask.unsqueeze(0).repeat(fg_query_num_split[fg_obj_id-1], 1)
            fg_obj_mask[torch.where(fg_obj_mask.sum(-1) == fg_obj_mask.shape[-1])] = False
            fg_attn_mask.append(fg_obj_mask)

        fg_attn_mask = torch.cat(fg_attn_mask, dim=0)

        # including attention mask for gt clicks
        if GT_query_feat is not None:
            if bg_target is None:
                fg_target_preds =  torch.einsum("kd, mnd->kmn", mask_features, fg_target) # [N_points, N_samples, N_query]
                fg_target_preds = fg_target_preds.split(GT_query_split, dim=-1)
                fg_target_mask = []
                for fg_target_pred in fg_target_preds:
                    fg_target_mask.append(fg_target_pred.max(dim=-1, keepdim=True)[0])
                fg_target_mask = torch.cat(fg_target_mask, dim=-1) # [N_points, N_samples, N_objs]
                target_masks = torch.cat([bg_context_mask, fg_target_mask], dim=-1).permute(0, 2, 1)  # [N_points, N_objs+1, N_samples]
            else:
                bg_target_preds = torch.einsum("kd, mnd->kmn", mask_features, bg_target) # [N_points, N_samples, N_bg_query]
                bg_target_mask = bg_target_preds.max(dim=-1, keepdim=True)[0]  # [N_points, N_samples, 1]

                fg_target_preds =  torch.einsum("kd, mnd->kmn", mask_features, fg_target) # [N_points, N_samples, N_query]
                fg_target_preds = fg_target_preds.split(GT_query_split[1:], dim=-1)
                fg_target_mask = []
                for fg_target_pred in fg_target_preds:
                    fg_target_mask.append(fg_target_pred.max(dim=-1, keepdim=True)[0])
                fg_target_mask = torch.cat(fg_target_mask, dim=-1) # [N_points, N_samples, N_objs]
                target_masks = torch.cat([bg_target_mask, fg_target_mask], dim=-1).permute(0, 2, 1)  # [N_points, N_objs+1, N_samples]
            target_labels = target_masks.mean(dim=-1).argmax(1)

            # gt foreground click query attention mask
            gt_fg_attn_mask = []
            for gt_obj_id in range(1, fg_target_mask.size(-1)+1):
                gt_obj_mask = ~(target_labels == gt_obj_id)
                num_gt_points = GT_query_split[gt_obj_id]
                gt_obj_mask = gt_obj_mask.unsqueeze(0).repeat(num_gt_points, 1)
                gt_obj_mask[torch.where(gt_obj_mask.sum(-1) == gt_obj_mask.shape[-1])] = False
                gt_fg_attn_mask.append(gt_obj_mask)
            gt_fg_attn_mask = torch.cat(gt_fg_attn_mask, dim=0)

            # background click query attention mask
            gt_bg_attn_mask = ~(target_labels == 0)
            gt_bg_attn_mask = gt_bg_attn_mask.unsqueeze(0).repeat(GT_query_split[0], 1)
            gt_bg_attn_mask[torch.where(gt_bg_attn_mask.sum(-1) == gt_bg_attn_mask.shape[-1])] = False
            return torch.cat([fg_attn_mask, bg_attn_mask, gt_bg_attn_mask, gt_fg_attn_mask], dim=0), np_loss, context_masks, target_masks
        else:
            attn_mask = torch.cat([fg_attn_mask, bg_attn_mask], dim=0)   # [fg&bg click number, num_points]
            return attn_mask, np_loss, context_masks, None

    @torch.jit.unused
    def _set_aux_loss(self, outputs_seg_masks, all=False):
        if all:
            return [{"pred_masks": a} for a in outputs_seg_masks]
        else:
            return [{"pred_masks": a} for a in outputs_seg_masks[:-1]]


def build_npiseg3d(args):

    backbone = build_backbone(args)

    model = NPISeg3D(
                    backbone=backbone, 
                    hidden_dim=args.hidden_dim,
                    num_heads=args.num_heads, 
                    dim_feedforward=args.dim_feedforward,
                    shared_decoder=args.shared_decoder,
                    num_decoders=args.num_decoders, 
                    num_bg_queries=args.num_bg_queries,
                    dropout=args.dropout, 
                    pre_norm=args.pre_norm, 
                    positional_encoding_type=args.positional_encoding_type,
                    normalize_pos_enc=args.normalize_pos_enc,
                    hlevels=args.hlevels, 
                    voxel_size=args.voxel_size,
                    gauss_scale=args.gauss_scale,
                    aux=args.aux,
                    np_sampling_num=args.np_sampling_num,
                    num_gt_sampling=args.num_gt_clicks,
                    np_start_layer=args.np_start_layer,
                    scene_weight=args.scene_weight)

    return model
