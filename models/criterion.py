import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional
from torch import Tensor


class NormalizedFocalLossSoftmax(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, max_mult=-1, eps=1e-12,
                 from_logits=False, detach_delimeter=True,
                 weight=None, size_average=True,
                 ignore_label=-1):
        super(NormalizedFocalLossSoftmax, self).__init__()
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0

        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._max_mult = max_mult
        self._k_sum = 0
        self._m_max = 0

    def forward(self, pred, label):
        # Convert labels to one-hot encoding
        label_one_hot = F.one_hot(label, num_classes=pred.size(-1)).float()  # [N, classes]

        sample_weight = (label != self._ignore_label).float()   # [N]

        if not self._from_logits:
            pred = F.softmax(pred, dim=-1)

        alpha = torch.where(label_one_hot.bool(), self._alpha * sample_weight.unsqueeze(-1), (1 - self._alpha) * sample_weight.unsqueeze(-1))
        pt = torch.where(sample_weight.unsqueeze(-1).bool(), 1.0 - torch.abs(label_one_hot - pred), torch.ones_like(pred))

        beta = (1 - pt) ** self._gamma

        sw_sum = torch.sum(sample_weight)
        beta_sum = torch.sum(beta)
        mult = sw_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = beta * mult
        if self._max_mult > 0:
            beta = torch.clamp_max(beta, self._max_mult)

        with torch.no_grad():
            ignore_area = torch.sum(label == self._ignore_label).cpu().numpy()
            sample_mult = torch.mean(mult).cpu().numpy()
            if ignore_area == 0:
                self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult

                beta_pmax, _ = torch.max(beta, dim=0)
                beta_pmax = beta_pmax.mean().item()
                self._m_max = 0.8 * self._m_max + 0.2 * beta_pmax

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        loss = self._weight * (loss * sample_weight.unsqueeze(-1))

        if self._size_average:
            bsum = torch.sum(sample_weight)
            loss = torch.sum(loss) / (bsum + self._eps)
        else:
            loss = torch.sum(loss)

        return loss


class SetCriterion(nn.Module):

    def __init__(self, weight_dict, losses, alpha=0.25, gamma=2, max_mult=-1, eps=1e-12,
                 from_logits=False, detach_delimeter=True,
                 weight=None, size_average=True,
                 ignore_label=-1):
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses

        # parameters for normalized focal loss
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0

        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._max_mult = max_mult
        self._k_sum = 0
        self._m_max = 0

        # parameters for GHM-C loss
        self.bins = 10
        self.momentum = 0.75
        self.edges = torch.arange(self.bins + 1).float().cuda() / self.bins
        self.edges[-1] += 1e-6
        if self.momentum > 0:
            self.acc_sum = torch.zeros(self.bins).cuda()
        self.loss_weight = 1.0
        self.device = 'cuda'


    def dice_loss(self, input: Tensor, target: Tensor, ignore_mask: Optional[Tensor] = None, eps: Optional[float] = 1e-6):
        """
        Computes the DICE or soft IoU loss.
        :param input: tensor of shape [N, *]
        :param target: tensor with shape identical to input
        :param ignore_mask: tensor of same shape as input. non-zero values in this mask will be
        :param eps
        excluded from the loss calculation.
        :return: tensor
        """
        assert input.shape == target.shape, f"Shape mismatch between input ({input.shape}) and target ({target.shape})"
        assert input.dtype == target.dtype

        if torch.is_tensor(ignore_mask):
            assert ignore_mask.dtype == torch.bool
            assert input.shape == ignore_mask.shape, f"Shape mismatch between input ({input.shape}) and " \
                f"ignore mask ({ignore_mask.shape})"
            input = torch.where(ignore_mask, torch.zeros_like(input), input)
            target = torch.where(ignore_mask, torch.zeros_like(target), target)

        input = input.flatten(1)
        target = target.detach().flatten(1)

        numerator = 2.0 * (input * target).mean(1)
        denominator = (input + target).mean(1)

        soft_iou = (numerator + eps) / (denominator + eps)

        return torch.where(numerator > eps, 1. - soft_iou, soft_iou * 0.)

    def multiclass_dice_loss(self, input: Tensor, target: Tensor, eps: float = 1e-6,
                         check_target_validity: bool = True,
                         ignore_mask: Optional[Tensor] = None) -> Tensor:
        """
        Computes DICE loss for multi-class predictions. API inputs are identical to torch.nn.functional.cross_entropy()
        :param input: tensor of shape [N, C, *] with unscaled logits
        :param target: tensor of shape [N, *]
        :param eps:
        :param check_target_validity: checks if the values in the target are valid
        :param ignore_mask: optional tensor of shape [N, *]
        :return: tensor
        """
        assert input.ndim >= 2
        input = input.softmax(1)
        num_classes = input.size(1)

        if check_target_validity:
            class_ids = target.unique()
            assert not torch.any(torch.logical_or(class_ids < 0, class_ids >= num_classes)), \
                f"Number of classes = {num_classes}, but target has the following class IDs: {class_ids.tolist()}"

        target = torch.stack([target == cls_id for cls_id in range(0, num_classes)], 1).to(dtype=input.dtype)  # [N, C, *]


        if ignore_mask is not None:
            ignore_mask = ignore_mask.unsqueeze(1)
            expand_dims = [-1, input.size(1)] + ([-1] * (ignore_mask.ndim - 2))
            ignore_mask = ignore_mask.expand(*expand_dims)

        return self.dice_loss(input, target, eps=eps, ignore_mask=ignore_mask)

    def loss_NFL_multi_class(self, pred, label):
        """pred: 【num_points, num_class】
            label: [num_points]
        """
        # Convert labels to one-hot encoding
        label_one_hot = F.one_hot(label, num_classes=pred.size(-1)).float()  # [N, classes]

        sample_weight = (label != self._ignore_label).float()   # [N]

        if not self._from_logits:
            pred = F.softmax(pred, dim=-1)

        alpha = torch.where(label_one_hot.bool(), self._alpha * sample_weight.unsqueeze(-1), (1 - self._alpha) * sample_weight.unsqueeze(-1))
        pt = torch.where(sample_weight.unsqueeze(-1).bool(), 1.0 - torch.abs(label_one_hot - pred), torch.ones_like(pred))

        beta = (1 - pt) ** self._gamma

        sw_sum = torch.sum(sample_weight, dim=0, keepdim=True)
        beta_sum = torch.sum(beta, dim=0, keepdim=True)
        mult = sw_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = beta * mult
        if self._max_mult > 0:
            beta = torch.clamp_max(beta, self._max_mult)

        with torch.no_grad():
            ignore_area = torch.sum(label == self._ignore_label).cpu().numpy()
            sample_mult = torch.mean(mult).cpu().numpy()
            if ignore_area == 0:
                self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult

                beta_pmax, _ = torch.max(beta, dim=0)
                beta_pmax = beta_pmax.mean().item()
                self._m_max = 0.8 * self._m_max + 0.2 * beta_pmax

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        loss = self._weight * (loss * sample_weight.unsqueeze(-1))

        if self._size_average:
            bsum = torch.sum(sample_weight)
            loss = torch.sum(loss) / (bsum + self._eps)
        else:
            loss = torch.sum(loss)

        return loss

    def loss_NFL(self, output, target):
        """Compute the binary cross-entropy loss."""
        pred_masks = output['pred_masks']

        loss = 0.0
        for i in range(len(pred_masks)):
            loss_sample = self.loss_NFL_multi_class(pred_masks[i], target[i].long()).mean()
            loss += loss_sample

        loss /= len(pred_masks)

        return {"loss_NFL": loss}

    def loss_bce(self, outputs, targets, weights=None, aux_coff=1.0):
        """Compute the binary cross-entropy loss.
        """
        pred_masks = outputs['pred_masks']

        loss = 0.0
        # iterate over the batch
        for i in range(len(pred_masks)):
            if pred_masks[i].ndim>2:
                loss_sample = (F.cross_entropy(pred_masks[i].permute(2, 1, 0), targets[i].unsqueeze(0).repeat(pred_masks[i].size(-1), 1).long(), reduction="none") * weights[i]).mean()
            else:
                loss_sample = (F.cross_entropy(pred_masks[i], targets[i].long(), reduction="none") * weights[i]).mean()
            loss += loss_sample

        loss = loss/len(pred_masks)
        loss = loss * aux_coff

        return {"loss_bce": loss}

    def loss_dice(self, outputs, targets, weights=None, aux_coff=1.0):
        """compute the dice loss."""
        pred_masks = outputs['pred_masks']
        loss = 0.0
        # iterate over the batch
        for i in range(len(pred_masks)):   #[N, C, B]
            if pred_masks[i].ndim>2:
                n_samples = pred_masks[0].size(0)
                loss_sample = (self.multiclass_dice_loss(pred_masks[i], targets[i].unsqueeze(1).repeat(1, pred_masks[i].size(-1)).long()) * weights[i]).mean()
            else:
                loss_sample = (self.multiclass_dice_loss(pred_masks[i], targets[i].long()) * weights[i]).mean()
            loss += loss_sample
            
        loss = loss/len(pred_masks)
        loss = loss * aux_coff
        return {
            "loss_dice": loss
        }

    def get_loss(self, loss, outputs, targets, weights=None, aux_coff=1.0):
        """Get all loss, i.e., dice, bce."""
        loss_map = {
            'bce': self.loss_bce,
            'dice': self.loss_dice,
            'NFL': self.loss_NFL,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        if loss == 'NFL':
            return self.loss_NFL(outputs, targets)
        else:
            return loss_map[loss](outputs, targets, weights, aux_coff)

    def forward(self, outputs, targets, weights=None):
        """Compute the loss."""
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Compute all the requested losses, dice, bce.
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, weights, aux_coff=1.0))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, weights, aux_coff=0.5)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        if 'aux_target_masks' in outputs:
            for i, aux_target_masks in enumerate(outputs['aux_target_masks']):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_target_masks, targets, weights, aux_coff=0.5)
                    l_dict = {k + f'_{i+3}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        if 'aux_prob_target_mask' in outputs:
            for i, aux_prob_target_mask in enumerate(outputs['aux_prob_target_mask']):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_prob_target_mask, targets, weights, aux_coff=1.0)
                    l_dict = {k + f'_{i+6}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def build_mask_criterion(args):

    weight_dict = {}
    losses = args.losses
    for i, loss in enumerate(losses):
        loss_name = f'loss_{loss}'
        loss_coef = args.loss_coef[i]
        weight_dict.update({loss_name: loss_coef})

    # aux loss for each transformer decoder
    if args.aux:
        aux_weight_dict = {}
        for i in range(args.num_decoders*len(args.hlevels)+4):  # TODO target branch 3
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(weight_dict, losses)

    return criterion


if __name__ == '__main__':

    criterion = NormalizedFocalLossSoftmax()

    pred = torch.randn(5, 3, requires_grad=True)  # 预测值
    label = torch.tensor([0, 1, 2, 1, 0])  # 标签

    simple_pred = torch.tensor([[2.0, 1.0, 0.1],
                                [0.1, 2.0, 1.0],
                                [0.1, 1.0, 2.0],
                                [1.0, 2.0, 0.1],
                                [2.0, 0.1, 1.0]], requires_grad=True)
    simple_label = torch.tensor([0, 1, 2, 1, 0])

    # 困难样本：预测概率与标签差距较大
    hard_pred = torch.tensor([[0.1, 1.0, 2.0],
                              [2.0, 0.1, 1.0],
                              [1.0, 2.0, 0.1],
                              [0.1, 2.0, 1.0],
                              [1.0, 0.1, 2.0]], requires_grad=True)
    hard_label = torch.tensor([0, 1, 2, 1, 0])

    # 计算简单样本的损失
    simple_loss = criterion(simple_pred, simple_label)
    print("Simple Loss:", simple_loss.item())

    # 计算困难样本的损失
    hard_loss = criterion(hard_pred, hard_label)
    print("Hard Loss:", hard_loss.item())

    print(1)


