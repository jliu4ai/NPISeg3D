# ------------------------------------------------------------------------
# Jie Liu
# University of Amsterdam
# The project is modified from AGILE3D，thanks to auhtors!
# ------------------------------------------------------------------------

import copy
import json
import math
import os
import sys
import time
from typing import Iterable


import numpy as np
import random
import MinkowskiEngine as ME
import wandb
import torch
import torch.nn.functional as F

from utils.seg import mean_iou, mean_iou_scene, cal_click_loss_weights, extend_clicks, get_simulated_clicks, random_sampling_clicks
import utils.misc as utils

from evaluation.evaluator_MO import EvaluatorMO

# TODO: add comments
def kl_scheduler(epoch, min_coeff=0.0005, max_coeff=0.005, epoch_max=600, step=1):
    total_steps = epoch_max // step
    increment_per_step = (max_coeff - min_coeff) / total_steps
    step_index = min(epoch // step, total_steps)
    return min_coeff + step_index * increment_per_step


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, train_total_iter, max_norm, args):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = 10

    for i, batched_inputs in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        coords, raw_coords, feats, labels, _, _, click_idx, scene_name, num_obj = batched_inputs
        coords = coords.to(device)
        labels = [l.to(device) for l in labels]
        labels_new = []
        raw_coords = raw_coords.to(device)
        feats = feats.to(device)
        batch_idx = coords[:, 0]  # [0-batchsize]

        data = ME.SparseTensor(coordinates=coords, features=feats, device=device)

        # get backbone feature [bsxN,128], aux feature, [bs,128]
        pcd_features, aux, coordinates, pos_encodings_pcd = model.forward_backbone(data, raw_coordinates=raw_coords)

        # 1. random sample obj number and obj index
        for idx in range(batch_idx.max()+1):
            sample_mask = batch_idx == idx
            sample_labels = labels[idx]
            sample_raw_coords = raw_coords[sample_mask]
            valid_obj_idxs = torch.unique(sample_labels)
            valid_obj_idxs = valid_obj_idxs[valid_obj_idxs!=-1]

            max_num_obj = len(valid_obj_idxs)

            # Attention, label mays has no background points
            num_obj = np.random.randint(1, min(10, max_num_obj))
            obj_idxs = valid_obj_idxs[torch.randperm(max_num_obj)[:num_obj]]
            sample_labels_new = torch.zeros(sample_labels.shape[0], device=device)
            # relabel pc
            for i, obj_id in enumerate(obj_idxs):
                obj_mask = sample_labels == obj_id
                sample_labels_new[obj_mask] = i+1
                # each scene, each object
                click_idx[idx][str(i+1)] = []

            click_idx[idx]['0'] = []
            labels_new.append(sample_labels_new)

            if 0 not in torch.unique(sample_labels_new):
                raise ValueError('No background points in the scene')

        # 2. randomly select positive and negative clicks
        max_pos_clicks = 3
        max_neg_clicks = 5  # TODO maybe an important hyperparameter
        GT_click_idx = [{}]*(batch_idx.max() + 1)
        for idx in range(batch_idx.max() + 1):
            sample_labels = labels_new[idx]
            sample_obj_num = (torch.unique(sample_labels) != 0).sum()
            pos_click_num = [random.randint(1, max_pos_clicks) for x in range(sample_obj_num)]
            neg_click_num = random.randint(1, max_neg_clicks)

            sample_mask = batch_idx == idx
            sample_raw_coords = raw_coords[sample_mask]

            #  randomly initialize clicks for each point cloud，
            click_idx[idx] = random_sampling_clicks(sample_labels, sample_raw_coords, pos_click_num, neg_click_num)

            # GT click
            GT_pos_click_num = [args.num_gt_clicks]*len(pos_click_num)
            GT_click_idx[idx] = random_sampling_clicks(sample_labels, sample_raw_coords, GT_pos_click_num, args.num_gt_clicks)
        
        # if there is no background points in the scene, skip this sample
        if 0 not in torch.unique(sample_labels):
            continue

        # 3. iterative click sampling
        max_iters = 5
        num_iters = random.randint(0, max_iters)
        current_num_iter = 0

        with torch.no_grad():
            model.eval()
            eval_model = model
            while  current_num_iter <= num_iters:
                if current_num_iter == 0:
                    pred = [torch.zeros(l.shape).to(device) for l in labels]
                    prev_mask = [torch.zeros(l.shape[0], 15).to(device) for l in labels]
                else:
                    outputs = eval_model.forward_mask(pcd_features, aux, coordinates, pos_encodings_pcd, click_idx=click_idx, prev_mask=prev_mask, GT_click=GT_click_idx)
                    pred_logits = outputs['pred_masks']
                    pred = [p.mean(dim=-1).argmax(1) for p in pred_logits]  # [batch, N]
                    one_hot_pred = [F.one_hot(p) for p in pred]
                    for i in range(len(prev_mask)):
                        prev_mask[i][:, :one_hot_pred[i].size(1)] = one_hot_pred[i]

                # process for each sample
                for idx in range(batch_idx.max()+1):
                    sample_mask = batch_idx == idx
                    sample_pred = pred[idx]

                    # update prediction with sparse gt, object id, click id
                    for obj_id, cids in click_idx[idx].items():
                        sample_pred[cids] = int(obj_id)

                    sample_labels = labels_new[idx]
                    sample_raw_coords = raw_coords[sample_mask]
                    # get new clicks
                    new_clicks, new_clicks_num, new_click_pos = get_simulated_clicks(sample_pred, sample_labels, sample_raw_coords, current_num_iter, training=True)

                    ### add new clicks ###
                    if new_clicks is not None:
                        for obj_id, click_ids in new_clicks.items():
                            if obj_id not in click_idx[idx]:
                                print(click_idx[idx])
                                print('\n')
                                print(new_clicks)
                                print('\n')
                                print((torch.unique(sample_labels) != 0).sum())
                            click_idx[idx][obj_id].extend(click_ids)
                
                current_num_iter += 1
                
        # 3. real forward pass with loss back propagation
        model.train() 
        # pcd_features: [N*bs, 128], aux: 5 intermediate feats, coordinates: [N*bs, 3]
        outputs = model.forward_mask(pcd_features, aux, coordinates, pos_encodings_pcd, click_idx=click_idx, prev_mask=prev_mask, GT_click=GT_click_idx)

        # computer loss # weights for each point, point closer to click point, larger weight
        click_weights = cal_click_loss_weights(coords[:, 0], raw_coords, torch.cat(labels_new), click_idx)
        loss_dict = criterion(outputs, labels_new, click_weights)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        kl_weights = kl_scheduler(epoch)

        losses+= kl_weights*outputs['np_loss']

        # adding np divergence loss here for logging
        loss_dict.update({'divergence': outputs['np_loss']})
        weight_dict.update({'divergence': kl_weights})

        # reduce loss for distributed training, can be ignored
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_dict_reduced_summary = {}
        loss_dict_reduced_summary['loss_bce'] = loss_dict_reduced_scaled['loss_bce']
        loss_dict_reduced_summary['loss_dice'] = loss_dict_reduced_scaled['loss_dice']
        loss_dict_reduced_summary['divergence'] = loss_dict_reduced_scaled['divergence']
        loss_dict_reduced_summary['deter_C_bce'] = sum([loss_dict_reduced_scaled[f'loss_bce_{i}'] for i in range(3)])
        loss_dict_reduced_summary['deter_C_dice'] = sum([loss_dict_reduced_scaled[f'loss_dice_{i}'] for i in range(3)])
        loss_dict_reduced_summary['deter_T_bce'] = sum([loss_dict_reduced_scaled[f'loss_bce_{i}'] for i in range(3, 6)])
        loss_dict_reduced_summary['deter_T_dice'] = sum([loss_dict_reduced_scaled[f'loss_dice_{i}'] for i in range(3, 6)])
        loss_dict_reduced_summary['prob_T_bce'] = loss_dict_reduced_scaled['loss_bce_6']
        loss_dict_reduced_summary['prob_T_dice'] = loss_dict_reduced_scaled['loss_dice_6']


        loss_value = losses_reduced_scaled.item()
        # in case of gradient explosion
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        train_total_iter+=1
        # optimizer step and gradient clip
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()
        
        # release memory
        torch.cuda.empty_cache()

        # record miou, grad_norm, loss, lr; pred/label_new: [bs, N]
        with torch.no_grad():
            pred_logits = outputs['pred_masks']
            pred = [p.mean(dim=-1).argmax(1) for p in pred_logits]
            metric_logger.update(mIoU=mean_iou(pred, labels_new))

            metric_logger.update(grad_norm=grad_total_norm)
            metric_logger.update(loss=loss_value, **loss_dict_reduced_summary)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
 
        # log to wandb
        if ((i + 1) % 100 == 0):
            wandb.log({
                "train/loss": metric_logger.meters['loss'].avg,
                "train/loss_bce": metric_logger.meters['loss_bce'].avg,
                "train/loss_dice": metric_logger.meters['loss_dice'].avg,
                "train/divergence": metric_logger.meters['divergence'].avg,
                #'train/loss_NFL': metric_logger.meters['loss_NFL'].avg,
                "train/mIoU": metric_logger.meters['mIoU'].avg,
                "train/total_iter": train_total_iter
                })

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, train_total_iter
    

@torch.no_grad()
def evaluate(model, criterion, data_loader, args, epoch, device):
    model.eval()
    criterion.eval()
    # metric logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # save val results
    instance_counter = 0
    results_file = os.path.join(args.valResults_dir, 'val_results_epoch_' + str(epoch) + '.csv')
    f = open(results_file, 'w')

    for batched_inputs in metric_logger.log_every(data_loader, 10, header):
        # coords: [N, 4], raw_coords: [N, 3], feats: [N, 128], labels: [N], labels_full: [N*], inverse_map: [N*], click_idx: [bs, dict], scene_name: [bs], num_obj: [bs]
        coords, raw_coords, feats, labels, labels_full, inverse_map, click_idx, scene_name, num_obj = batched_inputs
        coords = coords.to(device)
        raw_coords = raw_coords.to(device)
        labels = [l.to(device) for l in labels]
        labels_full = [l.to(device) for l in labels_full]
        # minkowski sparse tensor
        data = ME.SparseTensor(
                                coordinates=coords,
                                features=feats,
                                device=device
                                )

        ###### interactive evaluation ######
        batch_idx = coords[:, 0]

        # click ids set null, click_idx_sample, dict {obj_id: [click_idx]}
        for click_idx_sample in click_idx:
            for obj_id, _ in click_idx_sample.items():
                click_idx_sample[obj_id] = []

        current_num_clicks = 0

        # pre-compute backbone features only once
        pcd_features, aux, coordinates, pos_encodings_pcd = model.forward_backbone(data, raw_coordinates=raw_coords)
        # max_num_clicks: num_obj*20
        max_num_clicks = num_obj[0] * args.max_num_clicks 

        while current_num_clicks <= max_num_clicks:  
            # get prediction first
            if current_num_clicks == 0:
                pred = [torch.zeros(l.shape).to(device) for l in labels]
                prev_mask = [torch.zeros(l.shape[0], 15).to(device) for l in labels]
            else:
                outputs = model.forward_mask(pcd_features, aux, coordinates, pos_encodings_pcd, click_idx=click_idx, prev_mask=prev_mask, GT_click=None)
                pred_logits = outputs['pred_masks']
                pred = [p.mean(dim=-1).argmax(1) for p in pred_logits]
                one_hot_pred = [F.one_hot(p) for p in pred]
                for i in range(len(prev_mask)):
                    prev_mask[i][:, :one_hot_pred[i].size(1)] = one_hot_pred[i]
            # calculate loss for each sample
            if current_num_clicks != 0:
                click_weights = cal_click_loss_weights(batch_idx, raw_coords, torch.cat(labels), click_idx)
                loss_dict = criterion(outputs, labels, click_weights)
                weight_dict = criterion.weight_dict
                
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
                loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}

            updated_pred = []
            for idx in range(batch_idx.max()+1):
                sample_mask = batch_idx == idx
                sample_pred = pred[idx]
                # get sample features 
                sample_mask = sample_mask.to(feats.device)
                # update prediction with sparse gt
                if current_num_clicks != 0:
                    for obj_id, cids in click_idx[idx].items():
                        sample_pred[cids] = int(obj_id)
                    updated_pred.append(sample_pred)
                
                # get sample labels, raw coords, pred_full, labels_full
                sample_labels = labels[idx]
                sample_raw_coords = raw_coords[sample_mask]
                sample_pred_full = sample_pred[inverse_map[idx]]
                sample_labels_full = labels_full[idx]

                # calculate iou with full labels
                sample_iou, _ = mean_iou_scene(sample_pred_full, sample_labels_full)
                
                # save key results in csv
                line = str(instance_counter+idx) + ' ' + scene_name[idx].replace('scene','') + ' ' + str(num_obj[idx]) + ' ' + str(current_num_clicks/num_obj[idx]) + ' ' + str(
                sample_iou.cpu().numpy()) + '\n'
                f.write(line)
                print(scene_name[idx], 'Object: ', num_obj[idx], 'num clicks: ', current_num_clicks/num_obj[idx], 'IOU: ', sample_iou.item())
                
                # get new clicks from pred and labels
                new_clicks, new_clicks_num, new_click_pos = get_simulated_clicks(sample_pred, sample_labels, sample_raw_coords, current_num_clicks, training=False)

                ### add new clicks ###
                if new_clicks is not None:
                    click_idx[idx] = extend_clicks(click_idx[idx], new_clicks)

            # upate metric, i.e, ioU, loss
            if current_num_clicks != 0:
                metric_logger.update(mIoU=mean_iou(updated_pred, labels))
                metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                                    **loss_dict_reduced_scaled,
                                    **loss_dict_reduced_unscaled)

            # record click number
            if current_num_clicks == 0:
                new_clicks_num = num_obj[idx]
            else:
                new_clicks_num = 1
            current_num_clicks += new_clicks_num
        
        # update instance counter
        instance_counter += len(num_obj)

    f.close()
    # evaluate metrics
    evaluator = EvaluatorMO(args.val_list, results_file, [0.5,0.65,0.8,0.85,0.9])
    results_dict = evaluator.eval_results()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats.update(results_dict)

    return stats
