# ------------------------------------------------------------------------
# Yuanwen Yue
# ETH Zurich
# ------------------------------------------------------------------------
import torch
import numpy as np
import random


def mean_iou_single(pred, labels):
    """Calculate the mean IoU for a single object
    """
    truepositive = pred*labels
    intersection = torch.sum(truepositive==1)
    uni = torch.sum(pred==1) + torch.sum(labels==1) - intersection

    iou = intersection/uni
    return iou

def mean_iou(pred, labels):
    """Calculate the mean IoU for a batch
    """
    assert len(pred) == len(labels)
    bs = len(pred)
    iou_batch = 0.0
    for b in range(bs):
        pred_sample = pred[b]
        labels_sample = labels[b]
        obj_ids = torch.unique(labels_sample)
        obj_ids = obj_ids[obj_ids!=0]
        obj_num = len(obj_ids)
        iou_sample = 0.0
        for obj_id in obj_ids:
            obj_iou = mean_iou_single(pred_sample==obj_id, labels_sample==obj_id)
            iou_sample += obj_iou

        iou_sample /= obj_num

        iou_batch += iou_sample
    
    iou_batch /= bs
    return iou_batch

def mean_iou_scene(pred, labels):
    """Calculate the mean IoU for all target objects in the scene
    """
    obj_ids = torch.unique(labels)
    obj_ids = obj_ids[obj_ids!=0]
    obj_num = len(obj_ids)
    iou_sample = 0.0
    iou_dict = {}
    for obj_id in obj_ids:
        obj_iou = mean_iou_single(pred==obj_id, labels==obj_id)
        iou_dict[int(obj_id)] = float(obj_iou)
        iou_sample += obj_iou

    iou_sample /= obj_num

    return iou_sample, iou_dict


def loss_weights(points, clicks, tita, alpha, beta):
    """Points closer to clicks have bigger weights. Vice versa.
    points: [N, 3], coordinates of the point cloud
    clicks: [M, 3], coordinates of the clicks
    """
    pairwise_distances = torch.cdist(points, clicks)
    pairwise_distances, _ = torch.min(pairwise_distances, dim=1)

    weights = alpha + (beta-alpha) * (1 - torch.clamp(pairwise_distances, max=tita)/tita)

    return weights

def cal_click_loss_weights(batch_idx, raw_coords, labels, click_idx, alpha=0.8, beta=2.0, tita=0.3):
    """Calculate the loss weights for each point in the point cloud.
    """
    weights = []

    bs = batch_idx.max() + 1
    for i in range(bs):
         # click idx of sample i, dict {obj_id: [click_idx]}
        click_idx_sample = click_idx[i]
        # get the coord of smaple i [M, 3]-->[N, 3]
        sample_mask = batch_idx == i   
        raw_coords_sample = raw_coords[sample_mask]
        # merge click idx of all objects
        all_click_idx = [np.array(v) for k, v in click_idx_sample.items()]
        all_click_idx = np.hstack(all_click_idx).astype(np.int64).tolist()
        # click coordinates of sample i
        click_points_sample = raw_coords_sample[all_click_idx]
        weights_sample = loss_weights(raw_coords_sample, click_points_sample, tita, alpha, beta)
        weights.append(weights_sample)

    return weights


def get_next_click_coo_torch(discrete_coords, unique_labels, gt, pred, pairwise_distances, uncertainty=None):
    """Sample the next click from the center of the error region
    Args:
        discrete_coords: point cloud coordinates [N, 3]
        unique_labels: binary mask indicating foreground (1) and background (0)
        gt: ground truth labels
        pred: predicted labels
        pairwise_distances: distances from each foreground point to nearest background point
        uncertainty: uncertainty of each point (optional)
    Returns:
        center_global_id: global index of the selected point
        center_coo: coordinates of the selected point
        center_label: ground truth label of the selected point
        max_dist: maximum distance to boundary
        candidates: coordinates of all foreground points
    """
    # Split points into background (0) and foreground (1)
    zero_indices = (unique_labels == 0)  # boolean mask for background points
    one_indices = (unique_labels == 1)   # boolean mask for foreground points
    
    # Return None if either background or foreground is empty
    if zero_indices.sum() == 0 or one_indices.sum() == 0:
        return None, None, None, -1, None, None

    # If uncertainty is None or not provided, use only distance
    if uncertainty is None:
        weighted_distances = pairwise_distances
    else:
        # normalize uncertainty
        min_uncertainty, max_uncertainty = uncertainty.min(), uncertainty.max()
        normalized_uncertainty = (uncertainty - min_uncertainty) / (max_uncertainty - min_uncertainty + 1e-6)
        weighted_distances = pairwise_distances * normalized_uncertainty  # uncertainty as weight
    
    center_id = torch.where(weighted_distances == torch.max(weighted_distances, dim=0)[0])
    
    # Get coordinates and labels of the center point
    center_coo = discrete_coords[one_indices, :][center_id[0][0]]      # 3D coordinates of center point
    center_label = gt[one_indices][center_id[0][0]]                    # ground truth label of center point
    center_pred = pred[one_indices][center_id[0][0]]                   # predicted label of center point

    # Calculate global index
    local_mask = torch.zeros(pairwise_distances.shape[0], device=discrete_coords.device)
    global_id_mask = torch.zeros(discrete_coords.shape[0], device=discrete_coords.device)
    local_mask[center_id] = 1
    global_id_mask[one_indices] = local_mask
    center_global_id = torch.argwhere(global_id_mask)[0][0]

    candidates = discrete_coords[one_indices, :]
    max_dist = torch.max(pairwise_distances)

    return center_global_id, center_coo, center_label, max_dist, candidates

def get_next_simulated_click_multi(error_cluster_ids, error_cluster_ids_mask, pred_qv, labels_qv, coords_qv, error_distances, uncertainty):
    """Sample the next clicks for each error region
    """

    click_dict = {}
    new_click_pos = {}
    random.shuffle(error_cluster_ids)

    for cluster_id in error_cluster_ids:

        error = error_cluster_ids_mask == cluster_id
        pair_distances = error_distances[cluster_id]
        
        # uncertainty score
        if uncertainty is not None:
            uncertainty_distances = uncertainty[error]
        else:
            uncertainty_distances = None
        # get next click candidate
        center_id, center_coo, center_gt, max_dist, candidates = get_next_click_coo_torch(
            coords_qv, error,
            labels_qv, pred_qv, pair_distances, uncertainty_distances)

        if click_dict.get(str(int(center_gt))) == None:
            click_dict[str(int(center_gt))] = [int(center_id)]
            new_click_pos[str(int(center_gt))] = [center_coo]
        else:
            click_dict[str(int(center_gt))].append(int(center_id))
            new_click_pos[str(int(center_gt))].append(center_coo)
    
    click_num = len(error_cluster_ids)

    return click_dict, click_num, new_click_pos


def measure_error_size(discrete_coords, unique_labels):
    """Measure error size in 3D space
    """

    zero_indices = (unique_labels == 0)  # background
    one_indices = (unique_labels == 1)  # foreground
    if zero_indices.sum() == 0 or one_indices.sum() == 0:
        return None, None, None, -1, None, None

    # All distances from foreground points to background points
    pairwise_distances = torch.cdist(discrete_coords[zero_indices, :], discrete_coords[one_indices, :])
    # Bg points on the border， e.g., find closest bg points
    pairwise_distances, _ = torch.min(pairwise_distances, dim=0)

    return pairwise_distances


def calculate_pairwise_dist(discrete_coords, unique_labels):
    zero_indices = (unique_labels == 0)  # background
    one_indices = (unique_labels == 1)  # foreground

    # fg & bg coords
    bg_coords = discrete_coords[zero_indices, :]
    fg_coords = discrete_coords[one_indices, :]

    # fg-bg distance
    fg_bg_distance = torch.cdist(bg_coords, fg_coords)

    return fg_bg_distance


def get_simulated_clicks(pred_qv, labels_qv, coords_qv, uncertainty, current_num_clicks=0, training=True):
    """Sample simulated clicks. 
    The simulation samples next clicks from the top biggest error regions in the current iteration.
    """
    # label, pred, error points
    labels_qv = labels_qv.float()
    pred_label = pred_qv.float()
    error_mask = torch.abs(pred_label - labels_qv) > 0
    if error_mask.sum() == 0:
        return None, None, None

    # unique id for each point
    cluster_ids = labels_qv * 96 + pred_label * 11  # e.g., label [0,1,2,3,4] pred[0,1,2], 12 clusters, N dimension

    error_region = coords_qv[error_mask]  # [N_error, 3]

    # number of objects in the scene
    num_obj = (torch.unique(labels_qv) != 0).sum()

    # error cluster, id, number
    error_clusters = cluster_ids[error_mask]  # [N_error]
    error_cluster_ids = torch.unique(error_clusters)  # e.g., 12
    num_error_cluster = len(error_cluster_ids)  # e.g., 12
    # set cluster in original points
    error_cluster_ids_mask = torch.ones(coords_qv.shape[0], device=coords_qv.device) * -1  # [N_points] e.g., 33856
    error_cluster_ids_mask[error_mask] = error_clusters   # [N_points] e.g., 33856

    ### measure the size of each error cluster and store the distance
    error_sizes = {}
    error_distances = {}

    for cluster_id in error_cluster_ids:
        error = error_cluster_ids_mask == cluster_id
        pairwise_distances = measure_error_size(coords_qv, error)

        error_distances[int(cluster_id)] = pairwise_distances
        error_sizes[int(cluster_id)] = torch.max(pairwise_distances).tolist()

    error_cluster_ids_sorted = sorted(error_sizes, key=error_sizes.get, reverse=True)

    if training:
        if num_error_cluster >= num_obj:  # cluster is more, set num_obj as maximum
            selected_error_cluster_ids = error_cluster_ids_sorted[:num_obj]
        else:
            selected_error_cluster_ids = error_cluster_ids_sorted
    else:  # choose largest error cluster during evaluation
        if current_num_clicks == 0:
            selected_error_cluster_ids = error_cluster_ids_sorted
        else:
            selected_error_cluster_ids = error_cluster_ids_sorted[:1]

    new_clicks, new_click_num, new_click_pos = get_next_simulated_click_multi(selected_error_cluster_ids, error_cluster_ids_mask, pred_qv, labels_qv, coords_qv, error_distances, uncertainty)
 
    return new_clicks, new_click_num, new_click_pos


def extend_clicks(current_clicks,  new_clicks):
    """Append new click to existing clicks
    """
    for obj_id, click_ids in new_clicks.items():
        if obj_id not in current_clicks:
            print(current_clicks)
            print('\n')
            print(new_clicks)
        current_clicks[obj_id].extend(click_ids)

    return current_clicks


def is_valid_click(clicks, new_click, min_distance):
    if len(clicks) == 0:
        return True
    distances = torch.cdist(torch.stack(clicks).unsqueeze(0), new_click.unsqueeze(0)).squeeze()
    return torch.all(distances >= min_distance)


"""repeatly sample clicks for each object"""
def random_sampling_clicks(sample_labels, sample_raw_coords, pos_click_num, neg_click_num):
    num_objects = (torch.unique(sample_labels) != 0).sum()

    click_dict = {}

    # negative clicks sampling
    potential_neg_indices = torch.where(sample_labels == 0)[0]

    neg_clicks_indices = []
    while len(neg_clicks_indices) < neg_click_num:
        if len(potential_neg_indices) == 0:
            break
        idx = torch.randint(0, len(potential_neg_indices), (1,)).item()
        neg_clicks_indices.append(potential_neg_indices[idx].item())

    click_dict['0'] = neg_clicks_indices

    # positive clicks sampling
    for label in range(1, num_objects + 1):
        # instance foreground and background
        instance_label = torch.ones(sample_labels.shape[0], device=sample_labels.device) * -1
        instance_label[sample_labels == label] = 1
        instance_label[sample_labels != label] = 0

        # pairwise distance between foreground and background
        fg_bg_distance = calculate_pairwise_dist(sample_raw_coords, instance_label)
        fg_distance, _ = torch.min(fg_bg_distance, dim=0)

        # Find the foreground points
        fg_indices = torch.where(instance_label == 1)[0]

        # Positive clicks
        pos_clicks_indices = []

        #center_distance = torch.max(fg_distance).item()
        
        # object center as the first click
        max_fg_distance_idx = torch.argmax(fg_distance)
        max_fg_click_idx = fg_indices[max_fg_distance_idx].item()
        pos_clicks_indices.append(max_fg_click_idx)

        while len(pos_clicks_indices) < pos_click_num[label - 1]:
            idx = torch.randint(0, len(fg_indices), (1,)).item()
            pos_clicks_indices.append(fg_indices[idx].item())

        click_dict[str(label)] = pos_clicks_indices

    return click_dict
