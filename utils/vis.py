import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
import json
from utils.ply import read_ply


fixed_colors = {
    0: np.array([0.85, 0.85, 0.85]),  # Background (Light Gray)
    1: np.array([0.3, 1.0, 0.3]),     # Bright GREEN
    2: np.array([1.0, 0.3, 0.3]),     # Bright RED
    3: np.array([0.3, 0.3, 1.0]),     # Bright Blue
    4: np.array([1.0, 0.8, 0.1]),     # Bright Yellow
    5: np.array([1.0, 0.3, 1.0]),     # Bright Magenta
    6: np.array([0.3, 1.0, 1.0]),     # Bright Cyan
    7: np.array([1.0, 0.6, 0.2]),     # Bright Orange
    8: np.array([0.7, 0.3, 1.0]),     # Bright Purple
    9: np.array([0.3, 1.0, 0.6]),     # Bright Spring Green
    10: np.array([1.0, 0.5, 0.7]),    # Bright Pink
}

def visualize_all_instances(points, instance_labels):
        # 获取唯一的实例标签
    unique_labels = np.unique(instance_labels)
    
    # 创建颜色映射
    colormap = plt.get_cmap('tab20')  # 使用tab20色彩映射,最多支持20种不同颜色
    
    # 为每个实例创建一个点云对象并赋予不同颜色
    vis = o3d.visualization.Visualizer()
    vis.create_window("Instance Segmentation")
    
    # 设置渲染选项
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])  # 黑色背景
    opt.point_size = 0.1  # 设置点大小
    
    # 为每个实例创建点云并添加到可视化器
    for i, label in enumerate(unique_labels):
        # 获取当前实例的点
        mask = (instance_labels == label)
        instance_points = points[mask]
        
        # 创建点云对象
        instance_pcd = o3d.geometry.PointCloud()
        instance_pcd.points = o3d.utility.Vector3dVector(instance_points)
        
        # 为当前实例分配颜色
        color = colormap(i % 20)[:3]  # 取RGB值，不要alpha通道
        instance_colors = np.tile(color, (len(instance_points), 1))
        instance_pcd.colors = o3d.utility.Vector3dVector(instance_colors)
        
        # 添加到可视化器
        vis.add_geometry(instance_pcd)
    
    # 调整视角
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    # 运行可视化器
    vis.run()
    vis.destroy_window()

def visualize_instance(pcd, instance_labels, target_id):
    """
    可视化指定ID的实例, 其他实例显示为半透明
    Args:
        pcd: Open3D点云对象
        instance_labels: 实例标签数组
        target_id: 要高亮显示的实例ID
    """
    print(f"\n=== 显示实例 ID: {target_id} ===")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(f"Instance ID: {target_id}")
    
    # 设置渲染选项 - 在添加几何体之前设置
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])  # 黑色背景
    opt.point_size = 0.1  # 设置默认点大小
    
    # 准备点云数据
    points = np.asarray(pcd.points)
    mask = (instance_labels == target_id)
    target_points = points[mask]
    other_points = points[~mask]
    
    # 1. 添加背景点云
    pcd_others = o3d.geometry.PointCloud()
    pcd_others.points = o3d.utility.Vector3dVector(other_points)
    colors_others = np.ones((len(other_points), 3)) * np.array([0.5, 0.5, 0.5])  # 灰色
    pcd_others.colors = o3d.utility.Vector3dVector(colors_others)
    vis.add_geometry(pcd_others)
    
    # 2. 添加目标实例点云
    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(target_points)
    colors_target = np.ones((len(target_points), 3)) * np.array([1, 0, 0])  # 红色
    pcd_target.colors = o3d.utility.Vector3dVector(colors_target)
    
    # 增加目标点的大小
    opt.point_size = 0.1  # 设置更大的点大小
    vis.add_geometry(pcd_target)
    
    # 打印实例信息
    points_count = len(target_points)
    center = np.mean(target_points, axis=0)
    print(f"实例信息:")
    print(f"- 点数量: {points_count}")
    print(f"- 中心位置: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    
    # 调整视角
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    # 运行可视化器
    vis.run()
    vis.destroy_window()


def compute_uncertainty(predictions):
    """
    Compute uncertainty based on variance of maximum probabilities across MC samples
    Args:
        predictions: [[N, K, 10]] predicted logits where:
            N: number of points
            K: number of classes
            10: number of Monte Carlo samples
    Returns:
        uncertainty: [N] uncertainty scores based on variance of max probabilities
    """
    if predictions is not None:
        predictions = predictions[0]  # [N, K, 10]
        
        # Convert to probabilities for each MC sample
        probs = torch.softmax(predictions, dim=1)  # [N, K, 10]
        # Get maximum probability for each MC sample
        max_probs = torch.max(probs, dim=1)[0]  # [N, 10]
        
        # Compute variance across MC samples
        uncertainty = torch.var(max_probs, dim=1)  # [N]
        
    else:
        uncertainty = None
    
    return uncertainty


def visualize_uncertainty_and_mask(point_cloud, predictions, segmentation_mask):
    """
    Visualize both uncertainty map and segmentation mask in one window
    Args:
        point_cloud: [N, 3] point cloud coordinates
        predictions: [N, K, 10] predicted mask probabilities across samples
        segmentation_mask: [N] segmentation labels
    """
    # 计算不确定性
    predictions = predictions.permute(0, 2, 1)
    predictions = torch.softmax(predictions, dim=-1)
    mean_probs = predictions.mean(dim=1)
    entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)

    # 归一化熵值
    min_entropy, max_entropy = entropy.min(), entropy.max()
    normalized_entropy = (entropy - min_entropy) / (max_entropy - min_entropy)
    
    # 创建颜色映射
    colormap_uncertainty = plt.get_cmap('jet')
    uncertainty_colors = colormap_uncertainty(normalized_entropy.detach().cpu().numpy())[:, :3]
    
    # 创建分割掩码的颜色映射
    unique_labels = torch.unique(segmentation_mask)
    colormap_seg = plt.get_cmap('tab20')
    seg_colors = colormap_seg(segmentation_mask.cpu().numpy() / max(1, segmentation_mask.max().item()))[:, :3]

    # 获取点云的边界框
    points_np = point_cloud.cpu().numpy()[:, -3:]
    min_bound = np.min(points_np, axis=0)
    max_bound = np.max(points_np, axis=0)
    
    # 计算偏移量，使两个点云并排显示
    offset = np.array([max_bound[0] - min_bound[0] + 4.0, 0, 0])
    
    # 创建两个点云对象
    pcd_uncertainty = o3d.geometry.PointCloud()
    pcd_uncertainty.points = o3d.utility.Vector3dVector(points_np)
    pcd_uncertainty.colors = o3d.utility.Vector3dVector(uncertainty_colors)

    pcd_segmentation = o3d.geometry.PointCloud()
    pcd_segmentation.points = o3d.utility.Vector3dVector(points_np + offset)  # 添加偏移
    pcd_segmentation.colors = o3d.utility.Vector3dVector(seg_colors)

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Uncertainty and Segmentation Visualization")
    
    # 添加两个点云
    vis.add_geometry(pcd_uncertainty)
    vis.add_geometry(pcd_segmentation)
    
    # 设置渲染选项
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0

    # 添加文本标签（如果Open3D版本支持的话）
    try:
        # 创建两个文本标签
        labels = []
        labels.append({"pos": [min_bound[0], min_bound[1], max_bound[2]], 
                      "text": "Uncertainty"})
        labels.append({"pos": [min_bound[0] + offset[0], min_bound[1], max_bound[2]], 
                      "text": "Segmentation"})
        
        for label in labels:
            vis.add_3d_label(label["pos"], label["text"])
    except:
        print("Text labels not supported in this version of Open3D")

    # 自动调整视角以显示所有点
    vis.get_view_control().set_lookat([0, 0, 0])
    vis.get_view_control().set_front([0, 0, -1])
    vis.get_view_control().set_up([0, 1, 0])
    
    # 运行可视化
    vis.run()
    vis.destroy_window()

    print("Visualization complete")


""" CODE FOR VISUALIZE SINGLE OBJECT """
class VisualizeSingleSample:
    def __init__(self, dataset_name='S3DIS'):
        self.dataset_name = dataset_name
        self.dataset_path = '/home/jie/code/PCISeg/datasets/'
        if self.dataset_name == 'ScanNet':
            self.scans_path = os.path.join(self.dataset_path, self.dataset_name, 'scans')
        else:
            self.scans_path = os.path.join(self.dataset_path, self.dataset_name, 'single/crops')
        self.save_dir = '/home/jie/code/PCISeg/datasets/VISD/' + self.dataset_name + '_results'

    def visualize_allmasks(self, scene_name, object_id):

        self.object_id = object_id
        if self.dataset_name == 'ScanNet':
            scene_path = os.path.join(self.scans_path, scene_name + '.ply')
        else:
            scene_path = os.path.join(self.scans_path, scene_name, scene_name + '_crop_' + object_id + '.ply')
        point_cloud = read_ply(scene_path)

        coords_full = np.column_stack([point_cloud['x'], point_cloud['y'], point_cloud['z']]).astype(np.float64)
        colors_full = np.column_stack([point_cloud['R'], point_cloud['G'], point_cloud['B']])/255

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords_full)
        pcd.colors = o3d.utility.Vector3dVector(colors_full)


    def visualize_scene_GT(self, scene_name, labels):

        self.scene_name = scene_name

        self.save_sample_dir = os.path.join(self.save_dir, scene_name+'_'+self.object_id)
        if not os.path.exists(self.save_sample_dir):
            os.makedirs(self.save_sample_dir)
        if self.dataset_name == 'ScanNet':
            scene_path = os.path.join(self.scans_path, scene_name + '.ply')
        else:
            scene_path = os.path.join(self.scans_path, scene_name, scene_name + '_crop_' + self.object_id + '.ply')
        point_cloud = read_ply(scene_path)

        coords_full = np.column_stack([point_cloud['x'], point_cloud['y'], point_cloud['z']]).astype(np.float64)
        colors_full = np.column_stack([point_cloud['R'], point_cloud['G'], point_cloud['B']])/255

        coords_full = coords_full
        colors_full = colors_full
        labels = labels.cpu().numpy()

        self.visualize_scene(coords_full, colors_full)

        self.visualize_GT(coords_full, labels)

        self.coords_full = coords_full
        self.colors_full = colors_full

    def visualize_scene(self, point, colors):
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(point)
        scene_pcd.colors = o3d.utility.Vector3dVector(colors)

        # save scene
        save_dir = os.path.join(self.save_sample_dir, f"{self.scene_name}_pcd.ply")
        o3d.io.write_point_cloud(save_dir, scene_pcd, write_ascii=True)

        # vis = o3d.visualization.Visualizer()
        # vis.create_window(window_name="Scene Visualization")
        # vis.add_geometry(scene_pcd)
        # vis.run()
        # vis.destroy_window()

    def visualize_GT(self, point, labels):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point)
        # Create color map for different labels
        unique_labels = np.unique(labels)
        color_map = np.zeros((len(point), 3))
        
        # Assign different colors to different labels based on fixed_colors
        for label in unique_labels:
            color_map[labels == label] = fixed_colors[label]

        pcd.colors = o3d.utility.Vector3dVector(color_map)

        # save pcd
        save_dir = os.path.join(self.save_sample_dir, f"{self.scene_name}_obj_{self.object_id}_gt.ply")
        o3d.io.write_point_cloud(save_dir, pcd, write_ascii=True)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="GT Visualization")
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()

    def visualize_pred(self, pred, click_pos, iou):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.coords_full)

        pred = pred.cpu().numpy()
        unique_labels = np.unique(pred)
        color_map = np.zeros((len(self.coords_full), 3))
        for label in unique_labels:
            color_map[pred == label] = fixed_colors[label]

        pcd.colors = o3d.utility.Vector3dVector(color_map)

        num_clicks = 0
        for label, pos in click_pos.items():
            for coord in pos:
                num_clicks += 1
        
        # save pcd
        save_dir = os.path.join(self.save_sample_dir, f"{self.scene_name}_obj_{self.object_id}_pred_iou_{iou:.3f}_num_clicks_{num_clicks}.ply")
        o3d.io.write_point_cloud(save_dir, pcd, write_ascii=True)

        # vis = o3d.visualization.Visualizer()
        # vis.create_window(window_name=f"Pred Visualization - IoU: {iou:.3f}")
        # vis.add_geometry(pcd)
        # # add click points
        # for label, pos in click_pos.items():
        #     original_color = fixed_colors[int(label)]
        #     darker_color = original_color * 0.7
        #     for coord in pos:
        #         coord = coord.cpu().numpy()
        #         sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)  
        #         sphere.translate(coord)  
        #         sphere.paint_uniform_color(darker_color) 
        #         vis.add_geometry(sphere)
        
        # vis.run()
        # vis.destroy_window()
    
    def visualize_uncertainty(self, coords, predictions, iou, click_pos, entropy=False):
        """predictions: [N, K, 10]"""
        predictions = predictions.permute(0, 2, 1)
        predictions = torch.softmax(predictions, dim=-1)
        if entropy:
            mean_probs = predictions.mean(dim=1)
            entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)
            min_entropy, max_entropy = entropy.min(), entropy.max()
            normalized_uncertainty = (entropy - min_entropy) / (max_entropy - min_entropy)
        else:
            max_probs = torch.max(predictions, dim=-1)[0]  # [N, 10]
            uncertainty = torch.var(max_probs, dim=1)  # [N]
            min_uncertainty, max_uncertainty = uncertainty.min(), uncertainty.max()
            normalized_uncertainty = (uncertainty - min_uncertainty) / (max_uncertainty - min_uncertainty)
        
        # 创建颜色映射
        def custom_colormap(values):
            # 获取jet颜色映射
            jet_cmap = plt.get_cmap('jet')
            jet_colors = jet_cmap(values)[:, :3]
            # 浅灰色 RGB
            light_gray = np.array([0.8, 0.8, 0.8])
            
            # 设置阈值，当不确定性低于此值时使用浅灰色
            threshold = 0.2
            colors = np.zeros((len(values), 3))
            for i, value in enumerate(values):
                if value < threshold:
                    colors[i] = light_gray
                else:
                    # 重新归一化高于阈值的部分，使颜色过渡更平滑
                    normalized_value = (value - threshold) / (1 - threshold)
                    colors[i] = jet_colors[i]
            return colors

        # 使用自定义颜色映射
        uncertainty_colors = custom_colormap(normalized_uncertainty.detach().cpu().numpy())

        coords = coords.cpu().numpy()[:, -3:]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.colors = o3d.utility.Vector3dVector(uncertainty_colors)
        
        num_clicks = 0
        for label, pos in click_pos.items():
            for coord in pos:
                num_clicks += 1

        # save pcd
        save_dir = os.path.join(self.save_sample_dir, f"{self.scene_name}_uncertainty_iou_{iou:.3f}_num_clicks_{num_clicks}.ply")
        o3d.io.write_point_cloud(save_dir, pcd, write_ascii=True)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Uncertainty Visualization")
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()


""" CODE FOR VISUALIZE MULTI OBJECT """

class VisualizeSample:
    def __init__(self, dataset_name='ScanNet'):
        self.dataset_name = dataset_name
        self.dataset_path = '/home/jie/code/PCISeg/datasets/'
        self.scans_path = os.path.join(self.dataset_path, self.dataset_name, 'scans')
        self.save_dir = '/home/jie/code/PCISeg/datasets/VISD/' + self.dataset_name + '_multi_results'
 
    def visualize_allmasks(self, scene_name):
        scene_path = os.path.join(self.scans_path, scene_name + '.ply')

        self.sample_save_dir = os.path.join(self.save_dir, scene_name)
        if not os.path.exists(self.sample_save_dir):
            os.makedirs(self.sample_save_dir)

        point_cloud = read_ply(scene_path)

        coords_full = np.column_stack([point_cloud['x'], point_cloud['y'], point_cloud['z']]).astype(np.float64)
        colors_full = np.column_stack([point_cloud['R'], point_cloud['G'], point_cloud['B']])/255
        labels_full = point_cloud['label'].astype(np.int32)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords_full)
        pcd.colors = o3d.utility.Vector3dVector(colors_full)
        
        # visualize instance one by one
        # unique_labels = np.unique(labels_full)
        # for label in unique_labels:
        #     visualize_instance(pcd, labels_full, label)
        #     print('visualizing instance: ', label)

        # print('visualizing all instances scene: ', scene_name)
        # #visualize all instances
        # visualize_all_instances(coords_full, labels_full)

        remove_instance_labe = [9,10,2,3,4,6,11,13,15,24,25]
        self.keep_mask = ~np.isin(labels_full, remove_instance_labe)


    def visualize_scene_GT(self, scene_name, labels):

        self.scene_name = scene_name

        scene_path = os.path.join(self.scans_path, scene_name + '.ply')
        point_cloud = read_ply(scene_path)

        coords_full = np.column_stack([point_cloud['x'], point_cloud['y'], point_cloud['z']]).astype(np.float64)
        colors_full = np.column_stack([point_cloud['R'], point_cloud['G'], point_cloud['B']])/255


        coords_full = coords_full[self.keep_mask]
        colors_full = colors_full[self.keep_mask]
        labels = labels.cpu().numpy()[self.keep_mask]

        self.visualize_scene(coords_full, colors_full)

        self.visualize_GT(coords_full, labels)

        self.coords_full = coords_full
        self.colors_full = colors_full

    def visualize_scene(self, point, colors):
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(point)
        scene_pcd.colors = o3d.utility.Vector3dVector(colors)

        # save scene
        save_dir = os.path.join(self.sample_save_dir, f"{self.scene_name}_pcd.ply")
        o3d.io.write_point_cloud(save_dir, scene_pcd, write_ascii=True)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Scene Visualization")
        vis.add_geometry(scene_pcd)
        vis.run()
        vis.destroy_window()

    def visualize_GT(self, point, labels):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point)
        # Create color map for different labels
        unique_labels = np.unique(labels)
        color_map = np.zeros((len(point), 3))
        
        # Assign different colors to different labels based on fixed_colors
        for label in unique_labels:
            color_map[labels == label] = fixed_colors[label]

        pcd.colors = o3d.utility.Vector3dVector(color_map)

        # save pcd
        save_dir = os.path.join(self.sample_save_dir, f"{self.scene_name}_gt.ply")
        o3d.io.write_point_cloud(save_dir, pcd, write_ascii=True)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="GT Visualization")
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()

    def visualize_pred(self, pred, click_pos, iou, num_clicks):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.coords_full)

        pred = pred.cpu().numpy()[self.keep_mask]
        unique_labels = np.unique(pred)
        color_map = np.zeros((len(self.coords_full), 3))
        for label in unique_labels:
            color_map[pred == label] = fixed_colors[label]

        pcd.colors = o3d.utility.Vector3dVector(color_map)
        
        # save pcd
        save_dir = os.path.join(self.sample_save_dir, f"{self.scene_name}_pred_iou_{iou:.3f}_num_clicks_{num_clicks}.ply")
        o3d.io.write_point_cloud(save_dir, pcd, write_ascii=True)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"Pred Visualization - IoU: {iou:.3f}")
        vis.add_geometry(pcd)
        # add click points
        for label, pos in click_pos.items():
            original_color = fixed_colors[int(label)]
            darker_color = original_color * 0.7
            for coord in pos:
                coord = coord.cpu().numpy()
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)  
                sphere.translate(coord)  
                sphere.paint_uniform_color(darker_color) 
                vis.add_geometry(sphere)
        
        vis.run()
        vis.destroy_window()

    def visualize_uncertainty(self, coords, predictions, iou, num_clicks, entropy=False):
        """predictions: [N, K, 10]"""
        predictions = predictions.permute(0, 2, 1)
        predictions = torch.softmax(predictions, dim=-1)
        if entropy:
            mean_probs = predictions.mean(dim=1)
            entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)
            min_entropy, max_entropy = entropy.min(), entropy.max()
            normalized_uncertainty = (entropy - min_entropy) / (max_entropy - min_entropy)
        else:
            class_uncertainties = torch.var(predictions, dim=1)  # [N, K]
            
            # 对每个类别的uncertainty单独归一化
            min_vals = class_uncertainties.min(dim=0)[0]  # [K]
            max_vals = class_uncertainties.max(dim=0)[0]  # [K]
            normalized_class_uncertainties = (class_uncertainties - min_vals) / (max_vals - min_vals)  # [N, K]
            
            # 取每个点在所有类别中的最大不确定性
            uncertainty = normalized_class_uncertainties.max(dim=1)[0]  # [N]
            
            # 最后对所有点做一次全局归一化
            min_uncertainty, max_uncertainty = uncertainty.min(), uncertainty.max()
            normalized_uncertainty = (uncertainty - min_uncertainty) / (max_uncertainty - min_uncertainty)

        # 创建颜色映射
        def custom_colormap(values):
            # 获取jet颜色映射
            jet_cmap = plt.get_cmap('jet')
            jet_colors = jet_cmap(values)[:, :3]
            # 浅灰色 RGB
            light_gray = np.array([0.8, 0.8, 0.8])
            
            # 设置阈值，当不确定性低于此值时使用浅灰色
            threshold = 0.3
            colors = np.zeros((len(values), 3))
            for i, value in enumerate(values):
                if value < threshold:
                    colors[i] = light_gray
                else:
                    # 重新归一化高于阈值的部分，使颜色过渡更平滑
                    normalized_value = (value - threshold) / (1 - threshold)
                    colors[i] = jet_colors[i]
            return colors

        # 使用自定义颜色映射
        uncertainty_colors = custom_colormap(normalized_uncertainty.detach().cpu().numpy())

        coords = coords.cpu().numpy()[:, -3:]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.colors = o3d.utility.Vector3dVector(uncertainty_colors)


        # save pcd
        save_dir = os.path.join(self.sample_save_dir, f"{self.scene_name}_uncertainty_iou_{iou:.3f}_num_clicks_{num_clicks}.ply")
        o3d.io.write_point_cloud(save_dir, pcd, write_ascii=True)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Uncertainty Visualization")
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()

""" CODE FOR VISUALIZE DATASET """

class VisualizeDataset:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset_path = '/home/jie/code/PCISeg/datasets/'
        self.scans_path = os.path.join(self.dataset_path, self.dataset_name, 'scans')
        self.val_list = os.path.join(self.dataset_path, self.dataset_name, 'val_list.json')
    
        self.data_samples = json.load(open(self.val_list))
        self.scene_list = list(self.data_samples.keys())

    def visualize_scene(self):
        for scene in self.scene_list:
            scene_name = scene.split('_obj_')[0]
            scene_path = os.path.join(self.scans_path, scene_name + '.ply')
            point_cloud = read_ply(scene_path)

            coords_full = np.column_stack([point_cloud['x'], point_cloud['y'], point_cloud['z']]).astype(np.float64)
            colors_full = np.column_stack([point_cloud['R'], point_cloud['G'], point_cloud['B']])/255
            labels_full = point_cloud['label'].astype(np.int32)

            data_sample = self.data_samples[scene]
            labels_full_new = self.compute_labels(labels_full, data_sample['obj'])

            # Remove points with label 0
            mask = labels_full_new != 0
            coords_full = coords_full[mask]
            colors_full = colors_full[mask]
            labels_full_new = labels_full_new[mask]

            print('visualizing scene: ', scene)

            # visualize scene and labels
            self.visualize_scene_label(coords_full, colors_full, labels_full_new)

    def visualize_scene_label(self, point, colors, labels):
        """
        Visualize both the scene point cloud and its segmentation labels
        
        Args:
            point: Nx3 array of point cloud coordinates
            colors: Nx3 array of RGB colors
            labels: N array of segmentation labels
        """
        import open3d as o3d
        
        # Create two point clouds - one for scene, one for labels
        scene_pcd = o3d.geometry.PointCloud()
        label_pcd = o3d.geometry.PointCloud()
        
        # Set points for both point clouds
        scene_pcd.points = o3d.utility.Vector3dVector(point)
        label_pcd.points = o3d.utility.Vector3dVector(point)
        
        # Set colors for scene visualization
        scene_pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize RGB values
        
        # Generate colors for labels using distinct colors for each unique label
        label_colors = np.zeros_like(point)
        unique_labels = np.unique(labels)
        color_map = plt.get_cmap('tab20')(np.linspace(0, 1, len(unique_labels)))[:, :3]
        
        for idx, label in enumerate(unique_labels):
            label_mask = labels == label
            label_colors[label_mask] = color_map[idx]
        
        label_pcd.colors = o3d.utility.Vector3dVector(label_colors)
        
        # Create visualization window
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Scene and Labels Visualization", width=1920, height=1080, left=50, top=50)
        
        # Add geometries
        vis.add_geometry(scene_pcd)
        vis.add_geometry(label_pcd)
        
        # Set initial view
        view_control = vis.get_view_control()
        view_control.set_zoom(0.8)
        
        # Toggle function to switch between scene and labels view
        def toggle_view(vis, action, mods):
            global show_labels
            if not hasattr(toggle_view, 'show_labels'):
                toggle_view.show_labels = False
            
            toggle_view.show_labels = not toggle_view.show_labels
            if toggle_view.show_labels:
                scene_pcd.colors = o3d.utility.Vector3dVector(label_colors)
            else:
                scene_pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            return False
        
        # Register key callback (press 'T' to toggle between views)
        vis.register_key_callback(ord('T'), toggle_view)
        
        print("Press 'T' to toggle between scene and label visualization")
        print("Press 'Q' or 'Esc' to exit")
        
        # Run visualization
        vis.run()
        vis.destroy_window()

    def compute_labels(self, ori_labels, correspondence):

        new_labels = np.zeros(ori_labels.shape)

        for new_obj_id, ori_obj_id in correspondence.items():
            new_labels[ori_labels==ori_obj_id] = int(new_obj_id)

        return new_labels



if __name__ == '__main__':

    dataset_name = ['ScanNet', 'S3DIS', 'Replica', 'KITTI360', 'PartNet']

    visualize_dataset = VisualizeDataset(dataset_name[1])
    visualize_dataset.visualize_scene()