import open3d as o3d
import numpy as np
import json
import os
import sys
from utils import read_ply, write_ply

def load_and_process_data(mesh_file, label_file):
    """加载和处理网格及标签数据"""
    # 加载网格文件
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    points = np.asarray(mesh.vertices).astype(np.float32)
    colors = np.asarray(mesh.vertex_colors)

    # color-->255
    colors = (colors * 255).astype(np.uint8)

    # 加载标签文件
    with open(label_file, 'r') as file:
        labels = file.readlines()
    labels = [int(label.strip()) for label in labels]
    labels = np.array(labels)
    
    # 重新映射标签
    unique_labels = np.unique(labels)
    offset = np.max(labels) + 1
    temp_labels = labels + offset
    label_mapping = {old + offset: new + 1 for new, old in enumerate(unique_labels)}
    instance_labels = np.array([label_mapping[label] for label in temp_labels])
    
    return points, colors, instance_labels

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
    
    print("\n操作说明:")
    print("1) 使用鼠标滚轮缩放")
    print("2) 按住鼠标左键旋转视角")
    print("3) 按住鼠标右键平移视角")
    print("4) 按 'Q' 退出当前视图")
    
    # 运行可视化器
    vis.run()
    vis.destroy_window()

def main(scene_name, save):
    # 设置文件路径
    mesh_file = f"/home/jie/code/PCISeg/datasets/Replica/{scene_name}/{scene_name}_mesh.ply"
    label_file = f"/home/jie/code/PCISeg/datasets/Replica/ground_truth/{scene_name}.txt"
    save_dir = f"/home/jie/code/PCISeg/datasets/Replica/scans/{scene_name}.ply"
    
    # 加载和处理数据
    print("正在加载数据...")
    points, colors, instance_labels = load_and_process_data(mesh_file, label_file)
    
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 打印可用的实例ID
    unique_ids = np.unique(instance_labels)
    print(f"\n可用的实例ID: {sorted(unique_ids)}")

    if save:
        points = points.astype(np.float32)
        colors = colors.astype(np.uint8)
        instance_labels = instance_labels.astype(np.int32)
        write_ply(save_dir, [points, colors, instance_labels], ['x', 'y', 'z', 'R', 'G', 'B', 'label'])
        print(f"已保存点云到: {save_dir}")

def test_ply(scene_name):
    pcd = read_ply(f"/home/jie/code/PCISeg/datasets/Replica/scans/{scene_name}.ply")

    coords_full = np.column_stack([pcd['x'], pcd['y'], pcd['z']]).astype(np.float64)
    colors_full = np.column_stack([pcd['R'], pcd['G'], pcd['B']])/255
    labels_full = pcd['label'].astype(np.int32)

    unique_ids = np.unique(labels_full)

    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords_full)
    #pcd.colors = o3d.utility.Vector3dVector(colors_full)

    # 统计每个实例的点数量并筛选大于1000个点的实例
    instance_points_count = {}
    for instance_id in unique_ids:
        points_in_instance = np.sum(labels_full == instance_id)
        instance_points_count[instance_id] = points_in_instance
    
    #筛选大于1000个点的实例
    large_instances = {id: count for id, count in instance_points_count.items() if count > 3000}
    
    print("\n点数大于3000的实例:")

    for instance_id, count in large_instances.items():
        print(f"实例 {instance_id}: {count} 个点")
        visualize_instance(pcd, labels_full, instance_id)


def read_and_rewrite_ply():
    point_cloud = read_ply(f"/home/jie/code/PCISeg/datasets/VISD/scans/office_1.ply")
    save_dir = f"/home/jie/code/PCISeg/datasets/VISD/scans/office_1_new.ply"

    coords_full = np.column_stack([point_cloud['x'], point_cloud['y'], point_cloud['z']]).astype(np.float64)
    colors_full = np.column_stack([point_cloud['R'], point_cloud['G'], point_cloud['B']])/255

    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords_full)
    pcd.colors = o3d.utility.Vector3dVector(colors_full)

    o3d.io.write_point_cloud(save_dir, pcd, write_ascii=True)


def merge_ply(scene_name, save_dir):
    dataset_path = '/home/jie/code/PCISeg/datasets/VISD/interactive_dataset'
    scene_path = os.path.join(dataset_path, scene_name, 'scan.ply')
    label_path = os.path.join(dataset_path, scene_name, 'label.ply')

    mesh = o3d.io.read_triangle_mesh(scene_path)
    points = np.asarray(mesh.vertices).astype(np.float32)
    colors = np.asarray(mesh.vertex_colors).astype(np.uint8)

    labels = read_ply(label_path)
    labels = labels['label'].astype(np.int32)

    write_ply(save_dir, [points, colors, labels], ['x', 'y', 'z', 'R', 'G', 'B', 'label'])

    print(1)


if __name__ == "__main__":
    # scene_name = "room_2"
    #test_ply()
    #read_and_rewrite_ply()
    
    scene_list = ['office_8', 'conferenceRoom_1', '0000011079_0000011287_exp_8', '0000001537_0000001755_exp_5']
    scene_name = scene_list[3]
    save_path = f"/home/jie/code/PCISeg/datasets/VISD/scans/{scene_name}.ply"

    merge_ply(scene_name, save_path)