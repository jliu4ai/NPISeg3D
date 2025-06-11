import os
import glob
import numpy as np
import open3d as o3d
import json
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
from utils import read_ply, write_ply


def show_instance(target_obj_id, coords_full, labels_full):

    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords_full)

    # Create mask for target object
    mask = (labels_full == target_obj_id)
    target_points = coords_full[mask]

    # Create point cloud for target object
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)

    # Visualize only target object
    o3d.visualization.draw_geometries([target_pcd])


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
    points = np.asarray(pcd)
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


def process_scene(scene_name):      
    scene_dir = '/home/jie/code/PCISeg/datasets/Replica/scans/'

    scene_path = os.path.join(scene_dir, f'{scene_name}.ply')

    # load the scene
    scene = o3d.io.read_point_cloud(scene_path)

    points = np.asarray(scene.points)
    colors = np.asarray(scene.colors)
    labels = np.asarray(scene.labels)

    unique_labels = np.unique(labels)

    # Create output directory for crops
    crop_dir = os.path.join('/home/jie/code/PCISeg/datasets/Replica/single/crops', scene_name)
    os.makedirs(crop_dir, exist_ok=True)

    # Process each unique object instance
    for label in unique_labels:
        # Skip background (usually label 0)
        if label == 0:
            continue
            
        # Get points belonging to this instance
        mask = labels == label
        instance_points = points[mask]
        instance_colors = colors[mask]
        instance_labels = labels[mask]
        
        # Skip if too few points
        if len(instance_points) < 100:
            continue
            
        # Create point cloud for this instance
        instance_pcd = o3d.geometry.PointCloud()
        instance_pcd.points = o3d.utility.Vector3dVector(instance_points)
        instance_pcd.colors = o3d.utility.Vector3dVector(instance_colors)
        
        # Get bounding box and add some padding
        bbox = instance_pcd.get_axis_aligned_bounding_box()
        bbox_min = np.asarray(bbox.get_min_bound()) - 0.5  # 0.5m padding
        bbox_max = np.asarray(bbox.get_max_bound()) + 0.5
        
        # Crop points from full scene within padded bounds
        crop_mask = np.all((points >= bbox_min) & (points <= bbox_max), axis=1)
        crop_points = points[crop_mask]
        crop_colors = colors[crop_mask]
        crop_labels = (labels[crop_mask] == label).astype(np.int32)  # Convert to binary mask
        
        # Save cropped point cloud
        crop_filename = f"{scene_name}_crop_{label}.ply"
        crop_path = os.path.join(crop_dir, crop_filename)
        
        crop_pcd = o3d.geometry.PointCloud()
        crop_pcd.points = o3d.utility.Vector3dVector(crop_points)
        crop_pcd.colors = o3d.utility.Vector3dVector(crop_colors)
        
        # Save with custom attributes including binary labels
        o3d.io.write_point_cloud(crop_path, crop_pcd, write_ascii=True)
        
        # Also save the binary labels separately
        np.save(os.path.join(crop_dir, f"{scene_name}_crop_{label}_labels.npy"), crop_labels)
        
        print(f"Saved crop for instance {label} with {len(crop_points)} points")

def get_scene_semantic_labels(scene_name):
    path_in = f'/home/jie/mydisk/Replica_old/{scene_name}/habitat/mesh_semantic.ply'

    print("Reading input...")
    file_in = PlyData.read(path_in)
    vertices_in = file_in.elements[0]
    faces_in = file_in.elements[1]

    # 初始化顶点标签数组（默认为-1表示未赋值）
    num_vertices = len(vertices_in)
    vertex_labels = np.full(num_vertices, -1, dtype=np.int32)

    # 从faces遍历，将object_id赋值给对应的顶点
    for face in faces_in:
        vertex_indices = face[0]  # 获取面片的顶点索引
        object_id = face[1]      # 获取面片的object_id
        
        # 将object_id赋值给这个面片的所有顶点
        for vertex_idx in vertex_indices:
            vertex_labels[vertex_idx] = object_id

    # 获取顶点坐标
    vertex_coords = np.vstack([vertices_in['x'], vertices_in['y'], vertices_in['z']]).T
    vertex_colors = np.vstack([vertices_in['red'], vertices_in['green'], vertices_in['blue']]).T

    # 将坐标和标签组合
    vertex_data = np.column_stack([vertex_coords, vertex_labels])
    
    print(f"Total vertices: {len(vertex_coords)}")
    print(f"Unique semantic labels: {np.unique(vertex_labels)}")
    
    # 可选：保存结果
    output_path = f"{scene_name}_vertex_labels.npy"
    np.save(output_path, vertex_data)
    
    return vertex_coords, vertex_labels, vertex_colors


def save_files():

    object_id_path = '/home/jie/code/PCISeg/datasets/Replica/single/object_ids.npy'
    object_class_path = '/home/jie/code/PCISeg/datasets/Replica/single/object_classes.txt'
    crop_dir = '/home/jie/code/PCISeg/datasets/Replica/single/crops'

    scene_names = ['office_0', 'office_1', 'office_2', 'office_3', 'office_4', 'room_0', 'room_1', 'room_2']

    object_list = []
    object_class_list = []
    
    for scene_name in scene_names:
        print(f"Processing scene: {scene_name}")

        # create crop dir
        if not os.path.exists(f'{crop_dir}/{scene_name}'):
            os.makedirs(f'{crop_dir}/{scene_name}')
        else:
            print(f'{crop_dir}/{scene_name} already exists')
            continue

        # pcd, coord， object_id, color for each point
        vertex_coords, vertex_labels, vertex_colors = get_scene_semantic_labels(scene_name)

        # get semantic class for each point
        semantic_path = f'/home/jie/mydisk/Replica_old/{scene_name}/habitat/info_semantic.json'
        with open(semantic_path, 'r') as file:
            semantic_info = json.load(file)
        
        id_to_semantic = semantic_info['id_to_label']  # mapping from object id to semantic class

        vertex_class = [id_to_semantic[i] for i in vertex_labels]  # class id for each point
        
        # get object id with more than 3000 points
        obj_ids = np.unique(vertex_labels)
        obj_ids = [i for i in obj_ids if np.sum(vertex_labels == i) > 3000]
        num_obj = len(obj_ids)

        # process each object for segmentation
        for idx, obj_id in enumerate(obj_ids):
            print(f"Processing object {idx+1} of {num_obj} in scene {scene_name}")
            # 获取目标物体的点
            obj_mask = vertex_labels == obj_id
            obj_points = vertex_coords[obj_mask]
            
            # 计算目标物体的边界框
            obj_min = np.min(obj_points, axis=0)
            obj_max = np.max(obj_points, axis=0)
            
            # 添加边界框padding
            padding = 2  # 0.5米的padding
            bbox_min = obj_min - padding
            bbox_max = obj_max + padding
            
            # 获取边界框内的所有点（包括其他物体的点）
            bbox_mask = np.all((vertex_coords >= bbox_min) & (vertex_coords <= bbox_max), axis=1)
            
            # 提取边界框内的所有点数据
            crop_points = vertex_coords[bbox_mask]
            crop_colors = vertex_colors[bbox_mask]
            crop_labels = vertex_labels[bbox_mask]
            
            # 创建二值分割标签：1表示目标物体，0表示其他所有物体（背景）
            binary_labels = (crop_labels == obj_id).astype(np.int32)

            # visualize the cropped object
            #visualize_instance(crop_points, crop_labels, obj_id)

            # save the cropped object
            crop_path = os.path.join(crop_dir, f"{scene_name}/{scene_name}_crop_{idx}.ply")

            points = crop_points.astype(np.float32)
            colors = crop_colors.astype(np.uint8)
            instance_labels = binary_labels.astype(np.int32)
            write_ply(crop_path, [points, colors, instance_labels], ['x', 'y', 'z', 'R', 'G', 'B', 'label'])

            # save object  class
            object_class_list.append(obj_id)

            # save object id
            object_list.append([scene_name, idx])
    
    # save as npy
    object_list = np.array(object_list)
    np.save(object_id_path, object_list)
    
    # save as txt
    with open(object_class_path, 'w') as f:
        for obj_class in object_class_list:
            f.write(f'{obj_class}\n')
                
    print(1)

if __name__ == '__main__':
    # ids_path =  '/home/jie/code/PCISeg/datasets/KITTI360/single/object_ids.npy'
    # classes_path = '/home/jie/code/PCISeg/datasets/KITTI360/single/object_classes.txt'

    # # load ids and classes
    # ids = np.load(ids_path)
    # classes = np.loadtxt(classes_path, dtype=str)

    object_path = '/home/jie/code/PCISeg/datasets/Replica/single/crops/office_4/office_4_crop_5.ply'

    pcd = read_ply(object_path)

    coords_full = np.column_stack([pcd['x'], pcd['y'], pcd['z']]).astype(np.float64)
    colors_full = np.column_stack([pcd['R'], pcd['G'], pcd['B']])/255
    labels_full = pcd['label'].astype(np.int32)

    visualize_instance(coords_full, labels_full, 1)

    print(1)