import numpy as np
from utils import read_ply, write_ply

def save_selected_files():
    notion_path = '/home/jie/mydisk/PartNet/all_valid_anno_info.txt'
    input_path = '/home/jie/mydisk/PartNet/data_v0/'
    output_path = '/home/jie/code/PCISeg/datasets/PartNet/'

    # 1. read the text file
    with open(notion_path, encoding='utf-8') as f:
        lines = f.readlines()

    # Split each line by spaces and store in a list
    file_name_list = []
    file_class_name_list = []
    for line in lines:
        split_line = line.strip().split()

        file_name = split_line[0]
        file_class_name = split_line[2]
        file_name_list.append(file_name)
        file_class_name_list.append(file_class_name)


    # Create a dictionary to store files by class
    class_to_files = {}

    # Group files by their class names
    for file_name, class_name in zip(file_name_list, file_class_name_list):
        if class_name not in class_to_files:
            class_to_files[class_name] = []
        class_to_files[class_name].append(file_name)


    target_classes = ['Chair', 'Bed', 'Display', 'Microwave', 'Refrigerator', 'Scissors']

    # For each target class, randomly select 10 files
    selected_files = {}
    for target_class in target_classes:
        if target_class in class_to_files:
            # Get list of files for this class
            class_files = class_to_files[target_class]
            # Randomly select 10 files (or all if less than 10)
            num_to_select = min(10, len(class_files))
            selected = np.random.choice(class_files, size=num_to_select, replace=False)
            
            # Store selected files for this class
            selected_files[target_class] = selected.tolist()
            
            # Copy files to output path
            for file_id in selected:
                src_path = input_path + file_id
                dst_path = output_path + file_id
                import shutil
                import os
                
                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                
                # Copy the file
                try:
                    shutil.copytree(src_path, dst_path)
                    print(f"Copied {file_id} to {dst_path}")
                except Exception as e:
                    print(f"Error copying {file_id}: {str(e)}")
        else:
            print(f"Warning: Class {target_class} not found in dataset")

    # Update class_to_files to only include selected files
    class_to_files = selected_files

    print('Done')

import os
import json
import matplotlib.pyplot as plt
import open3d as o3d


def visualize_point_cloud(pcd_data, pcd_colors, label_data):
    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_data)
    
    # Color points based on labels
    colors = np.zeros((len(label_data), 3))
    unique_labels = np.unique(label_data)
    color_map = plt.get_cmap('tab20')(np.linspace(0, 1, len(unique_labels)))[:,:3]
    
    for i, label in enumerate(unique_labels):
        mask = label_data == label
        colors[mask] = color_map[i]
        
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize
    o3d.visualization.draw_geometries([pcd])


def generate_multi_object_files():

    dir_path = '/home/jie/code/PCISeg/datasets/PartNet/'
    val_list = {}

    for file in os.listdir(dir_path):
        print('Processing', file)
        file_path = os.path.join(dir_path, file)
        
        if file != 'scans' and file != 'single':
            pcd_path = os.path.join(file_path, 'point_sample/sample-points-all-pts-nor-rgba-10000.txt')
            label_path = os.path.join(file_path, 'point_sample/sample-points-all-label-10000.txt')

            pcd_data = np.loadtxt(pcd_path)
            label_data = np.loadtxt(label_path)

            unique_labels = np.unique(label_data)
            
            # save the full point cloud
            pcd_coords = pcd_data[:, :3].astype(np.float32)
            pcd_colors = pcd_data[:, 6:9].astype(np.uint8)
            pcd_label = label_data.astype(np.int32)
            
            save_dir = os.path.join(dir_path, 'scans', f'sample_{file}.ply')
            write_ply(save_dir, [pcd_coords, pcd_colors, pcd_label], ['x', 'y', 'z', 'R', 'G', 'B', 'label'])

            # get the number of points in each label, select those with more than 100 points
            label_points = {}
            for label in unique_labels:
                label_points[label] = np.sum(label_data == label)
            label_points = sorted(label_points.items(), key=lambda x: x[1], reverse=True)
            selected_labels = [label for label, points in label_points if points > 100]
           
            # prepare the val list
            random_num = np.random.randint(3, 10)
            random_num = min(random_num, len(selected_labels))
            random_labels = np.random.choice(selected_labels, size=random_num, replace=False)

            key = f'sample_{file}_obj_{random_num}'

            clicks = {str(i+1): [] for i in range(random_num)}
            clicks['0'] = []
            obj = {str(i+1): random_labels[i] for i in range(random_num)}

            val_list[key] = {'clicks': clicks, 'obj': obj}

    with open(os.path.join(dir_path, 'val_list.json'), 'w') as f:
        json.dump(val_list, f)

    print('Done')

def generate_single_object_files():
    dir_path = '/home/jie/code/PCISeg/datasets/PartNet/'
    save_dir = '/home/jie/code/PCISeg/datasets/PartNet/single'

    object_id_path = '/home/jie/code/PCISeg/datasets/PartNet/single/object_ids.npy'
    object_class_path = '/home/jie/code/PCISeg/datasets/PartNet/single/object_classes.txt'
  
    object_list = []
    object_class_list = []
    for file in os.listdir(dir_path):
        print('Processing', file)
        file_path = os.path.join(dir_path, file)

        if file != 'scans' and file != 'single' and file != 'val_list.json':
            pcd_path = os.path.join(file_path, 'point_sample/sample-points-all-pts-nor-rgba-10000.txt')
            label_path = os.path.join(file_path, 'point_sample/sample-points-all-label-10000.txt')

            pcd_data = np.loadtxt(pcd_path)
            label_data = np.loadtxt(label_path)

            unique_labels = np.unique(label_data)
            print(unique_labels)

            # get the number of points in each label, select those with more than 100 points
            label_points = {}
            for label in unique_labels:
                label_points[label] = np.sum(label_data == label)
            label_points = sorted(label_points.items(), key=lambda x: x[1], reverse=True)
            selected_labels = [label for label, points in label_points if points > 300]

            # save each label as a separate file
            for idx, label in enumerate(selected_labels):
                mask = label_data == label

                pcd_coords = pcd_data[:, :3].astype(np.float32)
                pcd_colors = pcd_data[:, 6:9].astype(np.uint8)
                pcd_label = mask.astype(np.int32)

                object_save_dir = os.path.join(save_dir, 'crops', f'sample_{file}')
                if not os.path.exists(object_save_dir):
                    os.makedirs(object_save_dir)

                save_path = os.path.join(object_save_dir, f'sample_{file}_crop_{int(idx)}.ply')
                write_ply(save_path, [pcd_coords, pcd_colors, pcd_label], ['x', 'y', 'z', 'R', 'G', 'B', 'label'])

                object_class_list.append(label)
                object_list.append([f'sample_{file}', idx])
   
    # save as npy
    object_list = np.array(object_list)
    np.save(object_id_path, object_list)
    
    # save as txt
    with open(object_class_path, 'w') as f:
        for obj_class in object_class_list:
            f.write(f'{obj_class}\n')
                
    print(1)

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

def visualize_single_object_files():
    dir_path = '/home/jie/code/PCISeg/datasets/PartNet/single'
    for file in os.listdir(dir_path):
        print('Processing', file)
        scene_path = os.path.join(dir_path, file)
        
        for obj in os.listdir(scene_path):
            pcd_path = os.path.join(scene_path, obj)

            pcd = read_ply(pcd_path)

            coords_full = np.column_stack([pcd['x'], pcd['y'], pcd['z']]).astype(np.float64)
            colors_full = np.column_stack([pcd['R'], pcd['G'], pcd['B']])/255
            labels_full = pcd['label'].astype(np.int32)

            visualize_instance(coords_full, labels_full, 1)

            print(1)


if __name__ == '__main__':
    
    #generate_multi_object_files()
    #generate_single_object_files()
    #visualize_single_object_files()
    
    # load ids and classes
    classes_path = '/home/jie/code/PCISeg/datasets/Replica/single/object_classes.txt'
    classes = np.loadtxt(classes_path, dtype=int)
    unique_classes = np.unique(classes)

    print(unique_classes)
