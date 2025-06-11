import os
import open3d as o3d
import numpy as np
from utils.ply import read_ply
import matplotlib.pyplot as plt


def load_and_process_data(mesh_file, label_file):
    """Load and process mesh and label data"""
    # Load mesh file
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    points = np.asarray(mesh.vertices).astype(np.float32)
    colors = np.asarray(mesh.vertex_colors)

    # color-->255
    colors = (colors * 255).astype(np.uint8)

    # Load label file
    with open(label_file, 'r') as file:
        labels = file.readlines()
    labels = [int(label.strip()) for label in labels]
    labels = np.array(labels)
    
    # Remap labels
    unique_labels = np.unique(labels)
    offset = np.max(labels) + 1
    temp_labels = labels + offset
    label_mapping = {old + offset: new + 1 for new, old in enumerate(unique_labels)}
    instance_labels = np.array([label_mapping[label] for label in temp_labels])
    
    return points, colors, instance_labels

def visualize_instance(pcd, instance_labels, target_id):
    """
    Visualize instance with specified ID, other instances shown as semi-transparent
    Args:
        pcd: Open3D point cloud object
        instance_labels: Instance label array
        target_id: Instance ID to highlight
    """
    print(f"\n=== Displaying Instance ID: {target_id} ===")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(f"Instance ID: {target_id}")
    
    # Set rendering options - set before adding geometry
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])  # Black background
    opt.point_size = 0.1  # Set default point size
    
    # Prepare point cloud data
    points = np.asarray(pcd.points)
    mask = (instance_labels == target_id)
    target_points = points[mask]
    other_points = points[~mask]
    
    # 1. Add background point cloud
    pcd_others = o3d.geometry.PointCloud()
    pcd_others.points = o3d.utility.Vector3dVector(other_points)
    colors_others = np.ones((len(other_points), 3)) * np.array([0.5, 0.5, 0.5])  # Gray
    pcd_others.colors = o3d.utility.Vector3dVector(colors_others)
    vis.add_geometry(pcd_others)
    
    # 2. Add target instance point cloud
    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(target_points)
    colors_target = np.ones((len(target_points), 3)) * np.array([1, 0, 0])  # Red
    pcd_target.colors = o3d.utility.Vector3dVector(colors_target)
    
    # Increase target point size
    opt.point_size = 0.1  # Set larger point size
    vis.add_geometry(pcd_target)
    
    # Print instance information
    points_count = len(target_points)
    center = np.mean(target_points, axis=0)
    print(f"Instance Information:")
    print(f"- Number of points: {points_count}")
    print(f"- Center position: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    
    # Adjust view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    print("\nOperation Instructions:")
    print("1) Use mouse wheel to zoom")
    print("2) Hold left mouse button to rotate view")
    print("3) Hold right mouse button to pan view")
    print("4) Press 'Q' to exit current view")
    
    # Run visualizer
    vis.run()
    vis.destroy_window()

def visualize_all_instances(points, instance_labels):
    # Get unique instance labels
    unique_labels = np.unique(instance_labels)
    
    # Create color mapping
    colormap = plt.get_cmap('tab20')  # Use tab20 colormap, supports up to 20 different colors
    
    # Create a point cloud object for each instance and assign different colors
    vis = o3d.visualization.Visualizer()
    vis.create_window("Instance Segmentation")
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])  # Black background
    opt.point_size = 0.1  # Set point size
    
    # Create point cloud for each instance and add to visualizer
    for i, label in enumerate(unique_labels):
        # Get points for current instance
        mask = (instance_labels == label)
        instance_points = points[mask]
        
        # Create point cloud object
        instance_pcd = o3d.geometry.PointCloud()
        instance_pcd.points = o3d.utility.Vector3dVector(instance_points)
        
        # Assign color for current instance
        color = colormap(i % 20)[:3]  # Get RGB values, exclude alpha channel
        instance_colors = np.tile(color, (len(instance_points), 1))
        instance_pcd.colors = o3d.utility.Vector3dVector(instance_colors)
        
        # Add to visualizer
        vis.add_geometry(instance_pcd)
    
    # Adjust view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    print("\nOperation Instructions:")
    print("1) Use mouse wheel to zoom")
    print("2) Hold left mouse button to rotate view")
    print("3) Hold right mouse button to pan view")
    print("4) Press 'Q' to exit current view")
    
    # Run visualizer
    vis.run()
    vis.destroy_window()


def visualize_original_point_cloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors/255.0)
    o3d.visualization.draw_geometries([pcd])
    print(1)


if __name__ == '__main__':
    data_path = '/home/jie/mydisk/replica/room1/room1_mesh.ply'
    label_path = '/home/jie/mydisk/replica/ground_truth/room1.txt'

    points, colors, instance_labels = load_and_process_data(data_path, label_path)

    remove_instance_label = [1, 2, 5, 10, 11, 12, 18, 19, 22, 23, 27, 30, 31, 44, 45, 46,47, 53, 54, 55, 56, 24, 42, 43, 49, 50, 51, 52]

    # Create mask for points to keep (where instance label is not in remove list)
    keep_mask = ~np.isin(instance_labels, remove_instance_label)
    
    # Filter points, colors and labels
    points = points[keep_mask]
    colors = colors[keep_mask] 
    instance_labels = instance_labels[keep_mask]

    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors/255.0)

    unique_labels = np.unique(instance_labels)


    labeled_instance = [7, 8, 17, 35, 37]

    # Create a new color array
    new_colors = np.ones_like(colors) * 230  # Default to light gray (230,230,230)
    
    # Set specific colors for each label in labeled_instance
    color_map = {
        7: [255, 0, 0],    # Red
        8: [0, 255, 0],    # Green
        17: [0, 0, 255],   # Blue
        35: [255, 255, 0], # Yellow
        37: [255, 0, 255]  # Purple
    }
    
    # Set the corresponding color for each labeled instance
    for label, color in color_map.items():
        mask = (instance_labels == label)
        new_colors[mask] = color
    
    # Visualize the result
    visualize_original_point_cloud(points, new_colors)


    #visualize_all_instances(points, instance_labels)
    # Set all point colors to light gray
    # colors = np.ones_like(colors) * 230  # Set RGB value to (230,230,230) for light gray
    # visualize_original_point_cloud(points, colors)






    
    
