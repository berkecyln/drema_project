"""
Aggregate RGB-D frames into a single world-frame point cloud.

Raw aggregation:
    python robot_scanner/tools/pointcloud_aggregation.py --input input/<task_dir> --voxel 0.0 --no_outliers

With outlier removal and voxel downsampling:
    python robot_scanner/tools/pointcloud_aggregation.py --input input/<task_dir>
"""

import os
import argparse
import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm


def read_pose_file(path, separator=" "):
    """DreMa pose file: lines 0-2 are [R|t] rows, lines 5-6 carry fx/cx and fy/cy."""
    rotation = np.zeros((3, 3))
    translation = np.zeros(3)
    intrinsics = np.eye(3)
    with open(path, "r") as file:
        lines = file.read().split("\n")

        rotation[0] = np.array(lines[0].split(separator)[:3]).astype(float)
        rotation[1] = np.array(lines[1].split(separator)[:3]).astype(float)
        rotation[2] = np.array(lines[2].split(separator)[:3]).astype(float)

        translation[0] = float(lines[0].split(separator)[3])
        translation[1] = float(lines[1].split(separator)[3])
        translation[2] = float(lines[2].split(separator)[3])

        intrinsics[0,0] = float(lines[5].split(separator)[0])  # fx
        intrinsics[0,2] = float(lines[5].split(separator)[2])  # cx
        intrinsics[1,1] = float(lines[6].split(separator)[1])  # fy
        intrinsics[1,2] = float(lines[6].split(separator)[2])  # cy

    return rotation, translation, intrinsics


def get_files(data_dir, depth_folder_name="depth_scaled"):
    """Collect frames that have all of image, depth and pose present."""
    img_dir = os.path.join(data_dir, "images")
    depth_dir = os.path.join(data_dir, depth_folder_name)
    pose_dir = os.path.join(data_dir, "poses")

    pose_files = sorted([f for f in os.listdir(pose_dir) if f.endswith(".txt")])

    data_samples = []

    for pf in pose_files:
        idx = pf.split('.')[0]

        img_path = os.path.join(img_dir, f"{idx}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, f"{idx}.jpg")

        depth_path = os.path.join(depth_dir, f"{idx}.npy")
        pose_path = os.path.join(pose_dir, pf)

        if os.path.exists(img_path) and os.path.exists(depth_path):
            data_samples.append({
                'id': idx,
                'img': img_path,
                'depth': depth_path,
                'pose': pose_path
            })

    return data_samples

def unproject_depth(depth, K):
    """Depth map to 3D points in camera frame, OpenCV convention (X right, Y down, Z forward)."""
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.flatten()
    v = v.flatten()
    z = depth.flatten()

    valid = z > 0
    u, v, z = u[valid], v[valid], z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points_cam = np.vstack((x, y, z)).T
    return points_cam, valid

def aggregate_pointcloud(
    data_dir,
    depth_folder="depth_scaled",
    voxel_size=0.005,
    depth_scale=1.0,
    max_depth=2.0,
    remove_outliers=True,
    visualize=True,
):
    print(f"Scanning directory: {data_dir}")
    samples = get_files(data_dir, depth_folder)
    print(f"Found {len(samples)} aligned frames.")

    global_pcd = o3d.geometry.PointCloud()

    for sample in tqdm(samples, desc="Aggregating Point Cloud"):
        img_bgr = cv2.imread(sample['img'])
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        depth = np.load(sample['depth']) * depth_scale
        rotation, translation, intrinsics = read_pose_file(sample['pose'])

        c2w = np.eye(4)
        c2w[:3, :3] = rotation
        c2w[:3, 3] = translation

        points_cam, valid_mask = unproject_depth(depth, intrinsics)
        colors = img_rgb.reshape(-1, 3)[valid_mask] / 255.0

        depth_filter = points_cam[:, 2] < max_depth
        points_cam = points_cam[depth_filter]
        colors = colors[depth_filter]

        pcd_frame = o3d.geometry.PointCloud()
        pcd_frame.points = o3d.utility.Vector3dVector(points_cam)
        pcd_frame.colors = o3d.utility.Vector3dVector(colors)
        pcd_frame.transform(c2w)

        global_pcd += pcd_frame

    print(f"Total points raw: {len(global_pcd.points)}")

    # coordinate frame at origin = robot base
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

    if voxel_size > 0:
        print(f"Downsampling with voxel size: {voxel_size}m")
        global_pcd = global_pcd.voxel_down_sample(voxel_size)
        print(f"Total points after downsample: {len(global_pcd.points)}")
    else:
        print("Skipping voxel downsampling.")
    
    if remove_outliers:
        print("Removing outliers...")
        global_pcd, _ = global_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"Total points after outlier removal: {len(global_pcd.points)}")
    else:
        print("Skipping outlier removal.")

    output_path = os.path.join(data_dir, "aggregated_pointcloud.ply")
    ok = o3d.io.write_point_cloud(output_path, global_pcd)
    if ok:
        print(f"Saved aggregated point cloud to: {output_path}")
    else:
        print(f"Failed to save point cloud to: {output_path}")

    if visualize:
        print("Visualizing... (Close window to exit)")
        o3d.visualization.draw_geometries([global_pcd, coord_frame], 
                                          window_name="Aggregated Point Cloud",
                                          width=1024, height=768)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate RGB-D frames into a single World-Frame Point Cloud.")
    parser.add_argument("--input", type=str, required=True, help="Path to the task directory")
    parser.add_argument("--depth_folder", type=str, default="depth_scaled", help="Folder name for depth files (default: depth_scaled)")
    parser.add_argument("--voxel", type=float, default=0.002, help="Voxel size for downsampling (m). Set to 0 to disable.")
    parser.add_argument("--max_depth", type=float, default=1.5, help="Ignore points further than this (m)")
    parser.add_argument("--no_outliers", action="store_true", help="Disable statistical outlier removal")
    parser.add_argument("--no_vis", action="store_true", help="Disable Open3D visualization window")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Directory {args.input} does not exist.")
        exit(1)
        
    aggregate_pointcloud(
        args.input,
        args.depth_folder,
        args.voxel,
        max_depth=args.max_depth,
        remove_outliers=not args.no_outliers,
        visualize=not args.no_vis,
    )