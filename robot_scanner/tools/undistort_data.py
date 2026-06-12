"""
Undistort RGB and depth images in a DreMa dataset folder.
Applies cv2.undistort using distortion coefficients and updates intrinsics in pose files.

Distortion coefficients are read from INPUT_DIR/distortion_coeffs.txt if present,
otherwise falls back to COLMAP-estimated values.

Modifies data in-place — duplicate your data before running.

Usage:
    python robot_scanner/tools/undistort_data.py
"""

import os
import glob
import cv2
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_DIR = os.path.join(PROJECT_DIR, "input", "gold_data_undistort")

# COLMAP-estimated distortion coefficients for RealSense D435 (k1, k2, p1, p2)
COLMAP_DIST_COEFFS = np.array([0.12388585, -0.27043119, 0.00075885, -0.00004048])

# alpha=0: crop to valid pixels (no black borders), alpha=1: keep all pixels
ALPHA = 0


def load_dist_coeffs(input_dir):
    dist_file = os.path.join(input_dir, "distortion_coeffs.txt")
    if os.path.exists(dist_file):
        coeffs = np.loadtxt(dist_file)
        print(f"Loaded distortion coefficients from {dist_file}: {coeffs}")
        return coeffs
    print(f"No distortion_coeffs.txt found, using COLMAP estimates: {COLMAP_DIST_COEFFS}")
    return COLMAP_DIST_COEFFS


def read_intrinsics_from_pose(pose_path):
    with open(pose_path, "r") as f:
        lines = f.read().split("\n")
    K = np.eye(3)
    K[0, 0] = float(lines[5].split()[0])
    K[0, 2] = float(lines[5].split()[2])
    K[1, 1] = float(lines[6].split()[1])
    K[1, 2] = float(lines[6].split()[2])
    return K


def update_intrinsics_in_pose(pose_path, K_new):
    with open(pose_path, "r") as f:
        lines = f.read().split("\n")
    extrinsic_lines = lines[:5]
    new_K_lines = [" ".join(f"{v:.6f}" for v in row) for row in K_new]
    with open(pose_path, "w") as f:
        f.write("\n".join(extrinsic_lines) + "\n")
        f.write("\n".join(new_K_lines) + "\n")


def undistort_dataset(input_dir, dist_coeffs, alpha):
    image_dir = os.path.join(input_dir, "images")
    depth_dir = os.path.join(input_dir, "depth_scaled")
    pose_dir = os.path.join(input_dir, "poses")

    image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if not image_files:
        print(f"No images found in {image_dir}")
        return

    first_pose = os.path.join(pose_dir, os.path.basename(image_files[0]).replace(".png", ".txt"))
    K = read_intrinsics_from_pose(first_pose)

    sample_img = cv2.imread(image_files[0])
    h, w = sample_img.shape[:2]
    K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), alpha, (w, h))

    print(f"Dataset: {input_dir}")
    print(f"Images: {len(image_files)}, Resolution: {w}x{h}")
    print(f"Original K: fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
    print(f"New K:      fx={K_new[0,0]:.2f} fy={K_new[1,1]:.2f} cx={K_new[0,2]:.2f} cy={K_new[1,2]:.2f}")

    map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeffs, None, K_new, (w, h), cv2.CV_32FC1)

    for img_path in image_files:
        base = os.path.splitext(os.path.basename(img_path))[0]

        img = cv2.imread(img_path)
        cv2.imwrite(img_path, cv2.remap(img, map1, map2, cv2.INTER_LINEAR))

        depth_path = os.path.join(depth_dir, f"{base}.npy")
        if os.path.exists(depth_path):
            depth = np.load(depth_path)
            np.save(depth_path, cv2.remap(depth, map1, map2, cv2.INTER_NEAREST))

        pose_path = os.path.join(pose_dir, f"{base}.txt")
        if os.path.exists(pose_path):
            update_intrinsics_in_pose(pose_path, K_new)

        print(f"  Undistorted {base}", end="\r")

    print(f"\nDone. {len(image_files)} frames undistorted in-place.")


if __name__ == "__main__":
    dist_coeffs = load_dist_coeffs(INPUT_DIR)
    undistort_dataset(INPUT_DIR, dist_coeffs, ALPHA)
