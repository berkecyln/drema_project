"""
One-time setup script for the overhead Azure Kinect camera.

Generates two files needed by prepare_scene_for_generation.py:
  - kinect_extrinsics.npy  : 4x4 world-to-camera matrix in DreMa sim world frame
  - kinect_intrinsics.npy  : 3x3 intrinsics matrix scaled to 640x360 output

Requires the Kinect to be physically connected (uses robotio conda env).

Run from the drema_project directory:
    conda activate robotio
    python tools/prepare_kinect_calibration.py
"""

import os
import sys
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

CALIBRATION_FILE = (
    "/home/gunnleif/Projects/drema/drema_project/assets/calibration/calibration_files/"
    "panda_azure_kinect_323922212_T_robot_cam_2026_04_27__12_12.npy"
)

# Robot physical base center (center of Joint 1 rotation axis at the mounting surface).
# - X: 0.260 (initial) - 0.268 (URDF joint offset) = -0.008
# - Y: 0.008 (initial) - 0.006 (URDF joint offset) = 0.002
# - Z: -0.831 (initial) + 0.820 (URDF mesh offset) = -0.011
ROBOT_PHYSICAL_BASE_POSITION = np.array([-0.008, 0.002, -0.011])

# Native Kinect 720p resolution
NATIVE_W, NATIVE_H = 1280, 720

# Target output resolution (must match kinect4.yaml resize_resolution and prepare_scene.yaml)
TARGET_W, TARGET_H = 640, 360

OUTPUT_DIR = (
    "/home/gunnleif/Projects/drema/drema_project/assets/calibration/calibration_files"
)

# ── Extrinsics ────────────────────────────────────────────────────────────────

def compute_extrinsics(calibration_file, robot_position):
    """
    Convert T_robot_cam to T_w2c_sim (DreMa simulation world frame -> camera frame).

    T_robot_cam (from robot_io calibration) is a c2w matrix: its translation column
    directly gives the camera position in robot base frame. So inv(T_robot_cam) is w2c
    in robot base, then we shift for the robot's position in the DreMa world.

      T_w2c = inv(T_robot_cam) @ T_world2robotbase
    """
    T_robot_cam = np.load(calibration_file)
    assert T_robot_cam.shape == (4, 4), f"Expected 4x4, got {T_robot_cam.shape}"

    T_world2rb = np.eye(4)
    T_world2rb[:3, 3] = -robot_position

    return np.linalg.inv(T_robot_cam) @ T_world2rb


# ── Intrinsics ────────────────────────────────────────────────────────────────

def compute_intrinsics_from_device():
    """
    Initialize the Kinect via robot_io, read factory-calibrated intrinsics at
    native 720p, then scale to TARGET_W x TARGET_H (no crop applied).
    """
    sys.path.insert(0, "/home/gunnleif/Projects/drema/robot/robot_io")
    from robot_io.cams.kinect4.kinect4 import Kinect4

    print("  Connecting to Kinect (device 0)...")
    cam = Kinect4(device=0, resolution="720p", undistort_image=False)
    K_raw = cam.get_camera_matrix()  # 3x3 at native 1280x720
    del cam
    print(f"  Raw camera matrix (1280x720):\n{K_raw}")

    scale_x = TARGET_W / NATIVE_W  # 640 / 1280 = 0.5
    scale_y = TARGET_H / NATIVE_H  # 360 / 720  = 0.5

    K_out = K_raw.copy()
    K_out[0, 0] *= scale_x  # fx
    K_out[1, 1] *= scale_y  # fy
    K_out[0, 2] *= scale_x  # cx
    K_out[1, 2] *= scale_y  # cy

    return K_out


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Extrinsics ===")
    T_w2c = compute_extrinsics(CALIBRATION_FILE, ROBOT_PHYSICAL_BASE_POSITION)
    print(f"T_w2c_sim:\n{T_w2c}")
    extrinsics_path = os.path.join(OUTPUT_DIR, "kinect_overhead_extrinsics.npy")
    np.save(extrinsics_path, T_w2c)
    print(f"Saved: {extrinsics_path}\n")

    print("=== Intrinsics ===")
    try:
        K = compute_intrinsics_from_device()
        print(f"K ({TARGET_W}x{TARGET_H}):\n{K}")
        intrinsics_path = os.path.join(OUTPUT_DIR, "kinect_overhead_intrinsics.npy")
        np.save(intrinsics_path, K)
        print(f"Saved: {intrinsics_path}\n")
    except Exception as e:
        print(f"  Warning: Could not read intrinsics from device ({e})")
        print("  Using existing kinect_overhead_intrinsics.npy if available.")

    print("=== Done ===")
    print("Update configs/prepare_scene.yaml:")
    print(f"  extrinsics_file: {extrinsics_path}")
    print(f"  intrinsics_file: {os.path.join(OUTPUT_DIR, 'kinect_overhead_intrinsics.npy')}")
