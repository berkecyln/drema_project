"""
Prepares a real-world scene for DreMa data generation:

  1. Converts the GELLO .npy recording to dictionary.pkl
  2. Adds camera fields to every trajectory step
  3. Creates low_dim_obs.pkl, variation_descriptions.pkl, variation_number.pkl

Configure everything in configs/prepare_scene.yaml, then run:

    python tools/prepare_scene_for_generation.py
"""

import os
import pickle
import sys

import types as _types

import hydra
import numpy as np
import pybullet as p
from omegaconf import DictConfig


_GRIPPER_OPEN_WIDTH = np.array([0.04, 0.04], dtype=np.float32)
_GRIPPER_CLOSE_WIDTH = np.array([0.0, 0.0], dtype=np.float32)


def _npy_to_demo(npy_path: str) -> list:
    """Convert a GELLO .npy trajectory to a list of steps in the format expected by DreMa."""
    traj = np.load(npy_path, allow_pickle=True)
    n = len(traj)
    demo = []

    for i, step in enumerate(traj):
        gripper_open = bool(step["gripper"] > 0)

        if n == 1:
            joint_velocities = np.zeros(7)
        elif i == 0:
            dt = max(traj[1]["timestamp"] - traj[0]["timestamp"], 1e-6)
            joint_velocities = (traj[1]["joint_positions"] - traj[0]["joint_positions"]) / dt
        elif i == n - 1:
            dt = max(traj[n-1]["timestamp"] - traj[n-2]["timestamp"], 1e-6)
            joint_velocities = (traj[n-1]["joint_positions"] - traj[n-2]["joint_positions"]) / dt
        else:
            dt = max(traj[i+1]["timestamp"] - traj[i-1]["timestamp"], 1e-6)
            joint_velocities = (traj[i+1]["joint_positions"] - traj[i-1]["joint_positions"]) / dt

        demo.append({
            "gripper_pose":            np.concatenate([step["tcp_pos"], step["tcp_orn"]]),
            "joint_positions":         step["joint_positions"].copy(),
            "gripper_joint_positions": _GRIPPER_OPEN_WIDTH.copy() if gripper_open else _GRIPPER_CLOSE_WIDTH.copy(),
            "gripper_open":            gripper_open,
            "joint_velocities":        joint_velocities,
        })

    return demo


def _compute_wrist_extrinsics_w2c(cfg) -> np.ndarray:
    """Compute wrist camera T_w2c at initial joint configuration using PyBullet FK.

    The URDF panda_wrist_hand joint origin encodes T_tcp_cam, so link wrist_camera_index
    already represents the physical camera pose — no separate calibration file needed.
    """
    robot_cfg = cfg.simulation.robot
    client = p.connect(p.DIRECT)
    robot_id = p.loadURDF(
        robot_cfg.robot_urdf,
        list(robot_cfg.initial_position),
        useFixedBase=True,
    )

    for i, jv in enumerate(robot_cfg.initial_joint_positions):
        p.resetJointState(robot_id, i, jv)
    p.resetJointState(robot_id, 7, 0)
    p.resetJointState(robot_id, 8, robot_cfg.initial_gripper_joint_positions[0])
    p.resetJointState(robot_id, 9, robot_cfg.initial_gripper_joint_positions[1])

    link_state = p.getLinkState(robot_id, robot_cfg.wrist_camera_index)
    pos_world = np.array(link_state[0])
    R_c2w = np.array(p.getMatrixFromQuaternion(link_state[1])).reshape(3, 3)
    p.disconnect()

    # Invert camera-to-world to get world-to-camera
    T_w2c = np.eye(4)
    T_w2c[:3, :3] = R_c2w.T
    T_w2c[:3, 3] = -R_c2w.T @ pos_world
    return T_w2c


def _read_intrinsics_from_poses(poses_dir: str) -> np.ndarray:
    '''Read camera intrinsics from the first .txt file in the poses directory.'''
    files = sorted(f for f in os.listdir(poses_dir) if f.endswith(".txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files in {poses_dir}")
    with open(os.path.join(poses_dir, files[0])) as f:
        lines = f.read().split("\n")
    # lines 0-3: extrinsics  |  line 4: blank  |  lines 5-7: intrinsics
    fx = float(lines[5].split()[0]);  cx = float(lines[5].split()[2])
    fy = float(lines[6].split()[1]);  cy = float(lines[6].split()[2])
    return np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])


import types as _types

def _make_low_dim_obs():
    # SimpleNamespace is stdlib — always unpicklable in any context
    return _types.SimpleNamespace(random_seed=np.random.get_state())


@hydra.main(version_base=None, config_path="../configs", config_name="prepare_scene")
def main(cfg: DictConfig) -> None:
    #print(OmegaConf.to_yaml(cfg))

    scene = cfg.data.source_path
    task = cfg.task

    # 1 - convert .npy to pkl
    npy_path = task.trajectory_npy
    if not os.path.isfile(npy_path):
        print(f"ERROR: trajectory_npy not found: {npy_path}")
        sys.exit(1)
    demo = _npy_to_demo(npy_path)
    print(f"Converted {npy_path}  ({len(demo)} steps)")

    # 2 - wrist camera
    K_wrist = _read_intrinsics_from_poses(os.path.join(scene, "poses"))
    T_wrist = _compute_wrist_extrinsics_w2c(cfg)
    # print(f"Wrist K:\n{K_wrist}")
    # print(f"Wrist T_w2c:\n{T_wrist}")

    # 3 - overhead camera
    T_overhead = None
    K_overhead = None
    if task.overhead.enabled:
        if task.overhead.extrinsics_file is None or task.overhead.intrinsics_file is None:
            print("ERROR: overhead.enabled is true but extrinsics_file / intrinsics_file are null")
            sys.exit(1)
        T_overhead = np.load(task.overhead.extrinsics_file)
        K_overhead = np.load(task.overhead.intrinsics_file)
        assert T_overhead.shape == (4, 4)
        assert K_overhead.shape == (3, 3)
        print("Overhead camera loaded")
    else:
        print("Overhead camera disabled - wrist only")

    # 4 - read actual image dimensions from the recorded images
    images_dir = os.path.join(scene, "images")
    sample_imgs = sorted(f for f in os.listdir(images_dir) if f.endswith(".png"))
    if not sample_imgs:
        print(f"ERROR: no images found in {images_dir}")
        sys.exit(1)
    from PIL import Image as _Image
    with _Image.open(os.path.join(images_dir, sample_imgs[0])) as _img:
        wrist_w, wrist_h = _img.size  # PIL returns (width, height)
    print(f"Wrist camera image size: {wrist_w}x{wrist_h}")

    # 5 - patch every step
    near = float(task.camera.near)
    far = float(task.camera.far)
    crop_top = int(task.camera.get("crop_top", 0))

    for step in demo:
        step["joint_forces"] = np.zeros(7)
        step["ignore_collisions"] = np.array([False])

        step["wrist_camera_intrinsics"] = K_wrist.copy()
        step["wrist_camera_extrinsics"] = T_wrist.copy()
        step["wrist_camera_near"] = near
        step["wrist_camera_far"] = far
        step["wrist_camera_width"] = wrist_w
        step["wrist_camera_height"] = wrist_h       # post-crop height (360)
        step["wrist_camera_crop_top"] = crop_top    # rows to remove from top of rendered image

        if T_overhead is not None:
            step["overhead_camera_intrinsics"] = K_overhead.copy()
            step["overhead_camera_extrinsics"] = T_overhead.copy()
            step["overhead_camera_near"] = near
            step["overhead_camera_far"] = far

    cameras = ["wrist"] + (["overhead"] if T_overhead is not None else [])
    print(f"Patched {len(demo)} steps - cameras: {cameras}")

    # 6 - save dictionary.pkl
    dict_path = os.path.join(scene, "dictionary.pkl")
    with open(dict_path, "wb") as f:
        pickle.dump(demo, f)
    print(f"Saved {dict_path}")

    # 7 - auxiliary files
    with open(os.path.join(scene, "low_dim_obs.pkl"), "wb") as f:
        pickle.dump(_make_low_dim_obs(), f)

    with open(os.path.join(scene, "variation_descriptions.pkl"), "wb") as f:
        pickle.dump([task.description], f)

    with open(os.path.join(scene, "variation_number.pkl"), "wb") as f:
        pickle.dump(0, f)

    print(f"Created auxiliary pkl files in {scene}")
    print("\nDone. Next steps:")
    print("  1. Set generation.generate_data: true in configs/simulation/real_world_simulation.yaml")
    print("  2. python generate_new_data.py")


if __name__ == "__main__":
    main()
