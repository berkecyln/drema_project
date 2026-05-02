"""
Real-robot PerAct inference.

Usage:
    conda activate robotio
    python deploy_peract.py \
        --weights  logs/peract/bottle_pickup/PERACT_BC/seed0/weights \
        --config   logs/peract/bottle_pickup/PERACT_BC/seed0/.hydra/config.yaml

The weights dir contains one .pt file per attention layer (e.g. QAttentionPerActBCAgent_layer0.pt).
The config is the hydra config saved automatically during training.
"""

import argparse
import sys
import os
import numpy as np
import torch
import clip as clip_lib
from PIL import Image as PILImage
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as ScipyRotation

PERACT_ROOT   = '/home/ceylanb/DreMa/drema_project/peract'
PROJECT_ROOT  = '/home/ceylanb/DreMa/drema_project'
ROBOT_IO_CONF = '/home/ceylanb/robot/robot_io/robot_io/conf'
sys.path.insert(0, PERACT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from agents.peract_bc.launch_utils import create_agent

TASK_DESC      = "pick up the dark blue bottle and place it"
IMAGE_SIZE     = (128, 128)
CAMERAS        = ['wrist', 'overhead']
SCENE_BOUNDS   = [0.15, -0.35, 0.10, 0.55, 0.30, 0.75]
DEVICE         = 'cuda:0'
EPISODE_LENGTH = 50


def _backproject(depth_m, K, T_w2c):
    """Backproject depth (H,W) meters to world-frame point cloud (H,W,3). T_w2c inverted to T_c2w."""
    H, W = depth_m.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - K[0, 2]) * depth_m / K[0, 0]
    y = (v - K[1, 2]) * depth_m / K[1, 1]
    xyz_cam = np.stack([x, y, depth_m], axis=-1)
    T_c2w = np.linalg.inv(T_w2c)
    R, t  = T_c2w[:3, :3], T_c2w[:3, 3]
    return (xyz_cam @ R.T + t).astype(np.float32)


def _wrist_T_w2c(robot, T_tcp_cam):
    """Per-step world-to-camera for the wrist camera using current TCP pose + fixed T_tcp_cam."""
    state = robot.get_state()
    T_world_tcp = np.eye(4, dtype=np.float64)
    T_world_tcp[:3, :3] = ScipyRotation.from_quat(state['tcp_orn']).as_matrix()  # tcp_orn: xyzw
    T_world_tcp[:3, 3]  = state['tcp_pos']
    T_world_cam = T_world_tcp @ T_tcp_cam   # T_world←cam
    return np.linalg.inv(T_world_cam).astype(np.float32)


def build_obs_dict(cam_manager, robot, K_wrist, K_overhead, T_tcp_cam,
                   T_overhead_w2c, lang_tokens, step, device):
    images = cam_manager.get_images()
    state  = robot.get_state()

    rgb_w = np.array(PILImage.fromarray(images['rgb_gripper']).resize(IMAGE_SIZE))
    rgb_o = np.array(PILImage.fromarray(images['rgb_static']).resize(IMAGE_SIZE))
    dep_w = np.array(PILImage.fromarray(images['depth_gripper']).resize(IMAGE_SIZE, PILImage.NEAREST))
    dep_o = np.array(PILImage.fromarray(images['depth_static']).resize(IMAGE_SIZE, PILImage.NEAREST))

    # robot_io cameras return float32 meters (RealSense: raw*depth_scale, Kinect: raw/1000)
    dep_w_m = dep_w.astype(np.float32)
    dep_o_m = dep_o.astype(np.float32)

    T_wrist_w2c = _wrist_T_w2c(robot, T_tcp_cam)
    pcd_w = _backproject(dep_w_m, K_wrist,   T_wrist_w2c)
    pcd_o = _backproject(dep_o_m, K_overhead, T_overhead_w2c)

    grip_open   = float(state['gripper_opening_width'] >= 0.078)
    # training data uses discrete [0.04, 0.04] / [0.0, 0.0], not continuous width/2
    grip_joints = np.array([0.04, 0.04] if grip_open else [0.0, 0.0], dtype=np.float32)
    # normalized time: matches peract/helpers/utils.py convention
    time_norm   = (1.0 - step / float(EPISODE_LENGTH - 1)) * 2.0 - 1.0

    def to_tensor(x, add_extra_dim=True):
        t = torch.tensor(x, dtype=torch.float32, device=device)
        return t.unsqueeze(0).unsqueeze(0) if add_extra_dim else t.unsqueeze(0)

    obs = {}
    for cam, rgb, pcd, K, T in [
        ('wrist',    rgb_w, pcd_w, K_wrist,   T_wrist_w2c),
        ('overhead', rgb_o, pcd_o, K_overhead, T_overhead_w2c),
    ]:
        # PreprocessAgent normalizes [0,255]→[-1,1] internally, so pass raw uint8 values
        obs[f'{cam}_rgb']               = torch.tensor(rgb, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
        obs[f'{cam}_point_cloud']       = torch.tensor(pcd, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
        obs[f'{cam}_camera_extrinsics'] = to_tensor(T)
        obs[f'{cam}_camera_intrinsics'] = to_tensor(K)

    obs['low_dim_state']     = torch.tensor(
        [[grip_open, grip_joints[0], grip_joints[1], time_norm]],
        dtype=torch.float32, device=device)
    obs['ignore_collisions'] = torch.tensor([[0.0]], dtype=torch.float32, device=device)
    obs['lang_goal_tokens']  = lang_tokens.to(device)

    return obs


def load_calibration(cam_manager):
    calib = os.path.join(PROJECT_ROOT, 'assets/calibration/calibration_files')

    # T_tcp_cam: camera-to-TCP transform (T_tcp←cam), shape (4,4)
    T_tcp_cam  = np.load(os.path.join(calib, 'panda_realsenseD435_T_tcp_cam.npy')).astype(np.float64)

    # K_wrist from live camera at resize_resolution 640x360 (crop/flip already applied)
    intr   = cam_manager.gripper_cam.get_intrinsics()
    K_wrist = np.array([[intr['fx'], 0, intr['cx']],
                         [0, intr['fy'], intr['cy']],
                         [0, 0, 1]], dtype=np.float32)

    # overhead stored at 640x360
    K_overhead = np.load(os.path.join(calib, 'kinect_overhead_intrinsics.npy')).astype(np.float32)
    T_overhead = np.load(os.path.join(calib, 'kinect_overhead_extrinsics.npy')).astype(np.float32)

    # scale intrinsics from 640x360 to training resolution 128x128
    orig_w, orig_h = 640, 360
    for K in [K_wrist, K_overhead]:
        K[0, 0] *= IMAGE_SIZE[0] / orig_w;  K[0, 2] *= IMAGE_SIZE[0] / orig_w
        K[1, 1] *= IMAGE_SIZE[1] / orig_h;  K[1, 2] *= IMAGE_SIZE[1] / orig_h

    return K_wrist, T_tcp_cam, K_overhead, T_overhead


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True,
                        help='Directory containing .pt weight files from training')
    parser.add_argument('--config', required=True,
                        help='Path to .hydra/config.yaml saved during training')
    parser.add_argument('--steps', type=int, default=EPISODE_LENGTH)
    args = parser.parse_args()

    device = torch.device(DEVICE)

    # reconstruct PerAct agent from training config and load weights
    cfg   = OmegaConf.load(args.config)
    agent = create_agent(cfg)
    agent.build(training=False, device=device)   # sets eval mode internally; do NOT call .eval()
    agent.load_weights(args.weights)
    print("Agent loaded from", args.weights)

    # encode task language once
    lang_tokens = clip_lib.tokenize([TASK_DESC])   # (1, 77)

    # init robot and cameras via Hydra (resolves nested defaults in config files)
    import hydra
    from hydra import initialize_config_dir, compose
    with initialize_config_dir(config_dir=ROBOT_IO_CONF, version_base=None):
        robot_cfg = compose(config_name='robot/panda_franky_interface_policy')
        cams_cfg  = compose(config_name='cams/camera_manager',
                            overrides=['use_static_cam=true', 'threaded_cameras=false'])

    robot       = hydra.utils.instantiate(robot_cfg)
    cam_manager = hydra.utils.instantiate(cams_cfg)

    K_wrist, T_tcp_cam, K_overhead, T_overhead_w2c = load_calibration(cam_manager)

    from robot_io.utils.utils import restrict_workspace, FpsController
    workspace_limits = robot_cfg.workspace_limits
    fps = FpsController(freq=15)

    print(f"Moving to neutral and opening gripper...")
    robot.move_to_neutral()
    robot.open_gripper(blocking=True)

    print(f"Deploying PerAct: '{TASK_DESC}'")
    for step in range(args.steps):
        obs = build_obs_dict(cam_manager, robot, K_wrist, K_overhead, T_tcp_cam,
                             T_overhead_w2c, lang_tokens, step, device)

        with torch.no_grad():
            result = agent.act(step, obs, deterministic=True)

        action    = result.action
        pos       = action[:3]
        quat      = action[3:7]   # xyzw, matches robot_io's quaternion convention
        grip_open = action[7] > 0.5

        pos = restrict_workspace(workspace_limits, pos)
        robot.move_async_cart_pos_abs_lin(pos, quat)
        if grip_open:
            robot.open_gripper()
        else:
            robot.close_gripper()

        fps.step()
        print(f"  step {step:02d}  pos={np.round(pos,3)}  grip={'open' if grip_open else 'close'}")


if __name__ == '__main__':
    main()
