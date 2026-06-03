"""
PerAct inference inside DreMa's PyBullet+GS simulation.

Mirrors deploy_peract.py but replaces real robot/cameras with the sim
environment so the same agent can be evaluated as a baseline.

Usage:
    conda activate drema_env
    python simulate_peract.py \
        [--weights assets/agents/peract_100voxel_gripperweight15] \
        [--config  assets/agents/peract_100voxel_gripperweight15/config.yaml] \
        [--scene   input/gello_bottle1_rawtsdf] \
        [--steps   50]
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as ScipyRotation
from hydra import initialize_config_dir, compose

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

for _candidate in [
    os.path.join(PROJECT_ROOT, 'peract'),
    os.path.join(PROJECT_ROOT, '..', 'peract'),
]:
    if os.path.isdir(_candidate):
        PERACT_ROOT = os.path.abspath(_candidate)
        break

sys.path.insert(0, PERACT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from agents.peract_bc.launch_utils import create_agent
from helpers.clip.core.clip import tokenize as clip_tokenize
from drema.environment.builder import Builder
from drema.environment.observer.camera import CameraManager

TASK_DESC      = "pick up the dark blue bottle"
IMAGE_SIZE     = (128, 128)
SCENE_BOUNDS   = [0.15, -0.35, 0.20, 0.47, 0.30, 0.70]
DEVICE         = 'cuda:0'
EPISODE_LENGTH = 50
PHYSICS_STEPS  = 240

DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, 'assets/agents/peract_100voxel_gripperweight15')
DEFAULT_CONFIG  = os.path.join(PROJECT_ROOT, 'assets/agents/peract_100voxel_gripperweight15/config.yaml')
DEFAULT_SCENE   = os.path.join(PROJECT_ROOT, 'input/gello_bottle1_rawtsdf')


# ---------------------------------------------------------------------------
# geometry helpers (same math as deploy_peract.py)
# ---------------------------------------------------------------------------

def _backproject(depth_m, K, T_w2c):
    """Backproject float32 depth (H,W) in metres to world-frame point cloud (H,W,3)."""
    H, W = depth_m.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - K[0, 2]) * depth_m / K[0, 0]
    y = (v - K[1, 2]) * depth_m / K[1, 1]
    xyz_cam = np.stack([x, y, depth_m], axis=-1)
    T_c2w = np.linalg.inv(T_w2c)
    R, t  = T_c2w[:3, :3], T_c2w[:3, 3]
    return (xyz_cam @ R.T + t).astype(np.float32)


def _cam_to_T_w2c(cam):
    """Build T_w2c (4x4 float32) from a CameraWrapper.

    CameraWrapper stores rotation=R_c2w and translation=t_w2c, so:
      R_w2c = R_c2w.T,  t_w2c = cam.translation
    """
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = cam.rotation.T   # R_w2c
    T[:3, 3]  = cam.translation  # t_w2c
    return T


def _scale_K(K, src_hw, dst_hw):
    """Scale intrinsics from src_hw to dst_hw (both (H,W))."""
    K = K.copy().astype(np.float32)
    K[0, 0] *= dst_hw[1] / src_hw[1];  K[0, 2] *= dst_hw[1] / src_hw[1]
    K[1, 1] *= dst_hw[0] / src_hw[0];  K[1, 2] *= dst_hw[0] / src_hw[0]
    return K


# ---------------------------------------------------------------------------
# observation building
# ---------------------------------------------------------------------------

def build_obs(env, cam_manager, lang_tokens, step, grip_open, device, overhead_mask):
    overhead_cam = cam_manager.simulation_cameras['overhead']
    wrist_cam    = cam_manager.simulation_cameras['wrist']

    # update wrist extrinsics from current PyBullet link state
    wrist_pos, wrist_rot = env.get_wrist_camera_extrinsics()
    cam_manager.update_camera_extrinsics('wrist', wrist_rot, wrist_pos)

    # render — compress_depth=False keeps raw float32 metres from GS
    rgbs, depths = env.render_cameras(
        [overhead_cam, wrist_cam],
        filter_depth=True, compress_depth=False,
    )
    rgb_o_raw, rgb_w_raw = rgbs[0],  rgbs[1]
    dep_o_raw, dep_w_raw = depths[0], depths[1]

    src_hw_o = rgb_o_raw.shape[:2]
    src_hw_w = rgb_w_raw.shape[:2]

    # resize to training resolution
    rgb_o = cv2.resize(rgb_o_raw, IMAGE_SIZE)
    rgb_w = cv2.resize(rgb_w_raw, IMAGE_SIZE)
    dep_o = cv2.resize(dep_o_raw, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
    dep_w = cv2.resize(dep_w_raw, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

    # zero out robot-arm region in overhead — same static mask used in training
    rgb_o[~overhead_mask] = 0
    dep_o[~overhead_mask] = 0

    dst_hw = (IMAGE_SIZE[1], IMAGE_SIZE[0])  # (H, W)
    K_o = _scale_K(overhead_cam.intrinsics, src_hw_o, dst_hw)
    K_w = _scale_K(wrist_cam.intrinsics,    src_hw_w, dst_hw)

    T_o = _cam_to_T_w2c(overhead_cam)
    T_w = _cam_to_T_w2c(wrist_cam)

    pcd_o = _backproject(dep_o, K_o, T_o)
    pcd_w = _backproject(dep_w, K_w, T_w)

    grip_joints = np.array([0.04, 0.04] if grip_open else [0.0, 0.0], dtype=np.float32)
    time_norm   = (1.0 - step / float(EPISODE_LENGTH - 1)) * 2.0 - 1.0

    def to_tensor(x):
        return torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    obs = {}
    for name, rgb, pcd, K, T in [
        ('overhead', rgb_o, pcd_o, K_o, T_o),
        ('wrist',    rgb_w, pcd_w, K_w, T_w),
    ]:
        # PreprocessAgent normalises [0,255]→[-1,1] internally, so pass raw uint8 values
        obs[f'{name}_rgb']               = torch.tensor(rgb, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0).unsqueeze(0)
        obs[f'{name}_point_cloud']       = torch.tensor(pcd, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0).unsqueeze(0)
        obs[f'{name}_camera_extrinsics'] = to_tensor(T)
        obs[f'{name}_camera_intrinsics'] = to_tensor(K)

    obs['low_dim_state']     = torch.tensor(
        [[[float(grip_open), grip_joints[0], grip_joints[1], time_norm]]],
        dtype=torch.float32, device=device)
    obs['ignore_collisions'] = torch.tensor([[[0.0]]], dtype=torch.float32, device=device)
    obs['lang_goal_tokens']  = lang_tokens.to(device)

    return obs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=DEFAULT_WEIGHTS,
                        help='Directory containing .pt weight files')
    parser.add_argument('--config',  default=DEFAULT_CONFIG,
                        help='Path to config.yaml saved during training')
    parser.add_argument('--scene',   default=DEFAULT_SCENE,
                        help='Path to scene directory (e.g. input/gello_bottle1_rawtsdf)')
    parser.add_argument('--steps',   type=int, default=EPISODE_LENGTH)
    args = parser.parse_args()

    device = torch.device(DEVICE)

    # --- load PerAct agent (identical to deploy_peract.py) ---
    peract_cfg = OmegaConf.load(args.config)
    agent = create_agent(peract_cfg)
    agent.build(training=False, device=device)
    agent.load_weights(args.weights)
    print(f"Agent loaded from {args.weights}")

    lang_tokens = clip_tokenize([TASK_DESC]).unsqueeze(0)   # (1, 1, 77)

    # --- build DreMa sim environment via Hydra ---
    with initialize_config_dir(
        config_dir=os.path.join(PROJECT_ROOT, 'configs'), version_base=None
    ):
        cfg = compose(
            config_name='config_real',
            overrides=[
                f'data.source_path={args.scene}',
                'simulation.robot.simulate_robot=True',
                'simulation.generation.generate_data=False',
                'simulation.visualization.visualize=True',
                'simulation.visualization.orbit_camera=False',
                'simulation.trajectory.load_trajectory=True',
            ],
        )

    builder    = Builder(cfg)
    trajectory = builder.load_trajectory()
    # pass trajectory=None so env.step() doesn't try to follow demo waypoints
    env        = builder.create_environment(trajectory=None)
    env.build_environment()

    # register overhead + wrist cameras at scale=1 (stored native resolution)
    cam_manager = CameraManager()
    for cam_name, params in trajectory.get_cameras().items():
        name, extrinsics, intrinsics, far, near, w, h, crop_top = params
        R_c2w = extrinsics[:3, :3].T
        t_w2c = extrinsics[:3, 3]
        width  = int(w) if w is not None else int(intrinsics[0, 2] * 2)
        height = int(h) if h is not None else int(intrinsics[1, 2] * 2)
        cam_manager.add_simulation_camera(
            0, 1, name, R_c2w, t_w2c, intrinsics.copy(), width, height, near, far, crop_top,
        )

    overhead_mask = np.load(
        os.path.join(PROJECT_ROOT, 'assets/calibration/overhead_mask_128x128.npy')
    )  # bool (128, 128), True = valid pixel

    # settle physics before inference starts (env.trajectory is None so step() is safe)
    for _ in range(240):
        env.client.stepSimulation()
    env.update_state()

    print(f"\nSim PerAct: '{TASK_DESC}'  ({args.steps} steps)\n")
    grip_open = True
    for step in range(args.steps):
        obs = build_obs(env, cam_manager, lang_tokens, step, grip_open, device, overhead_mask)

        with torch.no_grad():
            result = agent.act(step, obs, deterministic=True)

        action    = result.action
        pos       = action[:3]
        quat      = action[3:7]   # xyzw — matches ScipyRotation convention
        grip_open = action[7] > 0.5

        euler = ScipyRotation.from_quat(quat).as_euler('xyz', degrees=True)
        print(f"  step {step:02d}  pos={np.round(pos,3)}  euler_deg={np.round(euler,1)}  grip={'open' if grip_open else 'close'}")

        target_rot = ScipyRotation.from_quat(quat)
        env.robot.move_to_pose(env.client, pos, target_rot)

        if grip_open:
            env.robot.open_gripper(env.client)
        else:
            env.robot.close_gripper(env.client)

        for _ in range(PHYSICS_STEPS):
            env.client.stepSimulation()
        env.update_state()


if __name__ == '__main__':
    main()
