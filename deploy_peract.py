"""
Real-robot PerAct inference.

Usage:
    conda activate robotio
    python deploy_peract.py \
        --weights  logs/peract_100voxel_gripperweight15/bottle_pickup/PERACT_BC/seed0/weights/89900 \
        --config   logs/peract_100voxel_gripperweight15/bottle_pickup/PERACT_BC/seed0/config.yaml

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
sys.path.insert(0, PERACT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from agents.peract_bc.launch_utils import create_agent

TASK_DESC      = "pick up the dark blue bottle"
IMAGE_SIZE     = (128, 128)
CAMERAS        = ['wrist', 'overhead']
SCENE_BOUNDS   = [0.15, -0.35, 0.20, 0.47, 0.30, 0.70]
DEVICE         = 'cuda:0'
EPISODE_LENGTH = 50


def _backproject(depth_m, K, T_w2c):
    """Backproject depth (H,W) meters to world-frame point cloud (H,W,3)."""
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
    T_world_tcp[:3, :3] = ScipyRotation.from_quat(state['tcp_orn']).as_matrix()
    T_world_tcp[:3, 3]  = state['tcp_pos']
    T_world_cam = T_world_tcp @ T_tcp_cam
    return np.linalg.inv(T_world_cam).astype(np.float32)


def _update_dashboard(panels, step, grip_open):
    import cv2
    SCALE  = 3          # 128 → 384 px per image
    W = H  = IMAGE_SIZE[0] * SCALE
    HDR    = 26         # pixel height of label strip above each image

    def _label(img_bgr, text):
        bar = np.zeros((HDR, W, 3), dtype=np.uint8)
        cv2.putText(bar, text, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1, cv2.LINE_AA)
        return np.concatenate([bar, img_bgr], axis=0)

    rows = []
    for cam_label, rgb_raw, rgb_masked in panels:
        raw_bgr    = cv2.resize(rgb_raw[:, :, ::-1],    (W, H), interpolation=cv2.INTER_NEAREST)
        masked_bgr = cv2.resize(rgb_masked[:, :, ::-1], (W, H), interpolation=cv2.INTER_NEAREST)
        row = np.concatenate([_label(raw_bgr,    f'{cam_label}  raw'),
                              _label(masked_bgr, f'{cam_label}  masked')], axis=1)
        rows.append(row)

    dashboard = np.concatenate(rows, axis=0)

    status = np.zeros((28, dashboard.shape[1], 3), dtype=np.uint8)
    cv2.putText(status, f'step {step:02d}   gripper: {"OPEN" if grip_open else "CLOSE"}',
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 220, 120), 1, cv2.LINE_AA)
    dashboard = np.concatenate([dashboard, status], axis=0)

    cv2.imshow('PerAct Deploy', dashboard)
    cv2.waitKey(1)


def build_obs_dict(cam_manager, robot, K_wrist, K_overhead, T_tcp_cam,
                   T_overhead_w2c, lang_tokens, step, device,
                   overhead_only=False, wrist_only=False, overhead_mask=None,
                   show_dashboard=False):
    images = cam_manager.get_images()
    state  = robot.get_state()

    rgb_w = np.array(PILImage.fromarray(images['rgb_gripper']).resize(IMAGE_SIZE))
    dep_w = np.array(PILImage.fromarray(images['depth_gripper']).resize(IMAGE_SIZE, PILImage.NEAREST))
    T_wrist_w2c = _wrist_T_w2c(robot, T_tcp_cam)
    pcd_w = _backproject(dep_w.astype(np.float32), K_wrist, T_wrist_w2c)
    K_w, T_w = K_wrist, T_wrist_w2c

    if wrist_only:
        # model trained without overhead — zero out overhead slot so no unseen data leaks in
        rgb_o_raw = np.zeros((*IMAGE_SIZE[::-1], 3), dtype=np.uint8)
        rgb_o     = rgb_o_raw
        dep_o     = np.zeros(IMAGE_SIZE[::-1], dtype=np.uint8)
        pcd_o     = np.zeros((*IMAGE_SIZE[::-1], 3), dtype=np.float32)
    else:
        rgb_o_raw = np.array(PILImage.fromarray(images['rgb_static']).resize(IMAGE_SIZE))
        rgb_o     = rgb_o_raw.copy()
        dep_o     = np.array(PILImage.fromarray(images['depth_static']).resize(IMAGE_SIZE, PILImage.NEAREST))

        if overhead_mask is not None:
            rgb_o[~overhead_mask] = 0
            dep_o[~overhead_mask] = 0

        pcd_o = _backproject(dep_o.astype(np.float32), K_overhead, T_overhead_w2c)

    if overhead_only:
        rgb_w, pcd_w, K_w, T_w = rgb_o, pcd_o, K_overhead, T_overhead_w2c

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
        ('wrist',    rgb_w, pcd_w, K_w,       T_w),
        ('overhead', rgb_o, pcd_o, K_overhead, T_overhead_w2c),
    ]:
        # PreprocessAgent normalizes [0,255]→[-1,1] internally, so pass raw uint8 values
        obs[f'{cam}_rgb']               = torch.tensor(rgb, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0).unsqueeze(0)
        obs[f'{cam}_point_cloud']       = torch.tensor(pcd, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0).unsqueeze(0)
        obs[f'{cam}_camera_extrinsics'] = to_tensor(T)
        obs[f'{cam}_camera_intrinsics'] = to_tensor(K)

    obs['low_dim_state']     = torch.tensor(
        [[[grip_open, grip_joints[0], grip_joints[1], time_norm]]],
        dtype=torch.float32, device=device)
    obs['ignore_collisions'] = torch.tensor([[[0.0]]], dtype=torch.float32, device=device)
    obs['lang_goal_tokens']  = lang_tokens.to(device)

    if step == 0:
        import cv2
        _dbg = '/tmp/peract_debug'
        os.makedirs(_dbg, exist_ok=True)
        cv2.imwrite(f'{_dbg}/overhead_rgb.png',   rgb_o[:, :, ::-1])
        cv2.imwrite(f'{_dbg}/wrist_rgb.png',      rgb_w[:, :, ::-1])
        cv2.imwrite(f'{_dbg}/overhead_depth.png', (dep_o * 1000).clip(0, 65535).astype(np.uint16))
        cv2.imwrite(f'{_dbg}/wrist_depth.png',    (dep_w * 1000).clip(0, 65535).astype(np.uint16))
        _pcd_o = obs['overhead_point_cloud'][0, 0].cpu().numpy()
        _pcd_w = obs['wrist_point_cloud'][0, 0].cpu().numpy()
        print(f"  [debug] overhead pcd x:[{_pcd_o[0].min():.3f},{_pcd_o[0].max():.3f}]"
              f" y:[{_pcd_o[1].min():.3f},{_pcd_o[1].max():.3f}]"
              f" z:[{_pcd_o[2].min():.3f},{_pcd_o[2].max():.3f}]")
        print(f"  [debug] wrist   pcd x:[{_pcd_w[0].min():.3f},{_pcd_w[0].max():.3f}]"
              f" y:[{_pcd_w[1].min():.3f},{_pcd_w[1].max():.3f}]"
              f" z:[{_pcd_w[2].min():.3f},{_pcd_w[2].max():.3f}]")
        print(f"  [debug] images saved to {_dbg}/")

    if show_dashboard:
        panels = []
        if not wrist_only:
            panels.append(('overhead', rgb_o_raw, rgb_o))
        panels.append(('wrist', rgb_w, rgb_w))  # wrist has no mask
        _update_dashboard(panels, step, grip_open)

    return obs


def load_calibration(cam_manager):
    calib = os.path.join(PROJECT_ROOT, 'assets/calibration/calibration_files')

    # T_tcp_cam: camera-to-TCP transform (T_tcp←cam), shape (4,4)
    T_tcp_cam = np.load(os.path.join(calib, 'panda_realsenseD435_T_tcp_cam.npy')).astype(np.float64)

    # K_wrist from live camera at resize_resolution 640x360 (crop/flip already applied)
    intr    = cam_manager.gripper_cam.get_intrinsics()
    K_wrist = np.array([[intr['fx'], 0, intr['cx']],
                        [0, intr['fy'], intr['cy']],
                        [0, 0, 1]], dtype=np.float32)

    # T_robot_cam = T_c2w (camera-to-world), invert to get T_w2c for backprojection
    T_overhead = np.linalg.inv(
        np.load('/home/ceylanb/robot/robot_io/calibration/calibration_files/panda_azure_kinect_323922212_T_robot_cam_2026_05_11__12_27.npy')
    ).astype(np.float32)
    K_overhead = np.load(os.path.join(calib, 'kinect_overhead_intrinsics.npy')).astype(np.float32)

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
    parser.add_argument('--overhead-only', action='store_true',
                        help='Use only the overhead camera (duplicate for wrist slot)')
    parser.add_argument('--wrist-only', action='store_true',
                        help='Use only the wrist camera (duplicate for overhead slot)')
    args = parser.parse_args()

    device = torch.device(DEVICE)

    cfg   = OmegaConf.load(args.config)
    agent = create_agent(cfg)
    agent.build(training=False, device=device)   # sets eval mode internally; do NOT call .eval()
    agent.load_weights(args.weights)
    print("Agent loaded from", args.weights)

    lang_tokens = clip_lib.tokenize([TASK_DESC]).unsqueeze(0)   # (1, 1, 77)

    import hydra
    from hydra import initialize_config_dir, compose
    DEPLOY_CONF = '/home/ceylanb/DreMa/drema_project/configs'
    with initialize_config_dir(config_dir=DEPLOY_CONF, version_base=None):
        cfg = compose(config_name='deploy')

    robot       = hydra.utils.instantiate(cfg.robot)
    cam_manager = hydra.utils.instantiate(cfg.cams)
    robot_cfg   = cfg.robot

    K_wrist, T_tcp_cam, K_overhead, T_overhead_w2c = load_calibration(cam_manager)

    overhead_mask = np.load(
        os.path.join(PROJECT_ROOT, 'assets/calibration/overhead_mask_128x128.npy')
    )  # bool (128,128), True = valid pixel

    from robot_io.utils.utils import restrict_workspace
    workspace_limits = robot_cfg.workspace_limits

    import cv2
    print("Moving to neutral and opening gripper...")
    robot.move_to_neutral()
    robot.open_gripper(blocking=True)

    print(f"Deploying PerAct: '{TASK_DESC}'")
    print("Dashboard open. Arrange the scene, then press any key in the dashboard window to start.")
    # show first frame so user can verify scene before execution begins
    _first_obs = build_obs_dict(cam_manager, robot, K_wrist, K_overhead, T_tcp_cam,
                                T_overhead_w2c, lang_tokens, 0, device,
                                overhead_only=args.overhead_only,
                                wrist_only=args.wrist_only,
                                overhead_mask=overhead_mask,
                                show_dashboard=True)
    cv2.waitKey(0)  # block until any key pressed

    MAX_RETRIES = 3
    for step in range(args.steps):
        obs = build_obs_dict(cam_manager, robot, K_wrist, K_overhead, T_tcp_cam,
                             T_overhead_w2c, lang_tokens, step, device,
                             overhead_only=args.overhead_only,
                             wrist_only=args.wrist_only,
                             overhead_mask=overhead_mask,
                             show_dashboard=True)

        with torch.no_grad():
            result = agent.act(step, obs, deterministic=True)

        action    = result.action
        pos       = action[:3]
        quat      = action[3:7]   # xyzw, matches robot_io's quaternion convention
        grip_open = action[7] > 0.5

        pos   = restrict_workspace(workspace_limits, pos)
        euler = ScipyRotation.from_quat(quat).as_euler('xyz', degrees=True)
        print(f"  step {step:02d}  pos={np.round(pos,3)}  euler_deg={np.round(euler,1)}  grip={'open' if grip_open else 'close'}"
              f"  raw_pos={np.round(action[:3],3)}")

        for attempt in range(MAX_RETRIES):
            try:
                # LIN applies NE_T_EE internally (same as scanner/arch code)
                robot.move_cart_pos_abs_lin(pos, quat)
                break
            except Exception as e:
                print(f"  [warn] move failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                if attempt + 1 == MAX_RETRIES:
                    print("  [error] max retries reached, moving to neutral and stopping.")
                    robot.move_to_neutral()
                    cv2.destroyAllWindows()
                    return
                print("  [retry] retry...")

        if grip_open:
            robot.open_gripper(blocking=True)
        else:
            robot.close_gripper(blocking=True)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
