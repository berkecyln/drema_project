"""
PerAct inference inside DreMa's PyBullet + GS simulation.

Sim counterpart of deploy_peract.py: same agent, but the real robot/cameras are
replaced by the DreMa environment so the policy can be evaluated as a baseline.

    conda activate drema_env
    python simulate_peract.py --weights <dir> --config <config.yaml> --scene <dir> [--gs_view]
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
sys.path[:0] = [os.path.join(PROJECT_ROOT, 'peract'), PROJECT_ROOT]

from agents.peract_bc.launch_utils import create_agent
from helpers.clip.core.clip import tokenize as clip_tokenize
from drema.environment.builder import Builder
from drema.environment.observer.camera import CameraManager
from drema.utils.utils import prepare_depth

TASK_DESC        = "pick up the dark blue bottle"
IMAGE_SIZE       = (128, 128)
DEVICE           = 'cuda:0'
EPISODE_LENGTH   = 50
PHYSICS_STEPS    = 240
GRIPPER_Z_OFFSET = 0.02            # lower the grasp to offset shorter sim fingers
BOTTLE_SHIFT     = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # +Y = robot's left; zeros to disable

DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, 'assets/agents/peract_100voxel_gripperweight15')
DEFAULT_CONFIG  = os.path.join(DEFAULT_WEIGHTS, 'config.yaml')
DEFAULT_SCENE   = os.path.join(PROJECT_ROOT, 'input/gello_bottle1_rawtsdf')



def _backproject(depth_m, K, T_w2c):
    """Backproject float32 depth (H,W) in metres to a world-frame point cloud (H,W,3)."""
    H, W = depth_m.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - K[0, 2]) * depth_m / K[0, 0]
    y = (v - K[1, 2]) * depth_m / K[1, 1]
    xyz_cam = np.stack([x, y, depth_m], axis=-1)
    T_c2w = np.linalg.inv(T_w2c)
    return (xyz_cam @ T_c2w[:3, :3].T + T_c2w[:3, 3]).astype(np.float32)


def _cam_to_T_w2c(cam):
    """T_w2c (4x4) from a CameraWrapper (stores rotation=R_c2w, translation=t_w2c)."""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = cam.rotation.T
    T[:3, 3]  = cam.translation
    return T


def _scale_K(K, src_hw, dst_hw):
    """Scale intrinsics from src_hw to dst_hw (both (H,W))."""
    K = K.copy().astype(np.float32)
    K[0, 0] *= dst_hw[1] / src_hw[1];  K[0, 2] *= dst_hw[1] / src_hw[1]
    K[1, 1] *= dst_hw[0] / src_hw[0];  K[1, 2] *= dst_hw[0] / src_hw[0]
    return K


def show_gs_view(env, cam):
    """Render one camera's GS view into cv2 windows; return the pressed key code."""
    rgbs, depths = env.render_cameras([cam], filter_depth=True, compress_depth=False)
    cv2.imshow("RGB", cv2.cvtColor(rgbs[0], cv2.COLOR_RGB2BGR))
    cv2.imshow("Depth", prepare_depth(depths[0]))
    return cv2.waitKey(1) & 0xFF


def build_obs(env, cam_manager, lang_tokens, step, grip_open, device):
    overhead_cam = cam_manager.simulation_cameras['overhead']
    wrist_cam    = cam_manager.simulation_cameras['wrist']

    # refresh wrist extrinsics from the current PyBullet link state
    wrist_pos, wrist_rot = env.get_wrist_camera_extrinsics()
    cam_manager.update_camera_extrinsics('wrist', wrist_rot, wrist_pos)

    rgbs, depths = env.render_cameras([overhead_cam, wrist_cam],
                                      filter_depth=True, compress_depth=False)

    dst_hw = (IMAGE_SIZE[1], IMAGE_SIZE[0])  # (H, W)

    def img_tensor(x):   # (H,W,C) -> (1,1,C,H,W)
        return torch.tensor(x, dtype=torch.float32, device=device).permute(2, 0, 1)[None, None]

    def mat_tensor(x):   # (.,.) -> (1,1,.,.)
        return torch.tensor(x, dtype=torch.float32, device=device)[None, None]

    obs = {}
    for name, cam, rgb_raw, dep_raw in [
        ('overhead', overhead_cam, rgbs[0], depths[0]),
        ('wrist',    wrist_cam,    rgbs[1], depths[1]),
    ]:
        rgb = cv2.resize(rgb_raw, IMAGE_SIZE)
        dep = cv2.resize(dep_raw, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
        K   = _scale_K(cam.intrinsics, rgb_raw.shape[:2], dst_hw)
        T   = _cam_to_T_w2c(cam)
        pcd = _backproject(dep, K, T)

        # PreprocessAgent normalises [0,255]->[-1,1] internally, so pass raw values
        obs[f'{name}_rgb']               = img_tensor(rgb)
        obs[f'{name}_point_cloud']       = img_tensor(pcd)
        obs[f'{name}_camera_extrinsics'] = mat_tensor(T)
        obs[f'{name}_camera_intrinsics'] = mat_tensor(K)

    grip_joints = [0.04, 0.04] if grip_open else [0.0, 0.0]
    time_norm   = (1.0 - step / float(EPISODE_LENGTH - 1)) * 2.0 - 1.0
    obs['low_dim_state']     = torch.tensor([[[float(grip_open), *grip_joints, time_norm]]],
                                            dtype=torch.float32, device=device)
    obs['ignore_collisions'] = torch.tensor([[[0.0]]], dtype=torch.float32, device=device)
    obs['lang_goal_tokens']  = lang_tokens.to(device)
    return obs

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=DEFAULT_WEIGHTS, help='Directory with .pt weights')
    parser.add_argument('--config',  default=DEFAULT_CONFIG,  help='Training config.yaml')
    parser.add_argument('--scene',   default=DEFAULT_SCENE,   help='Scene directory')
    parser.add_argument('--steps',   type=int, default=EPISODE_LENGTH)
    parser.add_argument('--gs_view', action='store_true',
                        help='Show live GS RGB/Depth windows (default: fast, no rendering)')
    args = parser.parse_args()

    device = torch.device(DEVICE)

    # --- agent ---
    agent = create_agent(OmegaConf.load(args.config))
    agent.build(training=False, device=device)
    agent.load_weights(args.weights)
    lang_tokens = clip_tokenize([TASK_DESC]).unsqueeze(0)   # (1, 1, 77)
    print(f"Agent loaded from {args.weights}")

    # --- DreMa sim environment ---
    with initialize_config_dir(config_dir=os.path.join(PROJECT_ROOT, 'configs'), version_base=None):
        cfg = compose(config_name='config_real', overrides=[
            f'data.source_path={args.scene}',
            'simulation.robot.simulate_robot=True',
            'simulation.generation.generate_data=False',
            'simulation.visualization.visualize=True',
            'simulation.visualization.orbit_camera=False',
            'simulation.trajectory.load_trajectory=True',
        ])

    builder    = Builder(cfg)
    trajectory = builder.load_trajectory()
    env        = builder.create_environment(trajectory=None)
    env.build_environment()

    # optional: shift the bottle, moving its physics body and gaussians together
    if np.any(BOTTLE_SHIFT):
        bottle  = env.objects['1']
        new_pos = bottle.initial_position + BOTTLE_SHIFT
        env.client.resetBasePositionAndOrientation(bottle.id, new_pos, bottle.initial_orientation)
        bottle.position = bottle.previous_position = new_pos.copy()
        env.gs.translate(bottle.get_gaussians_mask(),
                         torch.tensor(BOTTLE_SHIFT, dtype=torch.float32, device=device))
        print(f"Moved bottle by {BOTTLE_SHIFT} -> {np.round(new_pos, 3)}")

    # register overhead + wrist cameras at native resolution
    cam_manager = CameraManager()
    for _, params in trajectory.get_cameras().items():
        name, extrinsics, intrinsics, far, near, w, h, crop_top = params
        width  = int(w) if w is not None else int(intrinsics[0, 2] * 2)
        height = int(h) if h is not None else int(intrinsics[1, 2] * 2)
        cam_manager.add_simulation_camera(0, 1, name, extrinsics[:3, :3].T, extrinsics[:3, 3],
                                          intrinsics.copy(), width, height, near, far, crop_top)
    overhead_cam = cam_manager.simulation_cameras['overhead']

    # settle physics before inference
    for _ in range(PHYSICS_STEPS):
        env.client.stepSimulation()
    env.update_state()

    hint = "   [ESC in a GS window to quit]" if args.gs_view else ""
    print(f"\nSim PerAct: '{TASK_DESC}'  ({args.steps} steps){hint}\n")

    grip_open, grasped = True, False
    for step in range(args.steps):
        obs    = build_obs(env, cam_manager, lang_tokens, step, grip_open, device)
        action = agent.act(step, obs, deterministic=True).action

        pos       = action[:3].copy()
        quat      = action[3:7]                 # xyzw
        grip_open = action[7] > 0.5
        if not grasped:                         # only offset while reaching to grab
            pos[2] -= GRIPPER_Z_OFFSET
        if not grip_open:
            grasped = True

        euler = ScipyRotation.from_quat(quat).as_euler('xyz', degrees=True)
        print(f"  step {step:02d}  pos={np.round(pos,3)}  euler_deg={np.round(euler,1)}  "
              f"grip={'open' if grip_open else 'close'}")

        env.robot.move_to_pose(env.client, pos, ScipyRotation.from_quat(quat))
        (env.robot.open_gripper if grip_open else env.robot.close_gripper)(env.client)

        for i in range(PHYSICS_STEPS):
            env.client.stepSimulation()
            if args.gs_view and i % 8 == 0 and show_gs_view(env, overhead_cam) == 27:  # ESC
                print("ESC pressed — exiting.")
                cv2.destroyAllWindows()
                return
        env.update_state()

    if args.gs_view:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
