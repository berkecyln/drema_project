"""
Generate robot_surf_gaussians PLY files by sampling mesh vertices directly
from the panda URDF using pybullet getMeshData + FK world transforms.

No camera or rendering needed — uses exact CAD geometry + forward kinematics.

Output: assets/robot_gaussians_from_urdf/  with link0.ply ... link10.ply
Gaussians are in world frame at the initial joint configuration.

Usage:
    conda run -n drema_env python tools/generate_robot_gaussians.py
Then set in real_world_simulation.yaml:
    robot_gaussians_dir: "./assets/robot_gaussians_from_urdf"
"""

import os
import numpy as np
import pybullet as p
from plyfile import PlyData, PlyElement

URDF           = "./assets/franka_panda/panda.urdf"
OUT_DIR        = "./assets/robot_gaussians_from_urdf"
INITIAL_POS    = [0.260, 0.008, -0.831]
INITIAL_JOINTS = [-0.017792, -0.760124, 0.019783, -2.342050, 0.029841, 1.541194, 0.753449]

# file index → pybullet link index (-1 = robot base)
CONNECTED_JOINTS = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 8]

POINTS_PER_LINK = 3000

GS_SCALE   = -5.0   # log-scale → ~0.007m disk radius
GS_OPACITY =  2.0   # pre-sigmoid → ~0.88
GS_COLOR   =  0.5   # neutral grey


def get_link_world_transform(robot, joint_idx):
    if joint_idx == -1:
        pos, orn = p.getBasePositionAndOrientation(robot)
    else:
        state = p.getLinkState(robot, joint_idx)
        pos, orn = state[4], state[5]
    R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    return np.array(pos), R


def get_mesh_world_points(robot, joint_idx):
    """Return (N, 3) world-frame vertices for a link via getMeshData + FK."""
    try:
        data = p.getMeshData(robot, joint_idx)
        verts = np.array(data[1], dtype=np.float32)
        if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) == 0:
            return None
    except Exception:
        return None

    pos, R = get_link_world_transform(robot, joint_idx)
    world_pts = (R @ verts.T).T + pos
    # PLY files are loaded with translation=initial_position, so store base-relative coords
    base_pts = world_pts - np.array(INITIAL_POS, dtype=np.float32)
    return base_pts.astype(np.float32)


def finger_proxy_points(robot, joint_idx, n=200):
    """For finger links without mesh data, create a small point cluster at the link origin."""
    pos, _ = get_link_world_transform(robot, joint_idx)
    rng = np.random.default_rng(joint_idx + 42)
    world_pts = rng.normal(0, 0.005, (n, 3)).astype(np.float32) + pos
    return (world_pts - np.array(INITIAL_POS, dtype=np.float32))


def make_ply(points, out_path, n_target=POINTS_PER_LINK):
    n = len(points)
    idx = np.random.choice(n, min(n_target, n), replace=(n < n_target))
    pts = points[idx].astype(np.float32)
    N = len(pts)

    dtype = ([('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
              ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
              ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')]
             + [(f'f_rest_{k}', 'f4') for k in range(45)]
             + [('opacity', 'f4'),
                ('scale_0', 'f4'), ('scale_1', 'f4'),
                ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')])

    data = np.zeros(N, dtype=dtype)
    data['x'], data['y'], data['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
    ctr = pts.mean(axis=0)
    nx = pts - ctr
    norms = np.linalg.norm(nx, axis=1, keepdims=True) + 1e-8
    nx /= norms
    data['nx'], data['ny'], data['nz'] = nx[:, 0], nx[:, 1], nx[:, 2]
    data['f_dc_0'] = data['f_dc_1'] = data['f_dc_2'] = GS_COLOR
    data['opacity'] = GS_OPACITY
    data['scale_0'] = data['scale_1'] = GS_SCALE
    data['rot_0'] = 1.0   # identity quaternion w=1

    PlyData([PlyElement.describe(data, 'vertex')]).write(out_path)
    print(f"  {os.path.basename(out_path)}: {N} Gaussians, center={pts.mean(0).round(3)}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    client = p.connect(p.DIRECT)
    robot = p.loadURDF(URDF, basePosition=INITIAL_POS, useFixedBase=True)
    for i, q in enumerate(INITIAL_JOINTS):
        p.resetJointState(robot, i, q)

    print("Writing PLY files:")
    for file_idx, joint_idx in enumerate(CONNECTED_JOINTS):
        pts = get_mesh_world_points(robot, joint_idx)
        if pts is None:
            # finger or no-mesh link — use proxy cluster at link origin
            pts = finger_proxy_points(robot, joint_idx)
            print(f"  link{file_idx} (joint {joint_idx}): no mesh, using proxy cluster")
        out_path = os.path.join(OUT_DIR, f"link{file_idx}.ply")
        make_ply(pts, out_path)

    p.disconnect(client)

    print(f"\nDone → {OUT_DIR}/")
    print("Set in real_world_simulation.yaml:")
    print(f'  robot_gaussians_dir: "{OUT_DIR}"')


if __name__ == "__main__":
    main()
