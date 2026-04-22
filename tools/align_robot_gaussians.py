"""
Realign robot_surf_gaussians PLY files from the sample-data robot configuration
to the current gello simulation configuration.

The bundled PLY files in assets/robot_surf_gaussians/ were extracted when the
robot base was at [0,0,0] in pybullet and joints were at the sample initial
config.  This script computes per-link SE3 transforms (source link frame →
target link frame) and applies them to positions, normals, and Gaussian
orientation quaternions.

Usage:
    python tools/align_robot_gaussians.py
    # writes corrected files to assets/robot_surf_gaussians_aligned/

Adjust SOURCE_BASE / SOURCE_JOINTS / TARGET_BASE / TARGET_JOINTS to match your
setup if needed.  The defaults assume:
  source: pybullet base=[0,0,0], joints = sample commented-out initial_joint_positions
  target: pybullet base=[0.260,0.008,-0.831], joints = gello initial_joint_positions
"""

import os
import numpy as np
import pybullet as p
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
URDF = "./assets/franka_panda/panda.urdf"
SRC_DIR = "./assets/robot_surf_gaussians"
DST_DIR = "./assets/robot_surf_gaussians_gello"

# Robot config when the PLY files were originally created
SOURCE_BASE   = [0.0, 0.0, 0.0]
SOURCE_JOINTS = [2.050e-07, 0.1745, 1.894e-06, -0.8727, -6.537e-06, 1.2217, 0.7854]

# Our simulation config (from real_world_simulation.yaml)
TARGET_BASE   = [0.260, 0.008, -0.831]
TARGET_JOINTS = [-0.017792, -0.760124, 0.019783, -2.342050, 0.029841, 1.541194, 0.753449]

# Mapping: file index → pybullet joint index (-1 = robot base)
# matches connected_joints: [-1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 8]
CONNECTED_JOINTS = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 8]
# --------------------------------------------------------------------------- #


def get_link_frames(base_pos, joint_pos):
    """Return (positions, rotation_matrices) for all pybullet links."""
    client = p.connect(p.DIRECT)
    robot = p.loadURDF(URDF, basePosition=base_pos, useFixedBase=True)
    for i, q in enumerate(joint_pos):
        p.resetJointState(robot, i, q)

    n = p.getNumJoints(robot)
    positions = []
    rotations = []
    for i in range(n):
        state = p.getLinkState(robot, i)
        positions.append(np.array(state[4]))           # link frame world pos
        rot_mat = np.array(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
        rotations.append(rot_mat)

    # Also store base frame (index -1 handled separately)
    base_state = p.getBasePositionAndOrientation(robot)
    base_pos_np = np.array(base_state[0])
    base_rot = np.array(p.getMatrixFromQuaternion(base_state[1])).reshape(3, 3)

    p.disconnect(client)
    return positions, rotations, base_pos_np, base_rot


def quat_to_mat(q):
    """q = [w, x, y, z] scalar-first → 3×3 rotation matrix."""
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def mat_to_quat_scalar_first(mat):
    """3×3 → [w, x, y, z]."""
    r = R.from_matrix(mat)
    xyzw = r.as_quat()  # scipy: [x,y,z,w]
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def transform_ply(ply_path, delta_R, delta_t, out_path):
    """Apply rigid transform (delta_R, delta_t) to a Gaussian PLY file."""
    ply = PlyData.read(ply_path)
    verts = ply['vertex']

    # --- positions ---------------------------------------------------------
    xyz = np.stack([verts['x'], verts['y'], verts['z']], axis=1)  # (N,3)
    xyz_new = (delta_R @ xyz.T).T + delta_t

    # --- normals -----------------------------------------------------------
    nx = np.stack([verts['nx'], verts['ny'], verts['nz']], axis=1)
    nx_new = (delta_R @ nx.T).T

    # --- Gaussian rotation quaternions (scalar-first: w,x,y,z) ----------
    rot_q = np.stack([verts['rot_0'], verts['rot_1'],
                      verts['rot_2'], verts['rot_3']], axis=1)  # (N,4) w,x,y,z
    rot_mats = np.array([quat_to_mat(q) for q in rot_q])        # (N,3,3)
    rot_mats_new = delta_R @ rot_mats                            # broadcast (3,3)@(N,3,3)
    rot_q_new = np.array([mat_to_quat_scalar_first(m) for m in rot_mats_new])

    # --- rebuild PLY -------------------------------------------------------
    data = {p_.name: verts[p_.name].copy() for p_ in verts.properties}
    data['x'], data['y'], data['z'] = xyz_new[:, 0], xyz_new[:, 1], xyz_new[:, 2]
    data['nx'], data['ny'], data['nz'] = nx_new[:, 0], nx_new[:, 1], nx_new[:, 2]
    data['rot_0'] = rot_q_new[:, 0]
    data['rot_1'] = rot_q_new[:, 1]
    data['rot_2'] = rot_q_new[:, 2]
    data['rot_3'] = rot_q_new[:, 3]

    props = [PlyElement.describe(
        np.array(list(zip(*[data[p_.name] for p_ in verts.properties])),
                 dtype=[(p_.name, verts[p_.name].dtype) for p_ in verts.properties]),
        'vertex')]
    PlyData(props).write(out_path)


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    src_pos, src_rot, src_base_pos, src_base_rot = get_link_frames(SOURCE_BASE, SOURCE_JOINTS)
    tgt_pos, tgt_rot, tgt_base_pos, tgt_base_rot = get_link_frames(TARGET_BASE, TARGET_JOINTS)

    files = sorted(os.listdir(SRC_DIR), key=lambda x: int(x[4:].split('.')[0]))

    for i, fname in enumerate(files):
        joint_idx = CONNECTED_JOINTS[i]

        if joint_idx == -1:
            # Robot base link
            R_src, p_src = src_base_rot, src_base_pos
            R_tgt, p_tgt = tgt_base_rot, tgt_base_pos
        else:
            R_src, p_src = src_rot[joint_idx], src_pos[joint_idx]
            R_tgt, p_tgt = tgt_rot[joint_idx], tgt_pos[joint_idx]

        # delta_R = R_tgt @ R_src.T  (rotate source frame to target frame)
        delta_R = R_tgt @ R_src.T
        # Output in base-relative frame: PLYs are loaded with translation=TARGET_BASE,
        # so subtract TARGET_BASE to avoid double-shifting.
        delta_t = (p_tgt - np.array(TARGET_BASE)) - delta_R @ p_src

        src_path = os.path.join(SRC_DIR, fname)
        dst_path = os.path.join(DST_DIR, fname)
        transform_ply(src_path, delta_R, delta_t, dst_path)
        print(f"  {fname}: delta_t={delta_t.round(3)}, "
              f"delta_R_euler={R.from_matrix(delta_R).as_euler('xyz', degrees=True).round(1)}°")

    print(f"\nAligned files written to {DST_DIR}/")
    print("Update real_world_simulation.yaml:")
    print(f"  robot_gaussians_dir: \"{DST_DIR}\"")


if __name__ == "__main__":
    main()
