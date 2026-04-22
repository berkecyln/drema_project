"""
Convert GELLO-recorded .npy trajectories to DreMa's .pkl format.

Usage:
    python drema/tools/convert_trajectory.py assets/trajectories/pickup_woodencube.npy
    python drema/tools/convert_trajectory.py assets/trajectories/  # convert all

The output .pkl is saved next to the .npy file with the same stem.
Copy the result to your scene's source_path as dictionary.pkl:
    cp assets/trajectories/pickup_woodencube.pkl input/<your_scene>/dictionary.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


GRIPPER_OPEN_WIDTH = [0.04, 0.04]   # metres, Franka fully open
GRIPPER_CLOSE_WIDTH = [0.0, 0.0]    # metres, Franka fully closed


def convert(npy_path: Path) -> Path:
    traj = np.load(npy_path, allow_pickle=True)

    demo = []
    n = len(traj)

    for i, step in enumerate(traj):
        # gripper_pose: [x, y, z, qx, qy, qz, qw]
        gripper_pose = np.concatenate([step["tcp_pos"], step["tcp_orn"]])

        gripper_open = step["gripper"] > 0  # GRIPPER_OPEN=1, GRIPPER_CLOSE=-1; bool(-1)==True is wrong
        gripper_joint_positions = GRIPPER_OPEN_WIDTH if gripper_open else GRIPPER_CLOSE_WIDTH

        # joint_velocities via central finite differences (forward/backward at edges)
        if n == 1:
            joint_velocities = np.zeros(7)
        elif i == 0:
            dt = traj[1]["timestamp"] - traj[0]["timestamp"]
            joint_velocities = (traj[1]["joint_positions"] - traj[0]["joint_positions"]) / max(dt, 1e-6)
        elif i == n - 1:
            dt = traj[n-1]["timestamp"] - traj[n-2]["timestamp"]
            joint_velocities = (traj[n-1]["joint_positions"] - traj[n-2]["joint_positions"]) / max(dt, 1e-6)
        else:
            dt = traj[i+1]["timestamp"] - traj[i-1]["timestamp"]
            joint_velocities = (traj[i+1]["joint_positions"] - traj[i-1]["joint_positions"]) / max(dt, 1e-6)

        demo.append({
            "gripper_pose":             gripper_pose,
            "joint_positions":          step["joint_positions"].copy(),
            "gripper_joint_positions":  np.array(gripper_joint_positions, dtype=np.float32),
            "gripper_open":             gripper_open,
            "joint_velocities":         joint_velocities,
        })

    out_path = npy_path.with_suffix(".pkl")
    with open(out_path, "wb") as f:
        pickle.dump(demo, f)

    print(f"Converted {npy_path.name} ({n} steps) → {out_path.name}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Convert GELLO .npy trajectories to DreMa .pkl format")
    parser.add_argument("path", help="Path to a .npy file or a directory containing .npy files")
    args = parser.parse_args()

    target = Path(args.path)

    if target.is_dir():
        npy_files = sorted(target.glob("*.npy"))
        if not npy_files:
            print(f"No .npy files found in {target}")
            sys.exit(1)
        for f in npy_files:
            convert(f)
    elif target.is_file() and target.suffix == ".npy":
        convert(target)
    else:
        print(f"Expected a .npy file or directory, got: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
