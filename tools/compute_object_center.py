"""
Compute the Gaussian center of a segmented object in a DreMa scene.
Use the output as rotation_center in configs/simulation/real_world_simulation.yaml.

Usage:
    python tools/compute_object_center.py input/gello_purplebottle3_rawtsdf --object-id 1
"""

import argparse
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene_path", help="Path to scene directory, e.g. input/gello_purplebottle3_rawtsdf")
    parser.add_argument("--object-id", type=int, required=True,
                        help="Numeric object id from labels.txt, e.g. 1")
    args = parser.parse_args()

    ply_path = os.path.join(args.scene_path, "output", "objects_ply", f"{args.object_id}.ply")
    if not os.path.isfile(ply_path):
        print(f"ERROR: {ply_path} not found - run create_simulation.py first")
        sys.exit(1)

    from plyfile import PlyData
    data = PlyData.read(ply_path).elements[0]
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    center = xyz.mean(axis=0)

    print(f"Object id={args.object_id}  |  {len(xyz)} Gaussians")
    print(f"Center: x={center[0]:.4f}  y={center[1]:.4f}  z={center[2]:.4f}")
    print(f"\nPaste into configs/simulation/real_world_simulation.yaml:")
    print(f"  rotation_center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")


if __name__ == "__main__":
    main()
