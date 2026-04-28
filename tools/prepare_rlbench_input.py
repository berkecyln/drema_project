"""
Creates the RLBench original-path structure required by prepare_data_for_peract.py.

For each scene, copies low_dim_obs.pkl, variation_descriptions.pkl, variation_number.pkl
from the DreMa input scene directory into:
  <rlbench_input>/<scene>/all_variations/episodes/episode0/

Run from the drema_project directory:
    python tools/prepare_rlbench_input.py
"""

import os
import shutil

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_ROOT = "/home/gunnleif/Projects/drema/drema_project/input"

SCENES = [
    "gello_bottle1_rawtsdf",
    "gello_bottle2_rawtsdf",
    "gello_bottle3_rawtsdf",
    "gello_bottle4_rawtsdf",
]

RLBENCH_INPUT_ROOT = "/home/gunnleif/Projects/drema/drema_project/data/rlbench_input"

PKL_FILES = [
    "low_dim_obs.pkl",
    "variation_descriptions.pkl",
    "variation_number.pkl",
]

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for scene in SCENES:
        src_dir = os.path.join(INPUT_ROOT, scene)
        dst_dir = os.path.join(RLBENCH_INPUT_ROOT, scene, "all_variations", "episodes", "episode0")

        os.makedirs(dst_dir, exist_ok=True)

        for fname in PKL_FILES:
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname)

            if not os.path.isfile(src):
                print(f"  MISSING: {src}")
                continue

            shutil.copyfile(src, dst)
            print(f"  {scene}/{fname} → {dst}")

    print("\nDone. Pass to prepare_data_for_peract.py:")
    print(f"  --original_path {RLBENCH_INPUT_ROOT}")
