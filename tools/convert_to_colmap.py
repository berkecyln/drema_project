import json
import argparse
import cv2
import numpy as np
from pathlib import Path


def convert(data_dir: Path):
    image_dir = data_dir / "images"
    pose_dir = data_dir / "poses"

    pose_files = sorted(pose_dir.glob("*.txt"))

    with open(pose_files[0], "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    K = np.array([list(map(float, lines[i].split())) for i in range(4, 7)])
    fl_x, fl_y = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    first_img = cv2.imread(str(image_dir / f"{pose_files[0].stem}.png"))
    h, w = first_img.shape[:2]

    frames = []
    for pose_path in pose_files:
        with open(pose_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        c2w = np.array([list(map(float, lines[i].split())) for i in range(4)])
        c2w[0:3, 1:3] *= -1
        frames.append({
            "file_path": f"images/{pose_path.stem}.png",
            "transform_matrix": c2w.tolist(),
        })

    out = {"fl_x": fl_x, "fl_y": fl_y, "cx": cx, "cy": cy, "w": w, "h": h, "camera_model": "OPENCV", "frames": frames}

    with open(data_dir / "transforms.json", "w") as f:
        json.dump(out, f, indent=4)

    print(f"Done. Created transforms.json with {len(frames)} frames in {data_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, help="Path to the input data directory")
    args = parser.parse_args()
    convert(args.data_dir)
