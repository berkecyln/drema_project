import json
import os
import numpy as np
from pathlib import Path

# SET THESE PATHS
DATA_DIR = Path("/home/ceylanb/DreMa/drema_project/input/test1")
IMAGE_DIR = DATA_DIR / "images"
POSE_DIR = DATA_DIR / "poses"

frames = []

# Sort files to ensure 0001.txt matches 0001.png
pose_files = sorted(list(POSE_DIR.glob("*.txt")))

# We take intrinsics from the first file
# Load with skiprows to handle blank lines
with open(pose_files[0], 'r') as f:
    lines = [line.strip() for line in f if line.strip()]  # Remove empty lines
    
# Parse 4x4 transformation matrix (first 4 lines)
c2w_first = np.array([list(map(float, lines[i].split())) for i in range(4)])

# Parse 3x3 intrinsics matrix (next 3 lines after the blank line)
K = np.array([list(map(float, lines[i].split())) for i in range(4, 7)])

fl_x = float(K[0, 0])
fl_y = float(K[1, 1])
cx = float(K[0, 2])
cy = float(K[1, 2])

# Get image dimensions from first image
import cv2
first_img = cv2.imread(str(IMAGE_DIR / f"{pose_files[0].stem}.png"))
h, w = first_img.shape[:2]

for pose_path in pose_files:
    # Load the file, removing blank lines
    with open(pose_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Parse 4x4 transformation matrix (first 4 lines)
    c2w = np.array([list(map(float, lines[i].split())) for i in range(4)])
    
    # Optional: If the scene is upside down in the viewer, 
    # uncomment the line below to flip Y and Z axes
    # c2w[0:3, 1:3] *= -1 

    frame_path = f"images/{pose_path.stem}.png"
    
    frames.append({
        "file_path": frame_path,
        "transform_matrix": c2w.tolist()
    })

out_json = {
    "fl_x": fl_x,
    "fl_y": fl_y,
    "cx": cx,
    "cy": cy,
    "w": w,
    "h": h,
    "camera_model": "OPENCV",
    "frames": frames
}

with open(DATA_DIR / "transforms.json", "w") as f:
    json.dump(out_json, f, indent=4)

print(f"Done! Created transforms.json with {len(frames)} frames.")