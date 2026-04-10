"""
Visualize camera extrinsics from a transforms.json file in 3D space.
Each camera is shown as a small coordinate frame (RGB = XYZ axes)
with the camera position marked as a dot.
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

DEFAULT_PATH = "input/test1/transforms.json"
AXIS_LENGTH = 0.02  # length of the drawn camera axes


def load_transforms(path: str) -> list[np.ndarray]:
    with open(path) as f:
        data = json.load(f)
    matrices = []
    for frame in data["frames"]:
        matrices.append(np.array(frame["transform_matrix"], dtype=np.float64))
    return matrices


def draw_camera_frame(ax, c2w: np.ndarray, length: float = AXIS_LENGTH):
    """Draw X (red), Y (green), Z (blue) axes for a camera given its c2w matrix."""
    origin = c2w[:3, 3]
    x_axis = c2w[:3, 0]
    y_axis = c2w[:3, 1]
    z_axis = c2w[:3, 2]

    for axis, color in zip([x_axis, y_axis, z_axis], ["r", "g", "b"]):
        end = origin + length * axis
        ax.plot(
            [origin[0], end[0]],
            [origin[1], end[1]],
            [origin[2], end[2]],
            color=color,
            linewidth=0.8,
            alpha=0.8,
        )


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    transforms = load_transforms(path)
    print(f"Loaded {len(transforms)} camera poses from '{path}'")

    positions = np.array([m[:3, 3] for m in transforms])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw each camera frame
    for c2w in transforms:
        draw_camera_frame(ax, c2w)

    # Draw camera positions as dots, coloured by index (trajectory order)
    sc = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=np.arange(len(positions)),
        cmap="plasma",
        s=6,
        zorder=5,
    )
    plt.colorbar(sc, ax=ax, label="Frame index", pad=0.1, shrink=0.6)

    # Draw trajectory line
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        color="gray",
        linewidth=0.5,
        alpha=0.4,
        label="trajectory",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Camera extrinsics  ({len(transforms)} frames)\n"
                 "axes: X=red  Y=green  Z=blue")
    ax.legend(loc="upper left", fontsize=8)

    # Equal aspect ratio
    max_range = (positions.max(axis=0) - positions.min(axis=0)).max() / 2
    mid = positions.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
