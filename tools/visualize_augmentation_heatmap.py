"""
Heatmap of augmented bottle training positions with approximate bottle shape
(rotated ellipse) showing orientation at each episode.

Usage:
    conda activate drema_env
    python tools/visualize_augmentation_heatmap.py
"""

import glob
import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
from scipy.spatial.transform import Rotation as ScipyR
from scipy.stats import gaussian_kde

DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data', 'generated_data')
OUTPUT_PATH        = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'augmentation_heatmap.png')
OUTPUT_PATH_DOTTED = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'augmentation_heatmap_dotted.png')

SCENES = [
    'gello_bottle1_rawtsdf_episode0_start',
    'gello_bottle2_rawtsdf_episode0_start',
    'gello_bottle3_rawtsdf_episode0_start',
    'gello_bottle4_rawtsdf_episode0_start',
]

BOTTLE_COLORS = ['#9b59b6', '#3498db', '#2ecc71', '#f39c12']
BOTTLE_LABELS = ['Bottle 1', 'Bottle 2', 'Bottle 3', 'Bottle 4']

X_MIN, X_MAX = 0.15, 0.47
Y_MIN, Y_MAX = -0.35, 0.30

BOTTLE_W = 0.025   # bottle width  (~diameter, metres)
BOTTLE_H = 0.055   # bottle height (elongated to show orientation)


def collect_episodes(scene_dir):
    paths = sorted(
        glob.glob(os.path.join(scene_dir, 'episode*/generated_trajectory.pkl')),
        key=lambda p: int(os.path.basename(os.path.dirname(p)).replace('episode', ''))
    )
    original = None
    augmented = []
    for p in paths:
        with open(p, 'rb') as f:
            traj = pickle.load(f)
        for step in traj:
            if not step['gripper_open']:
                ep_id = int(os.path.basename(os.path.dirname(p)).replace('episode', ''))
                pos  = step['gripper_pose'][:2]
                yaw  = ScipyR.from_quat(step['gripper_pose'][3:]).as_euler('xyz')[2]
                pos = np.array([
                    np.clip(pos[0], X_MIN, X_MAX),
                    np.clip(pos[1], Y_MIN, Y_MAX),
                ])
                entry = (pos, yaw)
                if ep_id == 0:
                    original = entry
                else:
                    augmented.append(entry)
                break
    return original, augmented


def make_figure(per_bottle, all_data, show_dots=False):
    """Render and return fig. show_dots=True overlays individual episode positions."""
    all_x = np.array([d[0][0] for d in all_data])
    all_y = np.array([d[0][1] for d in all_data])

    kde = gaussian_kde(np.vstack([all_x, all_y]), bw_method=0.4)
    xi  = np.linspace(X_MIN, X_MAX, 300)
    yi  = np.linspace(Y_MIN, Y_MAX, 300)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    ax.contourf(Yi, Xi, Zi, levels=30, cmap='YlOrRd', alpha=0.85, zorder=1)

    for original, augmented, color, label in per_bottle:
        orig_pos, _ = original
        if show_dots:
            aug_pos = np.array([e[0] for e in augmented])
            ax.scatter(aug_pos[:, 1], aug_pos[:, 0],
                       color=color, s=40, alpha=0.8, zorder=3, edgecolors='none')
        ax.scatter(orig_pos[1], orig_pos[0],
                   color=color, s=180, marker='*', zorder=5,
                   edgecolors='white', linewidths=0.8,
                   label=f'{label}  (★ = original)')

    rect = patches.Rectangle(
        (Y_MIN, X_MIN), Y_MAX - Y_MIN, X_MAX - X_MIN,
        linewidth=1.5, edgecolor='#cccccc', facecolor='none',
        linestyle='--', zorder=6
    )
    ax.add_patch(rect)

    ax.set_xlim(Y_MAX + 0.03, Y_MIN - 0.03)
    ax.set_ylim(X_MIN - 0.02, X_MAX + 0.04)
    ax.set_aspect('equal')
    ax.set_xlabel('Y (m)', fontsize=12)
    ax.set_ylabel('X (m)', fontsize=12)
    ax.grid(True, alpha=0.2, color='gray')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(
        'Augmented Training Object Locations\n'
        'Heatmap = position density  |  ★ = original demo position',
        fontsize=12, pad=12
    )
    plt.tight_layout()
    return fig


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    all_data = []
    per_bottle = []

    for scene, color, label in zip(SCENES, BOTTLE_COLORS, BOTTLE_LABELS):
        scene_dir = os.path.join(DATA_DIR, scene)
        if not os.path.isdir(scene_dir):
            continue
        original, augmented = collect_episodes(scene_dir)
        if not augmented:
            continue
        print(f"{label}: {len(augmented)} augmented episodes")
        per_bottle.append((original, augmented, color, label))
        for entry in augmented:
            all_data.append((*entry, color, label))
        all_data.append((*original, color, label))

    fig = make_figure(per_bottle, all_data, show_dots=False)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"Saved to {OUTPUT_PATH}")
    plt.close(fig)

    fig = make_figure(per_bottle, all_data, show_dots=True)
    fig.savefig(OUTPUT_PATH_DOTTED, dpi=150, bbox_inches='tight')
    print(f"Saved to {OUTPUT_PATH_DOTTED}")
    plt.close(fig)


if __name__ == '__main__':
    main()
