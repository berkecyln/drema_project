"""
Interactive bounding-box picker for SAM3 video prompts.
Reads config from configs/segmentation.yaml, saves visual_prompts.yaml per folder.

Usage:
    bash run_renderer.sh python tools/pick_visual_prompts.py [config_path]
"""

import copy
import os
# Force X11 — Wayland breaks matplotlib/Qt window creation
os.environ["GDK_BACKEND"] = "x11"
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["GLFW_PLATFORM"] = "x11"
os.environ.pop("WAYLAND_DISPLAY", None)
os.environ.pop("XDG_SESSION_TYPE", None)

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import RectangleSelector
from PIL import Image
from omegaconf import OmegaConf
import yaml

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
TABLE_ID = 50
DEFAULT_CONFIG = "configs/segmentation.yaml"


def load_config(config_path: str) -> OmegaConf:
    path = Path(config_path)
    if not path.exists():
        print(f"Error: config not found at '{path}'")
        sys.exit(1)
    return OmegaConf.load(path)


def resolve_folders(cfg) -> list:
    """Resolve folder list from config, same logic as run_segmentation.py."""
    input_dir = Path(cfg.input_dir)
    folders = list(cfg.folders) if cfg.folders else []

    if not folders:
        return sorted([d for d in input_dir.iterdir() if d.is_dir()])

    resolved = []
    for name in folders:
        candidate = Path(name)
        if candidate.is_dir():
            resolved.append(candidate)
        elif (input_dir / name).is_dir():
            resolved.append(input_dir / name)
        else:
            print(f"  Warning: folder '{name}' not found, skipping")
    return resolved


def load_prompt_image(data_folder: Path):
    images_dir = data_folder / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"No images/ folder in {data_folder}")

    files = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    try:
        files.sort(key=lambda p: int(p.stem))
    except ValueError:
        files.sort()

    if not files:
        raise RuntimeError(f"No images found in {images_dir}")

    mid_idx = len(files) // 2
    image = np.array(Image.open(files[mid_idx]).convert("RGB"))
    return image, files[mid_idx], mid_idx


def assign_ids(labels: list) -> list:
    specs = []
    obj_id = 1
    for label in labels:
        if label.lower() == "table":
            specs.append({"id": TABLE_ID, "label": label})
        else:
            specs.append({"id": obj_id, "label": label})
            obj_id += 1
    return specs


class BoxPicker:
    COLORS = ["#00FF00", "#FF4444", "#4499FF", "#FF9900", "#FF44FF", "#44FFFF"]

    def __init__(self, image_np: np.ndarray, label_specs: list, folder_name: str):
        self.image_np = image_np
        self.label_specs = label_specs
        self.current_idx = 0
        self.current_box = None  # [x, y, w, h] in pixels

        self.fig, self.ax = plt.subplots(figsize=(14, 9))
        try:
            self.fig.canvas.manager.set_window_title(f"SAM3 Prompt Picker — {folder_name}")
        except Exception:
            pass
        self.ax.imshow(image_np)
        self.ax.set_axis_off()

        self.rs = RectangleSelector(
            self.ax,
            self._on_select,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._update_title()
        self._draw_legend()

    def _color(self, idx: int) -> str:
        return self.COLORS[idx % len(self.COLORS)]

    def _update_title(self):
        if self.current_idx >= len(self.label_specs):
            self.ax.set_title(
                "All labels done.  Close window to save.",
                fontsize=13, color="lime", pad=10,
            )
        else:
            spec = self.label_specs[self.current_idx]
            color = self._color(self.current_idx)
            self.ax.set_title(
                f"[{self.current_idx + 1}/{len(self.label_specs)}]  "
                f"Draw box for: '{spec['label']}'  (id={spec['id']})\n"
                "Enter = confirm    s = skip    close window = save & quit",
                fontsize=12, color=color, pad=10,
            )
        self.fig.canvas.draw_idle()

    def _draw_legend(self):
        handles = [
            mpatches.Patch(color=self._color(i), label=f"id={s['id']}: {s['label']}")
            for i, s in enumerate(self.label_specs)
        ]
        self.ax.legend(
            handles=handles, loc="upper right", fontsize=9,
            framealpha=0.75, facecolor="black", labelcolor="white",
        )

    def _on_select(self, eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return
        x1, y1 = int(round(eclick.xdata)), int(round(eclick.ydata))
        x2, y2 = int(round(erelease.xdata)), int(round(erelease.ydata))
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)
        self.current_box = [x, y, w, h]

    def _confirm_current(self):
        if self.current_box is None:
            print(f"  [!] No box drawn for '{self.label_specs[self.current_idx]['label']}' — draw one first.")
            return
        spec = self.label_specs[self.current_idx]
        spec["box_xywh"] = self.current_box
        color = self._color(self.current_idx)
        x, y, w, h = self.current_box
        rect = mpatches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
        self.ax.add_patch(rect)
        self.ax.text(
            x + 4, y + 16, f"{spec['id']}: {spec['label']}",
            color=color, fontsize=9,
            bbox=dict(facecolor="black", alpha=0.5, pad=2, edgecolor="none"),
        )
        print(f"  Confirmed: '{spec['label']}'  id={spec['id']}  box={self.current_box}")
        self.current_box = None
        self.current_idx += 1
        self._update_title()
        if self.current_idx >= len(self.label_specs):
            self.rs.set_active(False)

    def _on_key(self, event):
        if self.current_idx >= len(self.label_specs):
            return
        if event.key == "enter":
            self._confirm_current()
        elif event.key == "s":
            spec = self.label_specs[self.current_idx]
            print(f"  Skipped: '{spec['label']}'")
            self.current_idx += 1
            self.current_box = None
            self._update_title()
            if self.current_idx >= len(self.label_specs):
                self.rs.set_active(False)

    def run(self) -> list:
        plt.tight_layout()
        plt.show()
        return [s for s in self.label_specs if "box_xywh" in s]


def save_prompts(data_folder: Path, results: list, prompt_frame: int):
    # Add sam_text to any object entry to use a different text for SAM tracking than the DreMa label.
    out_path = data_folder / "visual_prompts.yaml"
    with open(out_path, "w") as f:
        yaml.dump({"prompt_frame": prompt_frame, "objects": results}, f, default_flow_style=None, sort_keys=False)
    print(f"  Saved {len(results)} prompt(s) (frame {prompt_frame}) → {out_path}")
    for r in results:
        print(f"    id={r['id']}  label='{r['label']}'  box={r['box_xywh']}")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    cfg = load_config(config_path)

    labels = list(cfg.sam3_video.labels)
    if not labels:
        print("Error: sam3_video.labels is empty in config")
        sys.exit(1)

    label_specs_template = assign_ids(labels)

    folders = resolve_folders(cfg)
    if not folders:
        print(f"No folders found under '{cfg.input_dir}'")
        sys.exit(1)

    print(f"Config: {config_path}")
    print(f"Labels: {labels}")
    print(f"Folders to annotate: {[str(f) for f in folders]}\n")

    for folder in folders:
        print(f"--- {folder.name} ---")
        try:
            image_np, prompt_file, prompt_frame = load_prompt_image(folder)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  Skipping: {e}")
            continue

        print(f"  Prompt frame: {prompt_file.name} (index {prompt_frame})  ({image_np.shape[1]}x{image_np.shape[0]})")
        print("  Draw box → Enter to confirm, 's' to skip, close window to save\n")

        # Fresh copy per folder so box_xywh doesn't carry over between folders
        label_specs = copy.deepcopy(label_specs_template)

        picker = BoxPicker(image_np, label_specs, folder.name)
        results = picker.run()

        if not results:
            print(f"  No boxes confirmed for {folder.name}, skipping save.\n")
            continue

        save_prompts(folder, results, prompt_frame)
        print()


if __name__ == "__main__":
    main()
