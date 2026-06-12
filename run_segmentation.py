"""
Generates object masks for DreMa input folders.
Method and parameters are set in configs/segmentation.yaml.

Usage: python run_segmentation.py
"""

import os
import sys
from pathlib import Path

# system gcc for Triton JIT — conda's cross-compiler fails on Arch with newer glibc
os.environ["CC"] = "gcc"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sam3"))
from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
import hydra
from omegaconf import DictConfig, OmegaConf


def get_all_subdirs(input_dir: Path) -> List[Path]:
    return sorted([item for item in input_dir.iterdir() if item.is_dir()])


def pick_input_folders(input_dir: Path, desired: List[str]) -> List[Path]:
    """Resolve desired folders; falls back to all subdirs when list is empty."""
    if not desired:
        return get_all_subdirs(input_dir)

    selected = []
    missing = []
    for name in desired:
        # try as given first, then relative to input_dir
        candidate_direct = Path(name)
        if candidate_direct.is_dir():
            selected.append(candidate_direct)
            continue

        candidate_under_input = input_dir / name
        if candidate_under_input.is_dir():
            selected.append(candidate_under_input)
        else:
            missing.append(name)

    if missing:
        print(f"Warning: missing folders: {missing}")

    return selected


def get_image_files(images_dir: Path) -> List[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    return [f for f in sorted(images_dir.iterdir()) if f.suffix.lower() in extensions]


def save_mask(mask: np.ndarray, output_path: Path):
    Image.fromarray(mask).save(output_path)


def save_labels(labels: Dict[int, str], output_path: Path):
    """Write labels.txt in DreMa format: 'label_name;id', background excluded."""
    with open(output_path, 'w') as f:
        for label_id, label_name in sorted(labels.items()):
            if label_name.lower() != "background":
                clean_name = label_name.replace(" ", "_")
                f.write(f"{clean_name};{label_id}\n")


def merge_labels(all_labels: List[Dict[int, str]], background_id: int = 255) -> Dict[int, str]:
    """Merge per-frame labels into one set, taking the most common name per id."""
    from collections import Counter

    merged = {background_id: "background"}
    id_label_counts: Dict[int, Counter] = {}

    for labels in all_labels:
        for label_id, label_name in labels.items():
            if label_name.lower() == "background":
                continue
            if label_id not in id_label_counts:
                id_label_counts[label_id] = Counter()
            id_label_counts[label_id][label_name] += 1

    for label_id, counter in id_label_counts.items():
        merged[label_id] = counter.most_common(1)[0][0]

    return merged


def create_grounded_segmenter(cfg: DictConfig):
    from segmentation.grounded_segmenter import GroundedSegmenter

    return GroundedSegmenter(
        dino_model_id=cfg.grounded_sam.dino_model_id,
        sam_model_id=cfg.grounded_sam.sam_model_id,
        device=cfg.device,
        box_threshold=cfg.grounded_sam.box_threshold,
        text_threshold=cfg.grounded_sam.text_threshold,
    )


def create_sam3_segmenter(cfg: DictConfig):
    from segmentation.sam3_segmenter import SAM3Segmenter

    return SAM3Segmenter(
        model_id=cfg.sam3.model_id,
        device=cfg.device,
        threshold=cfg.sam3.threshold,
        mask_threshold=cfg.sam3.mask_threshold,
    )


def create_sam3_video_segmenter(cfg: DictConfig):
    from segmentation.sam3_video_segmenter import SAM3VideoSegmenter
    return SAM3VideoSegmenter(device=cfg.device)


def process_image_grounded(
    image: Image.Image,
    segmenter,
    cfg: DictConfig
) -> Tuple[np.ndarray, Dict[int, str]]:
    return segmenter.process_image(
        image,
        object_prompts=cfg.grounded_sam.object_prompts,
        table_prompt=cfg.grounded_sam.table_prompt,
    )


def process_image_sam3(
    image: Image.Image,
    segmenter,
    cfg: DictConfig
) -> Tuple[np.ndarray, Dict[int, str]]:
    return segmenter.process_image(
        image,
        object_prompts=cfg.sam3.object_prompts,
        table_prompt=cfg.sam3.table_prompt,
        background_id=cfg.background_id,
    )


def load_visual_prompts(input_dir: Path, cfg: DictConfig) -> List[dict]:
    """Load visual_prompts.yaml for a given input folder."""
    import yaml

    prompts_path = OmegaConf.select(cfg, "sam3_video.visual_prompts_file")
    if prompts_path:
        prompts_path = Path(prompts_path)
    else:
        prompts_path = input_dir / "visual_prompts.yaml"

    if not prompts_path.exists():
        raise FileNotFoundError(
            f"No visual_prompts.yaml found at {prompts_path}\n"
            f"Run: bash run_renderer.sh python tools/pick_visual_prompts.py {input_dir} --labels table \"<object>\""
        )

    with open(prompts_path) as f:
        data = yaml.safe_load(f)
    prompt_frame = data.get("prompt_frame", 0)
    return data["objects"], prompt_frame


def process_folder_sam3_video(input_dir: Path, segmenter, cfg: DictConfig):
    """Process an entire folder with SAM3 video tracking (one session per folder)."""
    images_dir = input_dir / "images"
    output_dir = input_dir / cfg.output_subdir

    if not images_dir.exists():
        print(f"  Skipping {input_dir.name}: no images/ folder")
        return

    visual_prompts, prompt_frame = load_visual_prompts(input_dir, cfg)
    print(f"  Loaded {len(visual_prompts)} visual prompt(s) from visual_prompts.yaml (prompt frame: {prompt_frame})")
    for p in visual_prompts:
        print(f"    id={p['id']}  label='{p['label']}'  box={p['box_xywh']}")

    output_dir.mkdir(exist_ok=True)

    debug_dir = None
    if cfg.get("save_debug_vis", False):
        debug_dir = input_dir / cfg.get("debug_subdir", "debug_vis")
        debug_dir.mkdir(exist_ok=True)

    frame_masks, labels = segmenter.process_folder(
        images_dir=images_dir,
        visual_prompts=visual_prompts,
        background_id=cfg.background_id,
        propagation_direction=OmegaConf.select(cfg, "sam3_video.propagation_direction", default="both"),
        prompt_frame=prompt_frame,
    )

    for filename, mask in tqdm(frame_masks.items(), desc=f"  {input_dir.name} (saving)", leave=False):
        out_name = Path(filename).stem + ".png"
        save_mask(mask, output_dir / out_name)

        if debug_dir is not None:
            image = Image.open(images_dir / filename).convert("RGB")
            debug_vis = segmenter.create_debug_visualization(
                image, mask, labels, background_id=cfg.background_id
            )
            Image.fromarray(debug_vis).save(debug_dir / out_name)

    save_labels(labels, input_dir / "labels.txt")
    print(f"  Processed {len(frame_masks)} frames, labels: {labels}")


def process_input_folder(
    input_dir: Path,
    segmenter,
    cfg: DictConfig,
    process_fn,
):
    """Process all images in a single folder."""
    images_dir = input_dir / "images"
    output_dir = input_dir / cfg.output_subdir
    
    if not images_dir.exists():
        print(f"  Skipping {input_dir.name}: no images/ folder")
        return
    
    image_files = get_image_files(images_dir)
    if not image_files:
        print(f"  Skipping {input_dir.name}: no images found")
        return

    output_dir.mkdir(exist_ok=True)

    debug_dir = None
    if cfg.get("save_debug_vis", False):
        debug_dir = input_dir / cfg.get("debug_subdir", "debug_vis")
        debug_dir.mkdir(exist_ok=True)

    all_labels = []

    for img_path in tqdm(image_files, desc=f"  {input_dir.name}", leave=False):
        image = Image.open(img_path).convert("RGB")
        mask, labels = process_fn(image, segmenter, cfg)
        all_labels.append(labels)

        output_name = img_path.stem + ".png"
        save_mask(mask, output_dir / output_name)

        if debug_dir is not None:
            debug_vis = segmenter.create_debug_visualization(
                image, mask, labels, background_id=cfg.background_id
            )
            Image.fromarray(debug_vis).save(debug_dir / output_name)

    # one consistent labels.txt across all frames
    merged_labels = merge_labels(all_labels, background_id=255)
    save_labels(merged_labels, input_dir / "labels.txt")

    print(f"  Processed {len(image_files)} images, found labels: {merged_labels}")


@hydra.main(version_base=None, config_path="configs", config_name="segmentation")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    input_dir = Path(cfg.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return

    input_folders = pick_input_folders(input_dir, list(cfg.folders))
    if not input_folders:
        print(f"No folders found in {input_dir} (folders filter: {list(cfg.folders)})")
        return

    print(f"Found {len(input_folders)} folders to process:")
    for tf in input_folders:
        print(f"  - {tf.name}")

    method = cfg.method.lower()
    print(f"\nInitializing {method} segmentation...")

    if method == "grounded_sam":
        segmenter = create_grounded_segmenter(cfg)
        process_fn = process_image_grounded
        print(f"Model: {cfg.grounded_sam.dino_model_id}, prompts: {cfg.grounded_sam.object_prompts}")

    elif method == "sam3":
        segmenter = create_sam3_segmenter(cfg)
        process_fn = process_image_sam3
        print(f"Model: {cfg.sam3.model_id}, prompts: {cfg.sam3.object_prompts}")

    elif method == "sam3_video":
        segmenter = create_sam3_video_segmenter(cfg)
        print("Visual prompts are read from visual_prompts.yaml in each folder")

    else:
        print(f"Error: Unknown method '{method}'. Use 'grounded_sam', 'sam3', or 'sam3_video'")
        return

    if method == "sam3_video":
        for input_dir in input_folders:
            process_folder_sam3_video(input_dir, segmenter, cfg)
    else:
        for input_dir in input_folders:
            process_input_folder(input_dir, segmenter, cfg, process_fn)

    print("Done.")


if __name__ == "__main__":
    main()