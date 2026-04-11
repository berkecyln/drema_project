"""
Segmentation script for DreMa dataset preparation.
Segmentation Methods: GroundingDINO+SAM and SAM 3

Usage: python run_segmentation.py

Note : Change the segmenter method and parameters in configs/segmentation.yaml
    
"""

import os
import sys
from pathlib import Path

# Use system gcc for Triton JIT compilation — conda's cross-compiler fails on Arch Linux
# (incompatible with newer glibc .relr.dyn sections). Must be set before any triton import.
os.environ["CC"] = "gcc"  # Force system gcc — conda's cross-compiler fails on Arch Linux with newer glibc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sam3"))
from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import hydra
from omegaconf import DictConfig, OmegaConf


def get_all_subdirs(input_dir: Path) -> List[Path]:
    """Get all subdirectories in the input directory."""
    return sorted([item for item in input_dir.iterdir() if item.is_dir()])


def pick_input_folders(input_dir: Path, desired: List[str]) -> List[Path]:
    """Resolve desired folders; falls back to all subdirs when list is empty."""
    if not desired:
        return get_all_subdirs(input_dir)

    selected = []
    missing = []
    for name in desired:
        # Try as given (absolute or relative to CWD)
        candidate_direct = Path(name)
        if candidate_direct.is_dir():
            selected.append(candidate_direct)
            continue

        # Fallback: relative to input_dir
        candidate_under_input = input_dir / name
        if candidate_under_input.is_dir():
            selected.append(candidate_under_input)
        else:
            missing.append(name)

    if missing:
        print(f"Warning: missing folders: {missing}")

    return selected


def get_image_files(images_dir: Path) -> List[Path]:
    """Get sorted list of image files from a directory."""
    extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    image_files = [
        f for f in sorted(images_dir.iterdir())
        if f.suffix.lower() in extensions
    ]
    return image_files


def save_mask(mask: np.ndarray, output_path: Path):
    """Save mask as PNG image."""
    Image.fromarray(mask).save(output_path)


def save_labels(labels: Dict[int, str], output_path: Path):
    """
    Save labels in DreMa format: 'label_name;id'
    Only saves non-background labels.
    """
    with open(output_path, 'w') as f:
        for label_id, label_name in sorted(labels.items()):
            if label_name.lower() != "background":
                # Replace spaces with underscores for consistency
                clean_name = label_name.replace(" ", "_")
                f.write(f"{clean_name};{label_id}\n")


def merge_labels(all_labels: List[Dict[int, str]], background_id: int = 255) -> Dict[int, str]:
    """
    Merge labels from multiple frames into a consistent label set.
    Uses the most common label name for each ID.
    """
    from collections import Counter
    
    merged = {background_id: "background"}
    id_label_counts: Dict[int, Counter] = {}
    
    for labels in all_labels:
        for label_id, label_name in labels.items():
            # Skip background
            if label_name.lower() == "background":
                continue
            if label_id not in id_label_counts:
                id_label_counts[label_id] = Counter()
            id_label_counts[label_id][label_name] += 1
    
    for label_id, counter in id_label_counts.items():
        merged[label_id] = counter.most_common(1)[0][0]
    
    return merged


def create_grounded_segmenter(cfg: DictConfig):
    """Initialize GroundingDINO + SAM segmenter."""
    from segmentation.grounded_segmenter import GroundedSegmenter
    
    return GroundedSegmenter(
        dino_model_id=cfg.grounded_sam.dino_model_id,
        sam_model_id=cfg.grounded_sam.sam_model_id,
        device=cfg.device,
        box_threshold=cfg.grounded_sam.box_threshold,
        text_threshold=cfg.grounded_sam.text_threshold,
    )


def create_sam3_segmenter(cfg: DictConfig):
    """Initialize SAM 3 text-based segmenter."""
    from segmentation.sam3_segmenter import SAM3Segmenter

    return SAM3Segmenter(
        model_id=cfg.sam3.model_id,
        device=cfg.device,
        threshold=cfg.sam3.threshold,
        mask_threshold=cfg.sam3.mask_threshold,
    )


def create_sam3_video_segmenter(cfg: DictConfig):
    """Initialize SAM3 video tracking segmenter."""
    from segmentation.sam3_video_segmenter import SAM3VideoSegmenter
    return SAM3VideoSegmenter(device=cfg.device)


def process_image_grounded(
    image: Image.Image,
    segmenter,
    cfg: DictConfig
) -> Tuple[np.ndarray, Dict[int, str]]:
    """Process image with GroundingDINO + SAM."""
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
    """Process image with SAM 3 text-based segmentation."""
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
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Create debug visualization directory if enabled
    debug_dir = None
    if cfg.get("save_debug_vis", False):
        debug_dir = input_dir / cfg.get("debug_subdir", "debug_vis")
        debug_dir.mkdir(exist_ok=True)
    
    all_labels = []
    
    for img_path in tqdm(image_files, desc=f"  {input_dir.name}", leave=False):
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Generate mask using selected method
        mask, labels = process_fn(image, segmenter, cfg)
        
        all_labels.append(labels)
        
        # Save mask with same name as input (but .png extension)
        output_name = img_path.stem + ".png"
        save_mask(mask, output_dir / output_name)
        
        # Save debug visualization if enabled
        if debug_dir is not None:
            debug_vis = segmenter.create_debug_visualization(
                image, mask, labels, background_id=cfg.background_id
            )
            debug_path = debug_dir / output_name
            Image.fromarray(debug_vis).save(debug_path)
    
    # Generate consistent labels.txt from all frames
    merged_labels = merge_labels(all_labels, background_id=255)
    save_labels(merged_labels, input_dir / "labels.txt")
    
    print(f"  Processed {len(image_files)} images, found labels: {merged_labels}")


@hydra.main(version_base=None, config_path="configs", config_name="segmentation")
def main(cfg: DictConfig) -> None:
    """Main segmentation pipeline with Hydra configuration."""
    
    print("=" * 70)
    print("DreMa Segmentation Pipeline")
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 70)
    
    input_dir = Path(cfg.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    
    # Find folders
    input_folders = pick_input_folders(input_dir, list(cfg.folders))
    if not input_folders:
        print(f"No folders found in {input_dir} (folders filter: {list(cfg.folders)})")
        return
    
    print(f"\nFound {len(input_folders)} folders to process:")
    for tf in input_folders:
        print(f"  - {tf.name}")
    
    # Initialize segmenter based on method
    method = cfg.method.lower()
    print(f"\nInitializing {method.upper()} segmentation...")
    
    if method == "grounded_sam":
        segmenter = create_grounded_segmenter(cfg)
        process_fn = process_image_grounded
        print(f"Using GroundedSAM model: {cfg.grounded_sam.dino_model_id}")
        print(f"Using prompts: {cfg.grounded_sam.object_prompts}")
        
    elif method == "sam3":
        segmenter = create_sam3_segmenter(cfg)
        process_fn = process_image_sam3
        print(f"Using SAM 3 model: {cfg.sam3.model_id}")
        print(f"using prompts: {cfg.sam3.object_prompts}")

    elif method == "sam3_video":
        segmenter = create_sam3_video_segmenter(cfg)
        print(f"Using SAM3 video tracking (visual prompts from visual_prompts.yaml per folder)")

    else:
        print(f"Error: Unknown method '{method}'. Use 'grounded_sam', 'sam3', or 'sam3_video'")
        return

    # Process each folder
    print("\n" + "=" * 70)
    print("Processing folders...")
    print("=" * 70)

    if method == "sam3_video":
        for input_dir in input_folders:
            process_folder_sam3_video(input_dir, segmenter, cfg)
    else:
        for input_dir in input_folders:
            process_input_folder(input_dir, segmenter, cfg, process_fn)
    
    print("=" * 70)
    print("Segmentation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()