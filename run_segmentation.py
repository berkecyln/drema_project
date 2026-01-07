"""
Batch segmentation script for DreMa dataset preparation.
Processes all task folders in input/ and generates object_mask/ and labels.txt.

Configs are at configs/segmentation.yaml for prompts and model settings.
"""

from pathlib import Path
from typing import List, Dict

import hydra
import numpy as np
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def get_task_folders(input_dir: Path) -> List[Path]:
    """Find all folders starting with 'task_' in the input directory."""
    task_folders = []
    for item in sorted(input_dir.iterdir()):
        if item.is_dir() and item.name.startswith("task_"):
            task_folders.append(item)
    return task_folders


def pick_task_folders(input_dir: Path, desired: List[str]) -> List[Path]:
    """Resolve desired task folders; falls back to all task_* when list is empty."""
    if not desired:
        return get_task_folders(input_dir)

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
        print(f"Warning: missing task folders: {missing}")

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
            # Skip all background IDs (both 0 and configured background_id)
            if label_name.lower() == "background":
                continue
            if label_id not in id_label_counts:
                id_label_counts[label_id] = Counter()
            id_label_counts[label_id][label_name] += 1
    
    for label_id, counter in id_label_counts.items():
        merged[label_id] = counter.most_common(1)[0][0]
    
    return merged


def process_task_folder(
    task_dir: Path,
    segmenter,
    object_prompts: List[str],
    table_prompt: str,
    background_id: int = 255,
    save_debug_vis: bool = False,
):
    """Process all images in a single task folder."""
    images_dir = task_dir / "images"
    output_dir = task_dir / "object_mask"
    debug_dir = task_dir / "debug_vis"
    
    if not images_dir.exists():
        print(f"  Skipping {task_dir.name}: no images/ folder")
        return
    
    image_files = get_image_files(images_dir)
    if not image_files:
        print(f"  Skipping {task_dir.name}: no images found")
        return
    
    # Create output directories
    output_dir.mkdir(exist_ok=True)
    if save_debug_vis:
        debug_dir.mkdir(exist_ok=True)
    
    all_labels = []
    
    for img_path in tqdm(image_files, desc=f"  {task_dir.name}", leave=False):
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Generate mask
        mask, labels = segmenter.process_image(
            image,
            object_prompts=object_prompts,
            table_prompt=table_prompt,
            background_id=background_id,
        )
        
        all_labels.append(labels)
        
        # Save mask with same name as input (but .png extension)
        output_name = img_path.stem + ".png"
        save_mask(mask, output_dir / output_name)
        
        # Save debug visualization
        if save_debug_vis:
            debug_overlay = segmenter.create_debug_visualization(
                image, mask, labels, background_id
            )
            debug_path = debug_dir / output_name
            Image.fromarray(debug_overlay).save(debug_path)
    
    # Generate consistent labels.txt from all frames
    merged_labels = merge_labels(all_labels, background_id)
    save_labels(merged_labels, task_dir / "labels.txt")
    
    print(f"  Processed {len(image_files)} images, found labels: {merged_labels}")


@hydra.main(version_base=None, config_path="configs", config_name="segmentation")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    # Controls
    input_dir = Path(cfg.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return

    task_folders = pick_task_folders(input_dir, list(cfg.tasks))
    if not task_folders:
        print(f"No task folders found in {input_dir} (tasks filter: {list(cfg.tasks)})")
        return

    print(f"Processing {len(task_folders)} task folders:")
    for tf in task_folders:
        print(f"  - {tf.name}")

    # Setup segmenter
    print("\nInitializing segmentation models...")
    from segmentation import GroundedSegmenter

    segmenter = GroundedSegmenter(
        dino_model_id=cfg.models.dino_model_id,
        sam_model_id=cfg.models.sam_model_id,
        device=cfg.device,
        box_threshold=cfg.box_threshold,
        text_threshold=cfg.text_threshold,
    )

    print(f"\nProcessing with prompts: objects={cfg.object_prompts}, table='{cfg.table_prompt}'")
    print(f"Background ID: {cfg.background_id}, Debug vis: {cfg.save_debug_vis}")
    print("-" * 60)

    for task_dir in task_folders:
        process_task_folder(
            task_dir,
            segmenter,
            cfg.object_prompts,
            cfg.table_prompt,
            background_id=cfg.background_id,
            save_debug_vis=cfg.save_debug_vis,
        )

    print("-" * 60)
    print("Done!")


if __name__ == "__main__":
    main()