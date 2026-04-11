"""SAM3 video tracking segmenter for DreMa dataset preparation."""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


class SAM3VideoSegmenter:
    """
    SAM3 video tracking segmenter.
    Prompt with bounding boxes on a chosen frame, propagate masks across all frames.

    Mask ID scheme: Background=255, Table=50, Objects=1,2,...
    """

    def __init__(self, device: str = None):
        from sam3.model.sam3_video_predictor import Sam3VideoPredictor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Locate BPE vocab (same approach as SAM3Segmenter)
        sam3_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bpe_path = os.path.join(sam3_dir, "sam3", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
        if not os.path.exists(bpe_path):
            raise FileNotFoundError(f"SAM3 BPE vocabulary not found at {bpe_path}")

        print("Loading SAM3 video model...")
        self.predictor = Sam3VideoPredictor(bpe_path=bpe_path)
        print(f"SAM3 video model loaded on {self.device}")

    def _sorted_image_files(self, images_dir: Path) -> List[Path]:
        """Sort by integer stem, matching SAM3's internal video loader order."""
        files = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        try:
            files.sort(key=lambda p: int(p.stem))
        except ValueError:
            files.sort()
        return files

    def process_folder(
        self,
        images_dir: Path,
        visual_prompts: List[Dict],
        background_id: int = 255,
        propagation_direction: str = "both",
        prompt_frame: int = 0,
    ) -> Tuple[Dict[str, np.ndarray], Dict[int, str]]:
        """
        Args:
            visual_prompts: list of dicts with id, label, box_xywh (pixels at prompt_frame),
                            and optional sam_text (SAM tracking text; falls back to label)
            propagation_direction: "forward", "backward", or "both"
            prompt_frame: frame index where boxes were drawn

        Returns:
            frame_masks: {filename: mask [H,W] uint8}, labels: {id: label_name}
        """
        image_files = self._sorted_image_files(images_dir)
        if not image_files:
            raise RuntimeError(f"No images found in {images_dir}")

        first = Image.open(image_files[0])
        h, w = first.height, first.width

        labels: Dict[int, str] = {background_id: "background"}
        for p in visual_prompts:
            labels[p["id"]] = p["label"]

        # Load all frames once; reuse session per object via reset_session.
        resp = self.predictor.handle_request({
            "type": "start_session",
            "resource_path": str(images_dir),
        })
        session_id = resp["session_id"]

        # Pre-fill all frames with background
        frame_masks: Dict[str, np.ndarray] = {
            f.name: np.full((h, w), background_id, dtype=np.uint8) for f in image_files
        }

        # Descending id order: table (50) painted first, objects (1,2,...) on top to win overlaps.
        for prompt in sorted(visual_prompts, key=lambda p: -p["id"]):
            bx, by, bw, bh = prompt["box_xywh"]
            box_norm = [bx / w, by / h, bw / w, bh / h]  # normalize to [0,1]
            sam_text = prompt.get("sam_text", prompt["label"])

            self.predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": prompt_frame,
                "text": sam_text,
                "bounding_boxes": [box_norm],
                "bounding_box_labels": [1],
            })

            for result in self.predictor.handle_stream_request({
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": propagation_direction,
                "start_frame_index": prompt_frame,
            }):
                frame_idx = result["frame_index"]
                outputs = result["outputs"]
                if outputs is None or frame_idx >= len(image_files):
                    continue

                masks = outputs["out_binary_masks"]  # [N, H, W] bool
                if len(masks) == 0:
                    continue

                obj_mask = masks.any(axis=0)  # union of all detected masks
                if obj_mask.any():
                    frame_masks[image_files[frame_idx].name][obj_mask] = prompt["id"]

            # Reset tracking state (frames stay loaded) for next object
            self.predictor.handle_request({
                "type": "reset_session",
                "session_id": session_id,
            })

        self.predictor.handle_request({
            "type": "close_session",
            "session_id": session_id,
        })

        return frame_masks, labels

    def create_debug_visualization(
        self,
        image: Image.Image,
        mask: np.ndarray,
        labels: Dict[int, str],
        background_id: int = 255,
    ) -> np.ndarray:
        image_np = np.array(image)
        overlay = image_np.copy()

        colors = {}
        for label_id in labels:
            if label_id == background_id:
                continue
            np.random.seed(label_id * 42)
            colors[label_id] = tuple(np.random.randint(50, 256, 3).tolist())

        for label_id, color in colors.items():
            region = mask == label_id
            if not region.any():
                continue
            overlay[region] = (0.6 * overlay[region] + 0.4 * np.array(color)).astype(np.uint8)
            mask_u8 = region.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)
            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    text = f"{label_id}: {labels[label_id]}"
                    cv2.putText(overlay, text, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(overlay, text, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        return overlay
