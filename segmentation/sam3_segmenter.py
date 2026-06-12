"""
SAM 3 segmentation pipeline for DreMa dataset preparation.
Uses SAM 3's text-based Promptable Concept Segmentation (PCS).
"""

import numpy as np
import torch
from PIL import Image
from typing import Dict, List, Tuple, Optional
import warnings
import cv2

warnings.filterwarnings("ignore", category=UserWarning)


class SAM3Segmenter:
    """SAM 3 text-prompted segmentation (Promptable Concept Segmentation), per frame."""

    def __init__(
        self,
        model_id: str = "facebook/sam3",
        device: str = None,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor as SAM3Proc
        import os

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.mask_threshold = mask_threshold

        print("Loading SAM 3...")
        # locate BPE vocab manually — pkg_resources fails with editable installs
        sam3_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bpe_path = os.path.join(sam3_dir, "sam3", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
        if not os.path.exists(bpe_path):
            raise FileNotFoundError(f"SAM3 BPE vocabulary not found at {bpe_path}")
        self.model = build_sam3_image_model(bpe_path=bpe_path)
        self.processor = SAM3Proc(self.model)
        print(f"SAM 3 loaded on {self.device}")

    def segment_with_text(
        self,
        image: Image.Image,
        text_prompt: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns masks [N,H,W], boxes [N,4] xyxy, scores [N]."""
        # SAM3 works better when prompts end with a dot
        if text_prompt and not text_prompt.strip().endswith("."):
            text_prompt = text_prompt.strip() + "."

        inference_state = self.processor.set_image(image)
        output = self.processor.set_text_prompt(
            state=inference_state,
            prompt=text_prompt
        )

        masks = output["masks"]    # [N, H, W]
        boxes = output["boxes"]    # [N, 4]
        scores = output["scores"]  # [N]

        if len(masks) == 0:
            h, w = image.size[1], image.size[0]
            return (
                np.zeros((0, h, w), dtype=np.uint8),
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32)
            )
        
        if torch.is_tensor(masks):
            masks = masks.cpu().numpy()
        if torch.is_tensor(boxes):
            boxes = boxes.cpu().numpy()
        if torch.is_tensor(scores):
            scores = scores.cpu().numpy()
        
        masks = masks.astype(np.uint8)

        # normalize to [N, H, W] — SAM3 may return 4D, 3D, or 2D
        if masks.ndim == 4:
            masks = masks[0]
        if masks.ndim == 2:
            masks = masks[np.newaxis, :, :]

        return masks, boxes, scores

    def process_image(
        self,
        image: Image.Image,
        object_prompts: List[str],
        table_prompt: Optional[str] = None,
        background_id: int = 255,
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Returns a combined mask [H,W] and {id: label}.
        Mask IDs: background=255, table=50, objects=1,2,...
        """
        h, w = image.size[1], image.size[0]
        combined_mask = np.full((h, w), background_id, dtype=np.uint8)
        labels = {background_id: "background"}

        TABLE_ID = 50
        OBJECT_START_ID = 1
        current_id = OBJECT_START_ID

        if table_prompt:
            table_masks, _, table_scores = self.segment_with_text(image, table_prompt)
            if len(table_masks) > 0:
                # highest confidence detection wins
                best_idx = np.argmax(table_scores)
                if best_idx >= len(table_masks):
                    best_idx = 0
                combined_mask[table_masks[best_idx] > 0] = TABLE_ID
                labels[TABLE_ID] = table_prompt

        for prompt in object_prompts:
            obj_masks, _, obj_scores = self.segment_with_text(image, prompt)
            if len(obj_masks) == 0:
                continue
            sorted_indices = obj_scores.argsort()[::-1]
            label_counts = {}
            clean_label = prompt.strip()
            for mask_idx, idx in enumerate(sorted_indices):
                if idx >= len(obj_masks):
                    print(f"[Warning] Not enough masks returned for prompt '{prompt}'. Skipping idx {idx}.")
                    continue
                if clean_label in label_counts:
                    label_counts[clean_label] += 1
                    instance_label = f"{clean_label}_{label_counts[clean_label]}"
                else:
                    label_counts[clean_label] = 1
                    # _1 suffix only if more instances of the same prompt follow
                    remaining_same = len(sorted_indices) - list(sorted_indices).index(idx) - 1
                    instance_label = f"{clean_label}_1" if remaining_same > 0 else clean_label
                combined_mask[obj_masks[idx] > 0] = current_id
                labels[current_id] = instance_label
                current_id += 1

        return combined_mask, labels

    def create_debug_visualization(
        self,
        image: Image.Image,
        mask: np.ndarray,
        labels: Dict[int, str],
        background_id: int = 255,
    ) -> np.ndarray:
        """Overlay colored masks, contours and label text on the image."""
        image_np = np.array(image)
        overlay = image_np.copy()

        colors = {}
        for label_id in labels.keys():
            if label_id == background_id:
                continue
            np.random.seed(label_id * 42)
            colors[label_id] = tuple(np.random.randint(50, 256, 3).tolist())

        for label_id, color in colors.items():
            mask_region = (mask == label_id)
            if not mask_region.any():
                continue
            overlay[mask_region] = (0.6 * overlay[mask_region] + 0.4 * np.array(color)).astype(np.uint8)

            mask_uint8 = mask_region.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)

            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    label_text = f"{label_id}: {labels[label_id]}"
                    cv2.putText(overlay, label_text, (cx-30, cy),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(overlay, label_text, (cx-30, cy),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        return overlay
