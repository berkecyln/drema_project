"""
GroundingDINO + SAM segmentation pipeline for DreMa dataset preparation.
Generates object masks and labels for the DreMa training pipeline.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress verbose model loading messages
warnings.filterwarnings("ignore", category=UserWarning)


class GroundedSegmenter:
    """
    Combines GroundingDINO (open-vocabulary detection) with SAM (segmentation)
    to generate per-object masks from text prompts.
    """
    
    def __init__(
        self,
        dino_model_id: str = "IDEA-Research/grounding-dino-base",
        sam_model_id: str = "facebook/sam-vit-huge",
        device: str = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ):
        """
        Args:
            dino_model_id: HuggingFace model ID for GroundingDINO
            sam_model_id: HuggingFace model ID for SAM
            device: Compute device (auto-detected if None)
            box_threshold: Confidence threshold for bounding box detection
            text_threshold: Confidence threshold for text-box association
        """
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from transformers import SamModel, SamProcessor
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        print(f"Loading GroundingDINO from {dino_model_id}...")
        self.dino_processor = AutoProcessor.from_pretrained(dino_model_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            dino_model_id
        ).to(self.device)
        
        print(f"Loading SAM from {sam_model_id}...")
        self.sam_processor = SamProcessor.from_pretrained(sam_model_id)
        self.sam_model = SamModel.from_pretrained(sam_model_id).to(self.device)
        
        self.dino_model.eval()
        self.sam_model.eval()
        print(f"Models loaded on {self.device}")
    
    def detect_objects(
        self, 
        image: Image.Image, 
        text_prompt: str
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Args:
            image: PIL Image
            text_prompt: Text description of objects to detect
        
        Returns:
            boxes: Bounding boxes [N, 4] in xyxy format
            scores: Confidence scores [N]
            labels: Label strings [N]
        """
        # GroundingDINO works better when labels end with a dot
        if text_prompt and not text_prompt.strip().endswith("."):
            text_prompt = text_prompt.strip() + "."

        inputs = self.dino_processor(
            images=image, 
            text=text_prompt, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
        
            results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]]  # (height, width)
        )[0]
        
            labels = results.get("text_labels", results.get("labels", []))
            return results["boxes"], results["scores"], labels
    
    def segment_from_boxes(
        self, 
        image: Image.Image, 
        boxes: torch.Tensor
    ) -> np.ndarray:
        """
        Generate segmentation masks from bounding boxes using SAM.
        
        Args:
            image: PIL Image
            boxes: Bounding boxes [N, 4] in xyxy format
        
        Returns:
            Binary masks [N, H, W]
        """
        if len(boxes) == 0:
            return np.zeros((0, image.size[1], image.size[0]), dtype=np.uint8)
        
        # SAM expects boxes as list of lists
        boxes_list = [boxes.cpu().tolist()]
        
        inputs = self.sam_processor(
            image, 
            input_boxes=boxes_list, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.sam_model(**inputs)
        
        # Shape: [num_boxes, 3, H, W] with 3 mask candidates per box
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0]

        # SAM returns 3 mask candidates per box; average and threshold them
        masks = masks.float()
        masks = masks.permute(0, 2, 3, 1)  # [num_boxes, H, W, 3]
        masks = masks.mean(axis=-1)  # [num_boxes, H, W]
        masks = (masks > 0).int()
        masks = masks.numpy().astype(np.uint8)

        if len(masks) == 0:
            return np.zeros((0, image.size[1], image.size[0]), dtype=np.uint8)

        return masks
    
    def process_image(
        self,
        image: Image.Image,
        object_prompts: List[str],
        table_prompt: str = "table",
        background_id: int = 255,
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Mask ID assignment:
          Background: 255
          Table: 50
          Objects: 1, 2, 3, ...
        
        Args:
            image: PIL Image (RGB)
            object_prompts: Text prompts for objects
            table_prompt: Text prompt for table
            background_id: Background pixel value
        
        Returns:
            mask: [H, W] with unique IDs per instance
            labels: ID -> label name mapping
        """
        h, w = image.size[1], image.size[0]
        combined_mask = np.full((h, w), background_id, dtype=np.uint8)
        labels = {background_id: "background"}
        
        TABLE_ID = 50
        OBJECT_START_ID = 1
        # Detect and segment table
        if table_prompt:
            table_boxes, table_scores, _ = self.detect_objects(image, table_prompt)
            if len(table_boxes) > 0:
                # Take highest confidence table detection
                best_idx = table_scores.argmax()
                table_mask = self.segment_from_boxes(image, table_boxes[best_idx:best_idx+1])
                if len(table_mask) > 0 and table_mask[0].sum() > 0:
                    combined_mask[table_mask[0] > 0] = TABLE_ID
                    labels[TABLE_ID] = "table"
        
        # Detect and segment objects
        current_id = OBJECT_START_ID
        for prompt in object_prompts:
            boxes, scores, detected_labels = self.detect_objects(image, prompt)
            
            if len(boxes) == 0:
                continue
            
            masks = self.segment_from_boxes(image, boxes)
            
            label_counts = {}
            for i, (mask, label) in enumerate(zip(masks, detected_labels)):
                clean_label = label.strip().rstrip('.') # clean label
                
                if clean_label in label_counts:
                    label_counts[clean_label] += 1
                    instance_label = f"{clean_label}_{label_counts[clean_label]}"
                else:
                    label_counts[clean_label] = 1
                    # Add _1 suffix only if more instances follow
                    remaining_same = sum(1 for l in detected_labels[i+1:] if l.strip().rstrip('.') == clean_label)
                    instance_label = f"{clean_label}_1" if remaining_same > 0 else clean_label
                
                combined_mask[mask > 0] = current_id
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
        """
        Create an RGB debug visualization with colored masks and labels.
        
        Args:
            image: Original PIL Image
            mask: Combined mask [H, W] with unique IDs
            labels: Dict mapping ID to label name
            background_id: ID used for background
        
        Returns:
            RGB image with colored overlays
        """
        image_np = np.array(image)
        overlay = image_np.copy()
        
        # Generate random bright colors for each object
        colors = {}
        for label_id in labels.keys():
            if label_id == background_id:
                continue
            np.random.seed(label_id * 42)
            colors[label_id] = tuple(np.random.randint(50, 256, 3).tolist())
        
        # Draw masks
        for label_id, color in colors.items():
            mask_region = (mask == label_id)
            if not mask_region.any():
                continue
            # Alpha blend the mask
            overlay[mask_region] = (0.6 * overlay[mask_region] + 0.4 * np.array(color)).astype(np.uint8)
            
            # Draw contours
            mask_uint8 = mask_region.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)
            
            # Add label text
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

