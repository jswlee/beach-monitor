"""
Save training data in YOLO format during inference for model fine-tuning.

This module saves:
1. Object detection data (bounding boxes) - train_object_detection/
2. Segmentation data (polygons) - train_segmentation/

Both follow YOLO format:
- images/ folder with .jpg files
- labels/ folder with .txt files (same name as image)
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import shutil
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class TrainingDataSaver:
    """Saves inference results as training data in YOLO format."""
    
    def __init__(self, base_dir: str = "training_data"):
        """
        Initialize the training data saver.
        
        Args:
            base_dir: Base directory for all training data
        """
        self.base_dir = Path(base_dir)
        
        # Setup directories
        self.detection_dir = self.base_dir / "train_object_detection"
        self.segmentation_dir = self.base_dir / "train_segmentation"
        
        # Create subdirectories
        for parent_dir in [self.detection_dir, self.segmentation_dir]:
            (parent_dir / "images").mkdir(parents=True, exist_ok=True)
            (parent_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Training data saver initialized at: {self.base_dir}")
    
    def save_detection_data(
        self, 
        image_path: str, 
        yolo_results,
        image_id: str = None
    ) -> Dict[str, str]:
        """
        Save object detection data in YOLO format.
        
        Args:
            image_path: Path to the original image
            yolo_results: YOLO results object from ultralytics
            image_id: Optional custom ID for the image (default: timestamp from filename)
        
        Returns:
            Dictionary with saved paths
        """
        try:
            # Generate image ID from filename if not provided
            if image_id is None:
                image_id = Path(image_path).stem
            
            # Copy image
            dest_image = self.detection_dir / "images" / f"{image_id}.jpg"
            shutil.copy2(image_path, dest_image)
            
            # Save labels in YOLO format
            dest_label = self.detection_dir / "labels" / f"{image_id}.txt"
            
            result = yolo_results[0]
            
            # Get image dimensions for normalization
            img_height, img_width = result.orig_shape
            
            with open(dest_label, 'w') as f:
                if result.boxes is not None:
                    for i, box in enumerate(result.boxes):
                        # Get class ID
                        cls_id = int(box.cls[0])
                        
                        # Get bounding box in xyxy format
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = xyxy
                        
                        # Convert to YOLO format (normalized center x, center y, width, height)
                        x_center = ((x1 + x2) / 2) / img_width
                        y_center = ((y1 + y2) / 2) / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        
                        # Write in YOLO format: class_id x_center y_center width height
                        f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            logger.info(f"Saved detection training data: {image_id}")
            
            return {
                'image_path': str(dest_image),
                'label_path': str(dest_label),
                'image_id': image_id
            }
            
        except Exception as e:
            logger.error(f"Error saving detection training data: {e}")
            return {}
    
    def save_segmentation_data(
        self,
        image_path: str,
        segmentation_mask: np.ndarray,
        image_id: str = None
    ) -> Dict[str, str]:
        """
        Save segmentation data in YOLO format with polygons extracted from segmentation mask.
        
        Args:
            image_path: Path to the original image
            segmentation_mask: Segmentation mask (H, W) with class IDs (0=background, 1=base, 2=beach, 3=water)
            image_id: Optional custom ID for the image
        
        Returns:
            Dictionary with saved paths
        """
        try:
            # Generate image ID from filename if not provided
            if image_id is None:
                image_id = Path(image_path).stem
            
            # Copy image
            dest_image = self.segmentation_dir / "images" / f"{image_id}.jpg"
            shutil.copy2(image_path, dest_image)
            
            # Save labels in YOLO segmentation format
            dest_label = self.segmentation_dir / "labels" / f"{image_id}.txt"
            
            img_height, img_width = segmentation_mask.shape
            
            with open(dest_label, 'w') as f:
                # Process each class: beach (2) and water (3)
                # Map to YOLO class IDs: 0=beach, 1=water
                class_mapping = {
                    2: 0,  # beach -> 0
                    3: 1   # water -> 1
                }
                
                for seg_class_id, yolo_class_id in class_mapping.items():
                    # Create binary mask for this class
                    class_mask = (segmentation_mask == seg_class_id).astype(np.uint8)
                    
                    # Find contours
                    contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Process each contour (separate regions)
                    for contour in contours:
                        # Skip very small contours (noise)
                        if cv2.contourArea(contour) < 100:
                            continue
                        
                        # Simplify contour to reduce points
                        epsilon = 0.001 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        
                        # Flatten and normalize coordinates
                        points = approx.reshape(-1, 2)
                        
                        # Normalize coordinates
                        normalized_points = []
                        for x, y in points:
                            x_norm = x / img_width
                            y_norm = y / img_height
                            normalized_points.extend([x_norm, y_norm])
                        
                        # Write in YOLO format: class_id x1 y1 x2 y2 x3 y3 ...
                        if len(normalized_points) >= 6:  # At least 3 points for a polygon
                            f.write(f"{yolo_class_id}")
                            for coord in normalized_points:
                                f.write(f" {coord:.6f}")
                            f.write("\n")
            
            logger.info(f"Saved segmentation training data: {image_id}")
            
            return {
                'image_path': str(dest_image),
                'label_path': str(dest_label),
                'image_id': image_id
            }
            
        except Exception as e:
            logger.error(f"Error saving segmentation training data: {e}")
            return {}
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get statistics about saved training data.
        
        Returns:
            Dictionary with counts of images and labels
        """
        stats = {
            'detection': {
                'images': len(list((self.detection_dir / "images").glob("*.jpg"))),
                'labels': len(list((self.detection_dir / "labels").glob("*.txt")))
            },
            'segmentation': {
                'images': len(list((self.segmentation_dir / "images").glob("*.jpg"))),
                'labels': len(list((self.segmentation_dir / "labels").glob("*.txt")))
            }
        }
        return stats


# Global singleton instance
_training_data_saver = None


def get_training_data_saver(base_dir: str = "training_data") -> TrainingDataSaver:
    """Get or create the global training data saver instance."""
    global _training_data_saver
    if _training_data_saver is None:
        _training_data_saver = TrainingDataSaver(base_dir=base_dir)
    return _training_data_saver
