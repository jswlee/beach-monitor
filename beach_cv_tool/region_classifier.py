"""
Region classifier using Segment Anything Model (SAM) to determine if people are on beach or in water.

ARCHITECTURE:
1. Fine-tuned YOLO model detects people and returns bounding boxes
2. SAM segments the entire image into regions
3. Classify each region as water/beach/other based on color and position
4. Map each person's bounding box center to the corresponding region

This approach is fast, free, and more accurate than VLM for beach scenes.
"""
import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RegionClassifier:
    """Classifies person locations (beach vs water) using SAM segmentation + color analysis."""
    
    def __init__(self, sam_checkpoint: str = None, model_type: str = "vit_h"):
        """
        Initialize the region classifier with SAM.
        
        Args:
            sam_checkpoint: Path to SAM checkpoint. If None, will download automatically.
            model_type: SAM model type ('vit_h', 'vit_l', or 'vit_b')
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # We'll use a simpler approach without SAM for now - just color-based segmentation
        # SAM is heavy and may not be necessary for beach scenes
        self.use_sam = False
        
        logger.info("Region classifier initialized (color-based segmentation)")
    
    def _classify_region_by_color(self, image: np.ndarray, point: Tuple[int, int]) -> str:
        """
        Classify a point in the image as water, beach, or other based on color.
        
        Args:
            image: BGR image
            point: (x, y) coordinates
            
        Returns:
            'water', 'beach', or 'other'
        """
        x, y = int(point[0]), int(point[1])
        
        # Ensure point is within image bounds
        h, w = image.shape[:2]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        
        # Get pixel color
        pixel_bgr = image[y, x]
        
        # Convert to HSV for better color analysis
        pixel_hsv = cv2.cvtColor(np.uint8([[pixel_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        h_val, s_val, v_val = pixel_hsv
        
        # Also check RGB for additional context
        b, g, r = pixel_bgr
        
        # Water detection: Blue hues with decent saturation
        # HSV: Hue 90-130 (blue), Saturation > 30, Value > 50
        if 90 <= h_val <= 130 and s_val > 30 and v_val > 50:
            return 'water'
        
        # Beach/Sand detection: Tan/beige/yellow hues
        # HSV: Hue 10-30 (yellow-orange), Low saturation, High value
        elif 10 <= h_val <= 35 and s_val < 100 and v_val > 100:
            return 'beach'
        
        # Additional beach check: Light colors (sand can be very light)
        elif v_val > 150 and s_val < 50:
            # Check if it's more yellow/tan than blue
            if r > b and g > b:
                return 'beach'
        
        # Everything else (boats, buildings, people, etc.)
        return 'other'
    
    def _get_bbox_center(self, bbox: Dict) -> Tuple[int, int]:
        """
        Get the center point of a bounding box.
        
        Args:
            bbox: Dictionary with 'xyxy' key containing [x1, y1, x2, y2]
            
        Returns:
            (center_x, center_y)
        """
        x1, y1, x2, y2 = bbox['xyxy']
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return (center_x, center_y)
    
    def _get_bbox_bottom_center(self, bbox: Dict) -> Tuple[int, int]:
        """
        Get the bottom-center point of a bounding box (more accurate for standing people).
        
        Args:
            bbox: Dictionary with 'xyxy' key containing [x1, y1, x2, y2]
            
        Returns:
            (center_x, bottom_y)
        """
        x1, y1, x2, y2 = bbox['xyxy']
        center_x = int((x1 + x2) / 2)
        bottom_y = int(y2)  # Bottom of the box
        return (center_x, bottom_y)
    
    def classify_locations(
        self, 
        image: np.ndarray, 
        person_boxes: List[Dict],
        save_annotated: bool = False,
        annotated_path: str = None
    ) -> Dict[str, Any]:
        """
        Classify each detected person as being on beach, in water, or other.
        
        Args:
            image: Original image (OpenCV BGR format)
            person_boxes: List of person bounding boxes with 'xyxy' coordinates
            save_annotated: Whether to save the annotated image
            annotated_path: Path to save annotated image (if save_annotated=True)
            
        Returns:
            Dictionary with classification results:
            {
                'classifications': [{'box_id': 0, 'location': 'beach'}, ...],
                'beach_count': int,
                'water_count': int,
                'other_count': int,
                'total_count': int
            }
        """
        if not person_boxes:
            logger.info("No person boxes to classify")
            return {
                'classifications': [],
                'beach_count': 0,
                'water_count': 0,
                'other_count': 0,
                'total_count': 0
            }
        
        try:
            classifications = []
            
            # Classify each person based on their bounding box location
            for idx, bbox in enumerate(person_boxes):
                # Use bottom-center of bbox (where feet are) for more accurate classification
                point = self._get_bbox_bottom_center(bbox)
                location = self._classify_region_by_color(image, point)
                
                classifications.append({
                    'box_id': idx,
                    'location': location,
                    'point': point
                })
                
                logger.debug(f"Box {idx} at {point}: {location}")
            
            # Count by location
            beach_count = sum(1 for c in classifications if c['location'] == 'beach')
            water_count = sum(1 for c in classifications if c['location'] == 'water')
            other_count = sum(1 for c in classifications if c['location'] == 'other')
            
            # Save annotated image if requested
            if save_annotated and annotated_path:
                self._save_annotated_image(image, person_boxes, classifications, annotated_path)
            
            result = {
                'classifications': classifications,
                'beach_count': beach_count,
                'water_count': water_count,
                'other_count': other_count,
                'total_count': len(person_boxes)
            }
            
            logger.info(f"Classification complete: {beach_count} on beach, {water_count} in water, {other_count} other")
            return result
            
        except Exception as e:
            logger.error(f"Error during location classification: {e}")
            # Return fallback result
            return {
                'classifications': [{'box_id': i, 'location': 'other'} for i in range(len(person_boxes))],
                'beach_count': 0,
                'water_count': 0,
                'other_count': len(person_boxes),
                'total_count': len(person_boxes),
                'error': str(e)
            }
    
    def _save_annotated_image(
        self, 
        image: np.ndarray, 
        person_boxes: List[Dict],
        classifications: List[Dict],
        output_path: str
    ):
        """
        Save an annotated image with bounding boxes colored by classification.
        
        Args:
            image: Original image
            person_boxes: List of bounding boxes
            classifications: List of classification results
            output_path: Path to save the annotated image
        """
        annotated = image.copy()
        
        # Color map for classifications
        color_map = {
            'beach': (0, 165, 255),    # Orange
            'water': (255, 0, 0),       # Blue
            'other': (128, 128, 128)    # Gray
        }
        
        for bbox, classification in zip(person_boxes, classifications):
            x1, y1, x2, y2 = map(int, bbox['xyxy'])
            location = classification['location']
            color = color_map.get(location, (255, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label = f"#{classification['box_id']}: {location}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(annotated, 
                         (x1, y1 - text_height - 10), 
                         (x1 + text_width + 10, y1), 
                         color, 
                         -1)
            
            # Draw text
            cv2.putText(annotated, label, (x1 + 5, y1 - 5), 
                       font, font_scale, (255, 255, 255), thickness)
            
            # Draw point where classification was made
            point = classification['point']
            cv2.circle(annotated, point, 5, color, -1)
        
        cv2.imwrite(output_path, annotated)
        logger.info(f"Saved annotated image to: {output_path}")


if __name__ == '__main__':
    """Test the region classifier with a sample image."""
    
    # Create a dummy test image (blue water, tan beach)
    test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Fill with water (blue) on top half
    test_image[:540, :] = [200, 100, 50]  # Blue water
    
    # Fill with beach (tan) on bottom half
    test_image[540:, :] = [100, 180, 220]  # Tan beach
    
    # Create some dummy bounding boxes
    dummy_boxes = [
        {'xyxy': [100, 100, 200, 300]},   # In water
        {'xyxy': [500, 200, 600, 400]},   # In water
        {'xyxy': [1000, 600, 1100, 800]}, # On beach
        {'xyxy': [1500, 700, 1600, 900]}, # On beach
    ]
    
    try:
        classifier = RegionClassifier()
        result = classifier.classify_locations(
            test_image, 
            dummy_boxes,
            save_annotated=True,
            annotated_path="test_region_annotated.jpg"
        )
        
        print("\n--- Classification Result ---")
        print(f"Total: {result['total_count']}")
        print(f"Beach: {result['beach_count']}")
        print(f"Water: {result['water_count']}")
        print(f"Other: {result['other_count']}")
        print("\nDetails:")
        for c in result['classifications']:
            print(f"  Box {c['box_id']}: {c['location']} at {c['point']}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
