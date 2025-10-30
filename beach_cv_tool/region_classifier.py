"""
Region classifier using water-beach-classifier segmentation model to determine if people are on beach or in water.

ARCHITECTURE:
1. Fine-tuned YOLO model detects people and returns bounding boxes
2. Apply polygon mask to image (mask.py)
3. Run segmentation model to classify pixels as water/beach/background
4. Map each person's bounding box center to the corresponding segmented region

This approach uses a trained SegFormer model for accurate beach/water segmentation.
"""
import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path
import os
import yaml
import torch
from transformers import AutoModelForSemanticSegmentation
import torch.nn.functional as F
import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from project root .env
project_root = Path(__file__).parent.parent
dotenv_path = project_root / '.env'
try:
    load_dotenv(dotenv_path=dotenv_path)
except Exception:
    pass


class RegionClassifier:
    """Classifies person locations (beach vs water) using trained segmentation model."""
    
    def __init__(self, model_path: str = None, config_path: str = "config.yaml"):
        """
        Initialize the region classifier with water-beach segmentation model.
        
        Args:
            model_path: Path to trained model. If None, uses default path.
            config_path: Path to configuration file.
        """
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            # Provide detailed error information if CUDA is not available
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
                try:
                    # Provides detailed error message if CUDA fails to initialize
                    torch.cuda.init()
                except Exception as e:
                    print(f"CUDA Initialization Error: {e}")

        print(f"--- Using device: {self.device} ---")

        logger.info(f"Using device: {self.device}")
        
        # Strategy:
        # 1) If model_path is provided, try to load directly from there
        # 2) Else, download from S3 (or use cached) into ./models_cache/segmentation/waterline and load

        if model_path is not None:
            source_dir = Path(model_path)
            logger.info(f"Attempting to load segmentation model from provided path: {source_dir}")
            self._load_model_from_dir(source_dir)
        else:
            # S3 locations (with env overrides)
            seg_bucket = os.getenv("SEG_S3_BUCKET_NAME", "beach-detection-model-yolo-finetuned")
            seg_config_key = os.getenv("SEG_S3_CONFIG_KEY", "waterline-model-config.json")
            seg_weights_key = os.getenv("SEG_S3_WEIGHTS_KEY", "waterline-model.safetensors")
            cache_dir = Path(os.getenv("SEG_LOCAL_CACHE_DIR", "./models_cache/segmentation/waterline")).resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)

            local_config = cache_dir / "config.json"
            local_weights = cache_dir / "model.safetensors"

            # Download if missing
            if not local_config.exists() or not local_weights.exists():
                logger.info(
                    f"Downloading segmentation model from s3://{seg_bucket}/{seg_config_key} and s3://{seg_bucket}/{seg_weights_key}"
                )
                try:
                    s3 = boto3.client('s3')
                    s3.download_file(seg_bucket, seg_config_key, str(local_config))
                    s3.download_file(seg_bucket, seg_weights_key, str(local_weights))
                    logger.info(f"Downloaded segmentation model to: {cache_dir}")
                except NoCredentialsError:
                    logger.error("AWS credentials not found for segmentation model download. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env")
                    raise
                except Exception as e:
                    logger.error(f"Failed to download segmentation model: {e}")
                    raise

            # Load from cache dir
            self._load_model_from_dir(cache_dir)
        
        # Load polygon mask from config
        self.mask_polygon = np.array(self.config['segmentation']['mask_polygon'])
        
        # Load class mapping from config
        self.id2label = self.config['segmentation']['class_mapping']

        logger.info("Region classifier initialized with segmentation model")

    def _load_model_from_dir(self, model_dir: Path) -> None:
        """
        Load the HuggingFace segmentation model from a local directory.
        Expects config.json and model.safetensors to be present.
        """
        try:
            self.model = AutoModelForSemanticSegmentation.from_pretrained(str(model_dir)).to(self.device).eval()
            logger.info(f"Loaded segmentation model from {model_dir}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_dir}: {e}")
            raise
    
    def _apply_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Apply polygon mask to image (keeps only beach area).
        
        Args:
            image: BGR image
            
        Returns:
            Masked image
        """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self.mask_polygon], 255)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return masked_image
    
    def _segment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Run segmentation model on image.
        
        Args:
            image: BGR image (masked)
            
        Returns:
            Segmentation map (H x W) with class IDs
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size (512x512)
        h, w = image_rgb.shape[:2]
        image_resized = cv2.resize(image_rgb, (512, 512))
        
        # Normalize and convert to tensor (ImageNet normalization for pre-trained model)
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        # Add batch dimension and move to device
        image_batch = image_tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(pixel_values=image_batch)
            logits = outputs.logits
            
            # Upsample to original size
            logits = F.interpolate(
                logits,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )
            
            # Get class predictions
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        
        return pred
    
    def _classify_point_from_segmentation(self, segmentation: np.ndarray, point: Tuple[int, int]) -> str:
        """
        Classify a point based on segmentation map.
        
        Args:
            segmentation: Segmentation map (H x W)
            point: (x, y) coordinates
            
        Returns:
            'water', 'beach', or 'other'
        """
        x, y = int(point[0]), int(point[1])
        
        # Ensure point is within bounds
        h, w = segmentation.shape
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        
        # Get class ID at point
        class_id = segmentation[y, x]
        class_name = self.id2label.get(class_id, 'other')
        
        # Map to simplified categories
        if class_name == 'water':
            return 'water'
        elif class_name == 'beach':
            return 'beach'
        else:
            return 'other'
    
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
            # Apply mask and run segmentation
            masked_image = self._apply_mask(image)
            segmentation = self._segment_image(masked_image)
            
            classifications = []
            
            # Classify each person based on their bounding box location
            for idx, bbox in enumerate(person_boxes):
                # Use bottom-center of bbox (where feet are) for more accurate classification
                point = self._get_bbox_bottom_center(bbox)
                location = self._classify_point_from_segmentation(segmentation, point)
                
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
