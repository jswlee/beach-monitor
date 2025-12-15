"""
Region classification module using SegFormer to determine if people are on beach or in water.

ARCHITECTURE:
1. Object detector provides person bounding boxes
2. Apply polygon mask to image to focus on beach area
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
try:
    load_dotenv()
except Exception:
    pass

BEACH_ROI = [
    np.array([[2, 447], [1862, 383], [1822, 993], [0, 1073]])
]

EXCLUSION_VERTICES = np.array([
    [3, 1077], [1903, 1079], [1911, 1074], [1909, 571], [1852, 585],
    [1851, 610], [1792, 627], [1792, 660], [1759, 677], [1792, 692],
    [1654, 744], [1610, 728], [1446, 777], [1308, 808], [1284, 806],
    [1212, 826], [1156, 816], [906, 866], [806, 909], [806, 924],
    [775, 930], [776, 941], [709, 934], [635, 958], [595, 956],
    [363, 991], [254, 1017], [211, 1023], [188, 1018], [46, 1053]
], dtype=np.int32)


class RegionClassifier:
    """
    Classifies person locations (beach vs water) using trained segmentation model.
    
    This class is responsible ONLY for region classification/segmentation.
    For complete beach analysis including object detection, use BeachAnalyzer.
    """
    
    def __init__(self, model_path: str = None, config_path: str = "config.yaml", data_yaml_path: str = "data.yaml"):
        """
        Initialize the region classifier with water-beach segmentation model.
        
        Args:
            model_path: Path to trained model. If None, downloads from S3.
            config_path: Path to configuration file.
            data_yaml_path: Path to data.yaml containing YOLO class mappings.
        """
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Load YOLO class mappings from data.yaml
        self.yolo_class_names = {}
        self.person_class_id = None
        
        if os.path.exists(data_yaml_path):
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
                self.yolo_class_names = data_config.get('names', {})
                
                # Find person class ID
                for class_id, class_name in self.yolo_class_names.items():
                    if class_name == 'person':
                        self.person_class_id = class_id
                        break
                        
                logger.info(f"Loaded YOLO class mappings: person={self.person_class_id}")
        else:
            logger.warning(f"data.yaml not found at {data_yaml_path}, using default person class ID")
            self.person_class_id = 2
        
        # Setup device - prefer CUDA, then MPS, then CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

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
            # S3 locations (hardcoded for segmentation v2 model)
            seg_bucket = "beach-detection-model-yolo-finetuned"
            seg_config_key = "segmentation_config_v2.json"
            seg_weights_key = "segmentation_model_v2.safetensors"
            cache_dir = Path("./models_cache/segmentation/waterline").resolve()
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
    
    def _crop_to_detection_roi(self, image: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """Crop image to the detection ROI bounding box.

        Returns cropped image and the (x_min, y_min) offsets so boxes/points
        can be mapped into the cropped coordinate space.
        """
        height, width = image.shape[:2]

        # Crop to the bounding box of the configured beach ROI polygon
        roi_points = BEACH_ROI[0]
        x_coords = roi_points[:, 0]
        y_coords = roi_points[:, 1]

        x_min = max(int(x_coords.min()), 0)
        x_max = min(int(x_coords.max()), width - 1)
        y_min = max(int(y_coords.min()), 0)
        y_max = min(int(y_coords.max()), height - 1)

        cropped_image = image[y_min:y_max, x_min:x_max]
        return cropped_image, x_min, y_min
    
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
        class_id = int(segmentation[y, x])

        # DEBUG: Log what class IDs we're actually seeing
        logger.debug(f"Point ({x}, {y}) has class_id: {class_id}")

        # Interpret raw numeric IDs directly for now.
        # Current assumption based on training:
        #   2 -> beach
        #   3 -> water
        #   anything else -> other/background
        if class_id == 3:
            return 'water'
        elif class_id == 2:
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
        annotated_path: str = None,
        save_raw_segmentation: bool = True
    ) -> Dict[str, Any]:
        """
        Classify each detected person as being on beach, in water, or other.
        
        Args:
            image: Original image (OpenCV BGR format)
            person_boxes: List of person bounding boxes with 'xyxy' coordinates
            save_annotated: Whether to save the annotated image
            annotated_path: Path to save annotated image (if save_annotated=True)
            save_raw_segmentation: Whether to save raw segmentation output for debugging
            
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
            # Match the training pipeline:
            # 1. Apply beach polygon mask on FULL image (as in training)
            # 2. Crop to detection ROI bounding box (for speed)
            # 3. Segment (which internally resizes to 512x512)
            # Note: We don't apply the pool exclusion mask here because the
            # segmentation model was trained without it.
            
            # Step 1: Apply beach mask on full image (training used this)
            masked_full_image = self._apply_mask(image)
            
            # Step 2: Crop to detection ROI bbox (no exclusion mask)
            detection_roi_image, x_offset, y_offset = self._crop_to_detection_roi(masked_full_image)
            
            # Step 3: Segment (resizes to 512x512 internally)
            segmentation = self._segment_image(detection_roi_image)
            
            # Save raw segmentation for debugging if requested
            if save_raw_segmentation:
                self._save_raw_segmentation(segmentation, annotated_path)
            
            classifications = []
            class_ids_seen = []
            
            # Classify each person based on their bounding box location
            for idx, bbox in enumerate(person_boxes):
                # Person boxes are already in cropped ROI coordinates (from YOLO),
                # so we don't need to shift them again.
                # Use bottom-center of bbox (where feet are) for more accurate classification
                point = self._get_bbox_bottom_center(bbox)
                
                # Store the class_id for debugging
                x, y = int(point[0]), int(point[1])
                h, w = segmentation.shape
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                class_id_at_point = int(segmentation[y, x])
                class_ids_seen.append(class_id_at_point)
                
                location = self._classify_point_from_segmentation(segmentation, point)
                
                classifications.append({
                    'box_id': idx,
                    'location': location,
                    'point': point
                })
                
                logger.debug(f"Box {idx} at {point}: {location} (class_id={class_id_at_point})")
            
            # Log summary of class IDs seen
            unique_ids = set(class_ids_seen)
            logger.info(f"Unique class IDs at person locations: {sorted(unique_ids)}")
            
            # Count by location
            beach_count = sum(1 for c in classifications if c['location'] == 'beach')
            water_count = sum(1 for c in classifications if c['location'] == 'water')
            other_count = sum(1 for c in classifications if c['location'] == 'other')
            
            # Save annotated image if requested
            if save_annotated and annotated_path:
                self._save_annotated_image(detection_roi_image, person_boxes, classifications, annotated_path)
            
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
                'other_count': 0,
                'total_count': len(person_boxes),
                'error': str(e)
            }
    
    def _save_raw_segmentation(self, segmentation: np.ndarray, base_path: str = None):
        """
        Save raw segmentation output as a color-coded image for debugging.
        
        Args:
            segmentation: Segmentation map (H x W) with class IDs
            base_path: Base path for the output (will create raw_seg_images folder)
        """
        if base_path is None:
            return
        
        # Create output directory
        raw_seg_dir = Path("raw_seg_images")
        raw_seg_dir.mkdir(exist_ok=True)
        
        # Generate output path
        base_name = Path(base_path).stem
        output_path = raw_seg_dir / f"{base_name}_raw_seg.png"
        
        # Create color-coded visualization
        # Map class IDs to colors
        color_map = {
            0: [0, 0, 0],       # Background - Black
            1: [255, 0, 0],     # Water - Blue (BGR)
            2: [0, 165, 255],   # Beach - Orange (BGR)
        }
        
        # Create RGB image
        h, w = segmentation.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in color_map.items():
            mask = segmentation == class_id
            colored[mask] = color
        
        # Save the image
        cv2.imwrite(str(output_path), colored)
        logger.info(f"Saved raw segmentation to: {output_path}")
    
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
