"""
Object detection module that loads a YOLO model from S3 for production use.

This module provides the BeachDetector class which detects people and boats
in beach images using a fine-tuned YOLOv8 model.
"""
import cv2
import numpy as np
import logging
import os
import yaml
from typing import Dict, Any, Tuple
from pathlib import Path
from ultralytics import YOLO
import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv
import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel


# Load environment variables from a .env file
project_root = Path(__file__).parent.parent
dotenv_path = project_root / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BEACH_ROI = [
    np.array([[2, 447], [1862, 383], [1822, 993], [0, 1073]])
]

class BeachDetector:
    """
    YOLO-based detector that loads its model from an S3 bucket.
    
    This class is responsible ONLY for object detection (people and boats).
    For complete beach analysis including region classification, use BeachAnalyzer.
    """
    
    def __init__(self, model_path: str = None, data_yaml_path: str = "data.yaml"):
        """
        Initialize the BeachDetector.
        
        Args:
            model_path: Path to the YOLO model file. If None, downloads from S3.
            data_yaml_path: Path to data.yaml containing class mappings.
        """
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load class mappings from data.yaml
        self.class_names = {}
        self.person_class_id = None
        self.boat_class_id = None
        
        if os.path.exists(data_yaml_path):
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
                self.class_names = data_config.get('names', {})
                
                # Find person and boat class IDs
                for class_id, class_name in self.class_names.items():
                    if class_name == 'person':
                        self.person_class_id = class_id
                    elif class_name == 'boat':
                        self.boat_class_id = class_id
                        
                logger.info(f"Loaded class mappings: person={self.person_class_id}, boat={self.boat_class_id}")
        else:
            logger.warning(f"data.yaml not found at {data_yaml_path}, using default class IDs")
            self.person_class_id = 2
            self.boat_class_id = 0
        
        # Hardcoded S3 location for the fine-tuned YOLO model
        self.bucket_name = "beach-detection-model-yolo-finetuned"
        self.model_key = "object_detection_v2.pt"
        # If a local model path is provided, prefer it over S3/cache
        if model_path:
            self.local_model_path = Path(model_path)
        else:
            self.local_model_path = Path("./models_cache") / Path(self.model_key).name
        self.model = None
        self._load_model()
    
    def _apply_roi_mask(self, image: np.ndarray, exclude_pool: bool = True) -> np.ndarray:
        """Apply ROI mask that excludes the pool/boardwalk area from detection."""
        if not exclude_pool:
            return image

        height, width = image.shape[:2]

        # Create a white mask (all 1s)
        mask = np.ones((height, width), dtype=np.uint8)

        # Exclusion area polygon (from Roboflow Polygon Zone tool)
        exclusion_vertices = np.array([
            [3, 1077], [1903, 1079], [1911, 1074], [1909, 571], [1852, 585],
            [1851, 610], [1792, 627], [1792, 660], [1759, 677], [1792, 692],
            [1654, 744], [1610, 728], [1446, 777], [1308, 808], [1284, 806],
            [1212, 826], [1156, 816], [906, 866], [806, 909], [806, 924],
            [775, 930], [776, 941], [709, 934], [635, 958], [595, 956],
            [363, 991], [254, 1017], [211, 1023], [188, 1018], [46, 1053]
        ], dtype=np.int32)

        # Fill the exclusion area with black (0s)
        cv2.fillPoly(mask, [exclusion_vertices], 0)

        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return masked_image
        
    def _load_model(self):
        """Load the YOLO model, downloading from S3 if it doesn't exist locally."""
        try:
            # Check if the model is already downloaded in the local cache
            if not self.local_model_path.exists():
                logger.info(f"Model not found locally. Downloading from S3...")
                logger.info(f"Source: s3://{self.bucket_name}/{self.model_key}")
                
                # Ensure the local cache directory exists
                self.local_model_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Boto3 will automatically use credentials from the environment variables
                s3 = boto3.client('s3')
                s3.download_file(self.bucket_name, self.model_key, str(self.local_model_path))
                logger.info(f"Model successfully downloaded to: {self.local_model_path}")

            # Now, load the model from the local file path
            # PyTorch 2.6+ defaults torch.load(weights_only=True), which blocks
            # pickled model classes. Since this checkpoint is our own fine-tuned
            # YOLO model and we trust its source, we temporarily patch torch.load
            # to allow loading with weights_only=False.
            logger.info(f"Loading YOLO model from: {self.local_model_path}")
            
            # Temporarily patch torch.load to use weights_only=False for YOLO
            _original_load = torch.load
            def _safe_load(*args, **kwargs):
                kwargs.setdefault('weights_only', False)
                return _original_load(*args, **kwargs)
            torch.load = _safe_load
            
            try:
                self.model = YOLO(self.local_model_path)
            finally:
                # Restore original torch.load
                torch.load = _original_load
            
            # Log device info for YOLO model
            model_device = next(self.model.model.parameters()).device
            logger.info(f"YOLO model loaded on device: {model_device}")
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found. Ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are in your .env file.")
            raise
        except Exception as e:
            logger.error(f"An error occurred while loading the model: {e}")
            raise
    
    def detect_objects(self, image: np.ndarray, conf_threshold: float = 0.20)-> Tuple[Dict[str, Any], Any]:
        """
        Detect people and boats in the image.
        
        Args:
            image: Input image as numpy array (BGR format from cv2)
            conf_threshold: Confidence threshold for detections (0.0 to 1.0)
        
        Returns:
            Tuple of:
                - Dictionary with detection results:
                    - people_count: Number of people detected
                    - boat_count: Number of boats detected
                - Raw YOLO results object (for accessing boxes, plotting, etc.)
        
        Raises:
            ValueError: If model is not loaded
        """
        if self.model is None:
            logger.error("Model is not loaded. Cannot perform detection.")
            raise ValueError("Model not loaded")

        # Apply ROI mask to exclude pool/boardwalk area from detection
        roi_image = self._apply_roi_mask(image)

        # Crop to the bounding box of the configured beach ROI polygon
        height, width = roi_image.shape[:2]
        roi_points = BEACH_ROI[0]
        x_coords = roi_points[:, 0]
        y_coords = roi_points[:, 1]

        x_min = max(int(x_coords.min()), 0)
        x_max = min(int(x_coords.max()), width - 1)
        y_min = max(int(y_coords.min()), 0)
        y_max = min(int(y_coords.max()), height - 1)

        cropped_image = roi_image[y_min:y_max, x_min:x_max]

        # Run YOLO inference on the cropped ROI image
        results = self.model(cropped_image, conf=conf_threshold, verbose=False, imgsz=1920)
        result = results[0]
        
        # Safely get class IDs, handle cases with no detections
        counts = result.boxes.cls.tolist() if result.boxes is not None else []

        # Count using dynamically loaded class IDs
        people_count = counts.count(self.person_class_id) if self.person_class_id is not None else 0
        boat_count = counts.count(self.boat_class_id) if self.boat_class_id is not None else 0

        analysis = {
            'people_count': people_count,
            'boat_count': boat_count,
        }

        logger.info(f"Detected {people_count} people and {boat_count} boats")
        return analysis, results


if __name__ == '__main__':
    # Test the detector
    try:
        detector = BeachDetector()
        
        # Create a dummy image to test (replace with a real image path)
        dummy_image_path = "demo-photos\youtube_snapshot_20250902_130003_annotated.jpg"
        dummy_image = cv2.imread(dummy_image_path)
        
        # Load and detect
        detection_result, raw_results = detector.detect_objects(dummy_image)
        
        print("\n--- Detection Result ---")
        print(f"People: {detection_result['people_count']}")
        print(f"Boats: {detection_result['boat_count']}")
        
        # Save annotated image
        annotated = raw_results[0].plot()
        cv2.imwrite("test_annotated.jpg", annotated)
        print("Saved annotated image to: test_annotated.jpg")

    except (ValueError, NoCredentialsError) as e:
        logger.error(f"Failed to initialize BeachDetector: {e}")
