"""
Object detection module that loads a YOLO model from S3 for production use.

This module provides the BeachDetector class which detects people and boats
in beach images using a fine-tuned YOLOv8 model.
"""
import cv2
import numpy as np
import logging
import os
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


class BeachDetector:
    """
    YOLO-based detector that loads its model from an S3 bucket.
    
    This class is responsible ONLY for object detection (people and boats).
    For complete beach analysis including region classification, use BeachAnalyzer.
    """
    
    def __init__(self, local_cache_dir: str = "./models_cache", model_path: str = None):
        """
        Initialize the detector.
        
        Args:
            local_cache_dir: Directory to cache the YOLO model
        """
        # Hardcoded S3 location for the fine-tuned YOLO model
        self.bucket_name = "beach-detection-model-yolo-finetuned"
        self.model_key = "object_detection_v2.pt"
        # If a local model path is provided, prefer it over S3/cache
        if model_path:
            self.local_model_path = Path(model_path)
        else:
            self.local_model_path = Path(local_cache_dir) / Path(self.model_key).name
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

        # Run YOLO inference on the default device (CUDA when available)
        results = self.model(roi_image, conf=conf_threshold, verbose=False, imgsz=1920)
        result = results[0]
        
        # Safely get class IDs, handle cases with no detections
        counts = result.boxes.cls.tolist() if result.boxes is not None else []

        # YOLO class IDs (v2): 0=boat, 1=chair, 2=person, 3=surfboard, 4=umbrella
        people_count = counts.count(2)  # 2 is 'person'
        boat_count = counts.count(0)    # 0 is 'boat'

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
