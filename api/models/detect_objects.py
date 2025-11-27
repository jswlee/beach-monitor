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
    
    def __init__(self, local_cache_dir: str = "./models_cache"):
        """
        Initialize the detector. The model configuration is read from environment
        variables and the model file is downloaded from S3 if not cached locally.
        
        Args:
            local_cache_dir: Directory to cache the YOLO model
        """
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        self.model_key = os.getenv("S3_MODEL_KEY")
        
        if not self.bucket_name or not self.model_key:
            raise ValueError("S3_BUCKET_NAME and S3_MODEL_KEY environment variables must be set.")

        self.local_model_path = Path(local_cache_dir) / Path(self.model_key).name
        self.model = None
        self._load_model()
        
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
            logger.info(f"Loading YOLO model from: {self.local_model_path}")
            self.model = YOLO(self.local_model_path)
            
        except NoCredentialsError:
            logger.error("AWS credentials not found. Ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are in your .env file.")
            raise
        except Exception as e:
            logger.error(f"An error occurred while loading the model: {e}")
            raise
    
    def detect_objects(self, image: np.ndarray, conf_threshold: float = 0.25) -> Tuple[Dict[str, Any], Any]:
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

        results = self.model(image, conf=conf_threshold, verbose=False)
        result = results[0]
        
        # Safely get class IDs, handle cases with no detections
        counts = result.boxes.cls.tolist() if result.boxes is not None else []

        people_count = counts.count(1)  # 1 is 'person'
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
        dummy_image_path = "dummy_beach_image.jpg"
        dummy_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.imwrite(dummy_image_path, dummy_image)
        
        # Load and detect
        image = cv2.imread(dummy_image_path)
        detection_result, raw_results = detector.detect_objects(image)
        
        print("\n--- Detection Result ---")
        print(f"People: {detection_result['people_count']}")
        print(f"Boats: {detection_result['boat_count']}")
        
        # Save annotated image
        annotated = raw_results[0].plot()
        cv2.imwrite("test_annotated.jpg", annotated)
        print("Saved annotated image to: test_annotated.jpg")

    except (ValueError, NoCredentialsError) as e:
        logger.error(f"Failed to initialize BeachDetector: {e}")
