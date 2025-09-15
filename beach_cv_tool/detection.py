"""
Beach detection module that loads a model from S3 for production use.
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
    """YOLO-based detector that loads its model from an S3 bucket."""
    
    def __init__(self, local_cache_dir: str = "./models_cache"):
        """
        Initializes the detector. The model configuration is read from environment
        variables and the model file is downloaded from S3 if not cached locally.
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
    
    def analyze_beach_activity(self, image_path: str) -> Dict[str, Any]:
        """
        Complete analysis of beach activity from an image path.
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from path: {image_path}")
            
            detection_results, raw_results = self.detect_objects(image)
            annotated_frame = raw_results[0].plot()

            source_path = Path(image_path)
            annotated_image_path = source_path.parent / f"{source_path.stem}_annotated.jpg"
            cv2.imwrite(str(annotated_image_path), annotated_frame)
            logger.info(f"Saved annotated image to: {annotated_image_path}")

            people_count = detection_results['people_count']
            boat_count = detection_results['boat_count']
            
            if people_count == 0 and boat_count == 0:
                activity_level = "empty"
            elif people_count <= 5:
                activity_level = "quiet"
            elif people_count <= 15:
                activity_level = "moderate"
            else:
                activity_level = "busy"
            
            analysis = {
                'image_path': image_path,
                'annotated_image_path': str(annotated_image_path),
                'people_count': people_count,
                'boat_count': boat_count,
                'activity_level': activity_level,
                'summary': f"Beach is {activity_level} with {people_count} people and {boat_count} boats visible."
            }
            return analysis
            
        except Exception as e:
            logger.error(f"Error during beach activity analysis: {e}")
            return {
                'image_path': image_path,
                'error': str(e),
                'activity_level': 'unknown',
                'summary': 'An error occurred during analysis.'
            }

if __name__ == '__main__':
    # The detector will automatically handle loading env variables and downloading the model
    try:
        detector = BeachDetector()
        
        # Create a dummy image to test (replace with a real image path)
        dummy_image_path = "dummy_beach_image.jpg"
        cv2.imwrite(dummy_image_path, np.zeros((1080, 1920, 3), dtype=np.uint8))
        
        # Analyze the image
        analysis_result = detector.analyze_beach_activity(dummy_image_path)
        print("\n--- Analysis Result ---")
        print(analysis_result)

    except (ValueError, NoCredentialsError) as e:
        logger.error(f"Failed to initialize BeachDetector: {e}")