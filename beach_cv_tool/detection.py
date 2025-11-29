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
from beach_cv_tool.region_classifier import RegionClassifier
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
    """YOLO-based detector that loads its model from an S3 bucket."""
    
    def __init__(self, local_cache_dir: str = "./models_cache", enable_location_classification: bool = True):
        """
        Initializes the detector. The model configuration is read from environment
        variables and the model file is downloaded from S3 if not cached locally.
        
        Args:
            local_cache_dir: Directory to cache the YOLO model
            enable_location_classification: Whether to use VLM for beach/water classification
        """
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        self.model_key = os.getenv("S3_MODEL_KEY")
        
        if not self.bucket_name or not self.model_key:
            raise ValueError("S3_BUCKET_NAME and S3_MODEL_KEY environment variables must be set.")

        self.local_model_path = Path(local_cache_dir) / Path(self.model_key).name
        self.model = None
        self._load_model()
        
        # Initialize region classifier if enabled
        self.enable_location_classification = enable_location_classification
        self.region_classifier = None
        if enable_location_classification:
            try:
                self.region_classifier = RegionClassifier()
                logger.info("Region classifier initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize region classifier: {e}. Will continue without location classification.")
                self.enable_location_classification = False
        
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
            
        except NoCredentialsError:
            logger.error("AWS credentials not found. Ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are in your .env file.")
            raise
        except Exception as e:
            logger.error(f"An error occurred while loading the model: {e}")
            raise
    
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
    
    def detect_objects(self, image: np.ndarray, conf_threshold: float = 0.25) -> Tuple[Dict[str, Any], Any]:
        """
        Detect people and boats in the image.
        """
        if self.model is None:
            logger.error("Model is not loaded. Cannot perform detection.")
            raise ValueError("Model not loaded")

        # Apply ROI mask to exclude pool/boardwalk area from detection
        roi_image = self._apply_roi_mask(image)

        # Run YOLO inference on the default device (CUDA when available)
        results = self.model(roi_image, conf=conf_threshold, verbose=False)
        
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
    
    def analyze_beach_activity(self, image_path: str, classify_locations: bool = None) -> Dict[str, Any]:
        """
        Complete analysis of beach activity from an image path.
        
        Args:
            image_path: Path to the beach image
            classify_locations: Whether to classify people as beach/water. If None, uses instance setting.
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
            
            # Initialize location counts
            beach_count = 0
            water_count = 0
            other_count = 0
            location_classifications = None
            
            # Perform location classification if enabled and there are people
            should_classify = (classify_locations if classify_locations is not None else self.enable_location_classification)
            if should_classify and self.region_classifier and people_count > 0:
                try:
                    # Extract person bounding boxes from YOLO results
                    result = raw_results[0]
                    person_boxes = []
                    
                    if result.boxes is not None:
                        for i, cls in enumerate(result.boxes.cls):
                            if int(cls) == 2:  # 2 is 'person' class
                                box = result.boxes.xyxy[i].cpu().numpy()
                                person_boxes.append({'xyxy': box.tolist()})
                    
                    # Classify locations by running the fine-tuned segmentation model
                    numbered_annotated_path = source_path.parent / f"{source_path.stem}_segmented.jpg"
                    location_result = self.region_classifier.classify_locations(
                        image,
                        person_boxes,
                        save_annotated=True,
                        annotated_path=str(numbered_annotated_path)
                    )
                    
                    beach_count = location_result['beach_count']
                    water_count = location_result['water_count']
                    other_count = location_result['other_count']
                    location_classifications = location_result['classifications']
                    
                    logger.info(f"Location classification: {beach_count} on beach, {water_count} in water, {other_count} other")
                    
                except Exception as e:
                    logger.error(f"Error during location classification: {e}. Continuing without location data.")
            
            # Determine activity level
            if people_count == 0 and boat_count == 0:
                activity_level = "empty"
            elif people_count <= 5:
                activity_level = "quiet"
            elif people_count <= 15:
                activity_level = "moderate"
            else:
                activity_level = "busy"
            
            # Build summary
            if beach_count > 0 or water_count > 0:
                summary = f"Beach is {activity_level} with {people_count} people ({beach_count} on beach, {water_count} in water"
                if other_count > 0:
                    summary += f", {other_count} other"
                summary += f") and {boat_count} boats visible."
            else:
                summary = f"Beach is {activity_level} with {people_count} people and {boat_count} boats visible."
            
            analysis = {
                'image_path': image_path,
                'annotated_image_path': str(annotated_image_path),
                'people_count': people_count,
                'boat_count': boat_count,
                'beach_count': beach_count,
                'water_count': water_count,
                'other_count': other_count,
                'activity_level': activity_level,
                'summary': summary
            }
            
            # Add location classifications if available
            if location_classifications:
                analysis['location_classifications'] = location_classifications
                analysis['regions_image_path'] = str(numbered_annotated_path)
            
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