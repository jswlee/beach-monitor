"""
Beach analysis module that combines object detection and region classification.

This module provides the BeachAnalyzer class which coordinates:
1. Object detection (people and boats) using YOLO
2. Region classification (beach vs water) using SegFormer
3. Activity level assessment and summary generation
4. Optional saving of training data in YOLO format
"""
import cv2
import numpy as np
import logging
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class BeachAnalyzer:
    """Analyzes beach activity by combining object detection and region classification."""
    
    def __init__(self, detector=None, region_classifier=None, save_training_data: bool = True):
        """
        Initialize the beach analyzer.
        
        Args:
            detector: BeachDetector instance. If None, will be lazy-loaded.
            region_classifier: RegionClassifier instance. If None, will be lazy-loaded.
            save_training_data: Whether to save training data during inference (default: True)
        """
        self._detector = detector
        self._region_classifier = region_classifier
        self.save_training_data = save_training_data
        
        # Initialize training data saver if enabled
        if self.save_training_data:
            from api.models.save_training_data import get_training_data_saver
            self.training_saver = get_training_data_saver()
        else:
            self.training_saver = None
    
    @property
    def detector(self):
        """Lazy-load detector on first access."""
        if self._detector is None:
            from beach_cv_tool.detect_objects import BeachDetector
            logger.info("Lazy-loading BeachDetector...")
            self._detector = BeachDetector()
        return self._detector
    
    @property
    def region_classifier(self):
        """Lazy-load region classifier on first access."""
        if self._region_classifier is None:
            from beach_cv_tool.classify_regions import RegionClassifier
            logger.info("Lazy-loading RegionClassifier...")
            self._region_classifier = RegionClassifier()
        return self._region_classifier
    
    def analyze_beach_activity(
        self, 
        image_path: str, 
        classify_locations: bool = True,
        save_annotated: bool = True
    ) -> Dict[str, Any]:
        """
        Complete analysis of beach activity from an image path.
        
        Args:
            image_path: Path to the beach image
            classify_locations: Whether to classify people as beach/water
            save_annotated: Whether to save annotated images
        
        Returns:
            Dictionary containing:
                - image_path: Original image path
                - annotated_image_path: Path to annotated image (if saved)
                - people_count: Number of people detected
                - boat_count: Number of boats detected
                - beach_count: Number of people on beach (if classified)
                - water_count: Number of people in water (if classified)
                - other_count: Number of people in other areas (if classified)
                - activity_level: Activity level (empty/quiet/moderate/busy)
                - summary: Human-readable summary
                - location_classifications: Detailed classifications (if available)
                - regions_image_path: Path to segmented image (if available)
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from path: {image_path}")
            
            # Step 1: Detect objects
            logger.info("Running object detection...")
            detection_results, raw_results = self.detector.detect_objects(image)
            
            people_count = detection_results['people_count']
            boat_count = detection_results['boat_count']
            
            # Save detection training data if enabled
            if self.save_training_data and self.training_saver:
                try:
                    self.training_saver.save_detection_data(
                        image_path=image_path,
                        yolo_results=raw_results
                    )
                except Exception as e:
                    logger.warning(f"Failed to save detection training data: {e}")
            
            # Save annotated image if requested
            annotated_image_path = None
            if save_annotated:
                annotated_frame = raw_results[0].plot()
                source_path = Path(image_path)
                annotated_image_path = source_path.parent / f"{source_path.stem}_annotated.jpg"
                cv2.imwrite(str(annotated_image_path), annotated_frame)
                logger.info(f"Saved annotated image to: {annotated_image_path}")
            
            # Initialize location counts
            beach_count = 0
            water_count = 0
            other_count = 0
            location_classifications = None
            regions_image_path = None
            
            # Step 2: Classify locations if enabled and there are people
            if classify_locations and people_count > 0:
                try:
                    logger.info("Running region classification...")
                    
                    # Extract person bounding boxes from YOLO results
                    result = raw_results[0]
                    person_boxes = []
                    
                    if result.boxes is not None:
                        for i, cls in enumerate(result.boxes.cls):
                            if int(cls) == 2:  # 2 is 'person' class in YOLO v2
                                box = result.boxes.xyxy[i].cpu().numpy()
                                person_boxes.append({'xyxy': box.tolist()})
                    
                    # Classify locations using segmentation model
                    source_path = Path(image_path)
                    regions_image_path = source_path.parent / f"{source_path.stem}_segmented.jpg"
                    
                    location_result = self.region_classifier.classify_locations(
                        image,
                        person_boxes,
                        save_annotated=save_annotated,
                        annotated_path=str(regions_image_path) if save_annotated else None
                    )
                    
                    beach_count = location_result['beach_count']
                    water_count = location_result['water_count']
                    other_count = location_result['other_count']
                    location_classifications = location_result['classifications']
                    
                    # Save segmentation training data if enabled
                    if self.save_training_data and self.training_saver:
                        try:
                            # Get segmentation mask from region classifier
                            # We'll need to re-segment to get the mask (lightweight operation)
                            masked_image = self.region_classifier._apply_mask(image)
                            segmentation_mask = self.region_classifier._segment_image(masked_image)
                            
                            self.training_saver.save_segmentation_data(
                                image_path=image_path,
                                segmentation_mask=segmentation_mask
                            )
                        except Exception as e:
                            logger.warning(f"Failed to save segmentation training data: {e}")
                    
                    logger.info(
                        f"Location classification: {beach_count} on beach, "
                        f"{water_count} in water, {other_count} other"
                    )
                    
                except Exception as e:
                    logger.error(f"Error during location classification: {e}. Continuing without location data.")
            
            # Step 3: Determine activity level
            activity_level = self._determine_activity_level(people_count, boat_count)
            
            # Step 4: Build summary
            summary = self._build_summary(
                activity_level, people_count, boat_count,
                beach_count, water_count, other_count
            )
            
            # Build result dictionary
            analysis = {
                'image_path': image_path,
                'people_count': people_count,
                'boat_count': boat_count,
                'beach_count': beach_count,
                'water_count': water_count,
                'other_count': other_count,
                'activity_level': activity_level,
                'summary': summary
            }
            
            # Add optional fields if available
            if annotated_image_path:
                analysis['annotated_image_path'] = str(annotated_image_path)
            
            if location_classifications:
                analysis['location_classifications'] = location_classifications
            
            if regions_image_path and save_annotated:
                analysis['regions_image_path'] = str(regions_image_path)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error during beach activity analysis: {e}")
            return {
                'image_path': image_path,
                'error': str(e),
                'activity_level': 'unknown',
                'summary': 'An error occurred during analysis.'
            }
    
    def _determine_activity_level(self, people_count: int, boat_count: int) -> str:
        """
        Determine activity level based on counts.
        
        Args:
            people_count: Number of people detected
            boat_count: Number of boats detected
        
        Returns:
            Activity level string (empty/quiet/moderate/busy)
        """
        if people_count == 0 and boat_count == 0:
            return "empty"
        elif people_count <= 5:
            return "quiet"
        elif people_count <= 15:
            return "moderate"
        else:
            return "busy"
    
    def _build_summary(
        self,
        activity_level: str,
        people_count: int,
        boat_count: int,
        beach_count: int,
        water_count: int,
        other_count: int
    ) -> str:
        """
        Build human-readable summary of beach activity.
        
        Args:
            activity_level: Activity level string
            people_count: Total number of people
            boat_count: Number of boats
            beach_count: Number of people on beach
            water_count: Number of people in water
            other_count: Number of people in other areas
        
        Returns:
            Summary string
        """
        if beach_count > 0 or water_count > 0:
            summary = (
                f"Beach is {activity_level} with {people_count} people "
                f"({beach_count} on beach, {water_count} in water"
            )
            if other_count > 0:
                summary += f", {other_count} other"
            summary += f") and {boat_count} boats visible."
        else:
            summary = (
                f"Beach is {activity_level} with {people_count} people "
                f"and {boat_count} boats visible."
            )
        
        return summary


if __name__ == '__main__':
    # Test the analyzer
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Create a dummy image for testing
        image_path = "dummy_beach_image.jpg"
        dummy_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.imwrite(image_path, dummy_image)
        print(f"Created dummy image: {image_path}")
    
    try:
        analyzer = BeachAnalyzer()
        result = analyzer.analyze_beach_activity(image_path)
        
        print("\n--- Analysis Result ---")
        for key, value in result.items():
            if key != 'location_classifications':  # Skip detailed classifications
                print(f"{key}: {value}")
        
    except Exception as e:
        logger.error(f"Failed to analyze beach activity: {e}")
        sys.exit(1)
