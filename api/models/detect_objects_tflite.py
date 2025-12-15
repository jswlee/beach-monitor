"""
TFLite-based object detection module for edge deployment.

This module provides a BeachDetectorTFLite class that uses TensorFlow Lite
for faster inference on edge devices or CPUs.
"""
import cv2
import numpy as np
import logging
import os
import yaml
from typing import Dict, Any, Tuple, List
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import TFLite runtime
try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False
    logger.warning("TensorFlow not available. TFLite inference will not work.")


BEACH_ROI = [
    np.array([[2, 447], [1862, 383], [1822, 993], [0, 1073]])
]


class BeachDetectorTFLite:
    """
    TFLite-based detector for edge deployment.
    Uses INT8 quantized model for faster inference.
    """
    
    def __init__(self, model_path: str, data_yaml_path: str = "data.yaml"):
        """
        Initialize the TFLite detector.
        
        Args:
            model_path: Path to the TFLite model file (.tflite)
            data_yaml_path: Path to data.yaml containing class mappings
        """
        if not TFLITE_AVAILABLE:
            raise ImportError("TensorFlow is required for TFLite inference. Install with: pip install tensorflow")
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
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
            self.person_class_id = 3
            self.boat_class_id = 1
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape
        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        
        logger.info(f"TFLite model loaded: {model_path}")
        logger.info(f"Input shape: {self.input_shape}")
        logger.info(f"Input dtype: {self.input_details[0]['dtype']}")
        
        # Check if model is quantized
        self.is_quantized = self.input_details[0]['dtype'] == np.uint8
        if self.is_quantized:
            logger.info("Model is INT8 quantized")
            self.input_scale, self.input_zero_point = self.input_details[0]['quantization']
        else:
            logger.info("Model is FP32")
    
    def _apply_roi_mask(self, image: np.ndarray, exclude_pool: bool = True) -> np.ndarray:
        """Apply ROI mask that excludes the pool/boardwalk area from detection."""
        if not exclude_pool:
            return image

        height, width = image.shape[:2]
        mask = np.ones((height, width), dtype=np.uint8)
        
        exclusion_vertices = np.array([
            [3, 1077], [1903, 1079], [1911, 1074], [1909, 571], [1852, 585],
            [1851, 610], [1792, 627], [1792, 660], [1759, 677], [1792, 692],
            [1654, 744], [1610, 728], [1446, 777], [1308, 808], [1284, 806],
            [1212, 826], [1156, 816], [906, 866], [806, 909], [806, 924],
            [775, 930], [776, 941], [709, 934], [635, 958], [595, 956],
            [363, 991], [254, 1017], [211, 1023], [188, 1018], [46, 1053]
        ], dtype=np.int32)
        
        cv2.fillPoly(mask, [exclusion_vertices], 0)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return masked_image
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for TFLite model input.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image ready for model input
        """
        # Resize to model input size
        img_resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        if self.is_quantized:
            # For INT8 quantized model, input should be uint8 [0, 255]
            img_input = img_rgb.astype(np.uint8)
        else:
            # For FP32 model, normalize to [0, 1]
            img_input = img_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_input = np.expand_dims(img_input, axis=0)
        
        return img_input
    
    def _postprocess_outputs(self, outputs: List[np.ndarray], conf_threshold: float = 0.20) -> Tuple[List, List, List]:
        """
        Postprocess TFLite model outputs to extract detections.
        
        Args:
            outputs: Raw model outputs
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Tuple of (boxes, scores, class_ids)
        """
        # TFLite YOLO output format varies, but typically:
        # outputs[0]: boxes [1, num_detections, 4] (x1, y1, x2, y2)
        # outputs[1]: scores [1, num_detections]
        # outputs[2]: class_ids [1, num_detections]
        
        # This may need adjustment based on your specific model export
        boxes = outputs[0][0]  # Remove batch dimension
        scores = outputs[1][0]
        class_ids = outputs[2][0]
        
        # Filter by confidence threshold
        valid_detections = scores >= conf_threshold
        
        boxes = boxes[valid_detections]
        scores = scores[valid_detections]
        class_ids = class_ids[valid_detections]
        
        return boxes, scores, class_ids
    
    def detect_objects(self, image: np.ndarray, conf_threshold: float = 0.20) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Detect people and boats in the image using TFLite model.
        
        Args:
            image: Input image as numpy array (BGR format from cv2)
            conf_threshold: Confidence threshold for detections (0.0 to 1.0)
        
        Returns:
            Tuple of:
                - Dictionary with detection results
                - Dictionary with raw detection data (boxes, scores, class_ids)
        """
        # Apply ROI mask
        roi_image = self._apply_roi_mask(image)
        
        # Crop to detection ROI
        height, width = roi_image.shape[:2]
        roi_points = BEACH_ROI[0]
        x_coords = roi_points[:, 0]
        y_coords = roi_points[:, 1]
        
        x_min = max(int(x_coords.min()), 0)
        x_max = min(int(x_coords.max()), width - 1)
        y_min = max(int(y_coords.min()), 0)
        y_max = min(int(y_coords.max()), height - 1)
        
        cropped_image = roi_image[y_min:y_max, x_min:x_max]
        
        # Preprocess
        input_data = self._preprocess_image(cropped_image)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get outputs
        outputs = [self.interpreter.get_tensor(detail['index']) for detail in self.output_details]
        
        # Postprocess
        boxes, scores, class_ids = self._postprocess_outputs(outputs, conf_threshold)
        
        # Count people and boats
        people_count = int(np.sum(class_ids == self.person_class_id)) if self.person_class_id is not None else 0
        boat_count = int(np.sum(class_ids == self.boat_class_id)) if self.boat_class_id is not None else 0
        
        analysis = {
            'people_count': people_count,
            'boat_count': boat_count,
        }
        
        # Package raw results for compatibility with PyTorch version
        raw_results = {
            'boxes': boxes,
            'scores': scores,
            'class_ids': class_ids,
            'cropped_image': cropped_image,
        }
        
        logger.info(f"Detected {people_count} people and {boat_count} boats")
        return analysis, raw_results


if __name__ == '__main__':
    # Test the TFLite detector
    try:
        detector = BeachDetectorTFLite(model_path="yolo11n_enlarged_boxes_1920x1080.tflite")
        
        # Load test image
        test_image = cv2.imread("demo-photos/youtube_snapshot_20250902_135358.jpg")
        
        if test_image is not None:
            detection_result, raw_results = detector.detect_objects(test_image)
            
            print("\n--- Detection Result ---")
            print(f"People: {detection_result['people_count']}")
            print(f"Boats: {detection_result['boat_count']}")
        else:
            print("Could not load test image")
    
    except Exception as e:
        logger.error(f"Test failed: {e}")
