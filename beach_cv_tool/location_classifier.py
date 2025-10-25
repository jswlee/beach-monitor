"""
Location classifier using Vision Language Models to determine if people are on beach or in water.
"""
import os
import base64
import json
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path
import cv2
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent
dotenv_path = project_root / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LocationClassifier:
    """Classifies person locations (beach vs water) using GPT-4V."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the location classifier.
        
        Args:
            api_key: OpenAI API key. If None, will use OPENAI_API_KEY from environment.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o"  # Using GPT-4o which has vision capabilities
        
    def _encode_image(self, image: np.ndarray) -> str:
        """
        Encode OpenCV image to base64 string.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Base64 encoded image string
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Encode to JPEG
        success, buffer = cv2.imencode('.jpg', image_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not success:
            raise ValueError("Failed to encode image")
        
        # Convert to base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    
    def _create_annotated_image(self, image: np.ndarray, boxes: List[Dict]) -> np.ndarray:
        """
        Create an annotated image with numbered bounding boxes.
        
        Args:
            image: Original image
            boxes: List of bounding box dictionaries with 'xyxy' coordinates
            
        Returns:
            Annotated image with numbered boxes
        """
        annotated = image.copy()
        
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box['xyxy'])
            
            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw box number
            label = f"#{idx}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(annotated, 
                         (x1, y1 - text_height - 10), 
                         (x1 + text_width + 10, y1), 
                         (0, 255, 0), 
                         -1)
            
            # Draw text
            cv2.putText(annotated, label, (x1 + 5, y1 - 5), 
                       font, font_scale, (0, 0, 0), thickness)
        
        return annotated
    
    def classify_locations(
        self, 
        image: np.ndarray, 
        person_boxes: List[Dict],
        save_annotated: bool = False,
        annotated_path: str = None
    ) -> Dict[str, Any]:
        """
        Classify each detected person as being on beach or in water.
        
        Args:
            image: Original image (OpenCV BGR format)
            person_boxes: List of person bounding boxes with 'xyxy' coordinates
            save_annotated: Whether to save the annotated image with numbered boxes
            annotated_path: Path to save annotated image (if save_annotated=True)
            
        Returns:
            Dictionary with classification results:
            {
                'classifications': [{'box_id': 0, 'location': 'beach'}, ...],
                'beach_count': int,
                'water_count': int,
                'unclear_count': int,
                'total_count': int
            }
        """
        if not person_boxes:
            logger.info("No person boxes to classify")
            return {
                'classifications': [],
                'beach_count': 0,
                'water_count': 0,
                'unclear_count': 0,
                'total_count': 0
            }
        
        try:
            # Create annotated image with numbered boxes
            annotated_image = self._create_annotated_image(image, person_boxes)
            
            # Save annotated image if requested
            if save_annotated and annotated_path:
                cv2.imwrite(annotated_path, annotated_image)
                logger.info(f"Saved numbered annotated image to: {annotated_path}")
            
            # Encode image
            image_base64 = self._encode_image(annotated_image)
            
            # Create prompt
            prompt = self._create_classification_prompt(len(person_boxes))
            
            # Call GPT-4V
            logger.info(f"Classifying {len(person_boxes)} people using GPT-4V...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.1  # Low temperature for consistent classifications
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            logger.info(f"GPT-4V Response: {response_text}")
            
            # Extract JSON from response
            classifications = self._parse_classification_response(response_text, len(person_boxes))
            
            # Count by location
            beach_count = sum(1 for c in classifications if c['location'] == 'beach')
            water_count = sum(1 for c in classifications if c['location'] == 'water')
            unclear_count = sum(1 for c in classifications if c['location'] == 'unclear')
            
            result = {
                'classifications': classifications,
                'beach_count': beach_count,
                'water_count': water_count,
                'unclear_count': unclear_count,
                'total_count': len(person_boxes)
            }
            
            logger.info(f"Classification complete: {beach_count} on beach, {water_count} in water, {unclear_count} unclear")
            return result
            
        except Exception as e:
            logger.error(f"Error during location classification: {e}")
            # Return fallback result
            return {
                'classifications': [{'box_id': i, 'location': 'unclear'} for i in range(len(person_boxes))],
                'beach_count': 0,
                'water_count': 0,
                'unclear_count': len(person_boxes),
                'total_count': len(person_boxes),
                'error': str(e)
            }
    
    def _create_classification_prompt(self, num_boxes: int) -> str:
        """Create the classification prompt for GPT-4V."""
        return f"""You are analyzing a beach scene with {num_boxes} detected people, each marked with a numbered green bounding box (e.g., #0, #1, #2, etc.).

For EACH numbered box, determine if the person is:
- "beach": On the sand, shore, or dry land
- "water": In the ocean, swimming, wading, or standing in water
- "unclear": Cannot determine with confidence

IMPORTANT: You must classify ALL {num_boxes} boxes (from #0 to #{num_boxes-1}).

Return ONLY a valid JSON object in this exact format:
{{
  "classifications": [
    {{"box_id": 0, "location": "beach"}},
    {{"box_id": 1, "location": "water"}},
    ...
  ]
}}

Do not include any explanation, just the JSON."""
    
    def _parse_classification_response(self, response_text: str, expected_count: int) -> List[Dict]:
        """
        Parse the GPT-4V response and extract classifications.
        
        Args:
            response_text: Raw response from GPT-4V
            expected_count: Expected number of classifications
            
        Returns:
            List of classification dictionaries
        """
        try:
            # Try to find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response_text[start_idx:end_idx]
            data = json.loads(json_str)
            
            classifications = data.get('classifications', [])
            
            # Validate classifications
            if len(classifications) != expected_count:
                logger.warning(f"Expected {expected_count} classifications but got {len(classifications)}")
            
            # Ensure all box_ids are present and locations are valid
            valid_locations = {'beach', 'water', 'unclear'}
            validated = []
            
            for i in range(expected_count):
                # Find classification for this box_id
                classification = next((c for c in classifications if c.get('box_id') == i), None)
                
                if classification and classification.get('location') in valid_locations:
                    validated.append(classification)
                else:
                    # Default to unclear if missing or invalid
                    logger.warning(f"Missing or invalid classification for box {i}, defaulting to 'unclear'")
                    validated.append({'box_id': i, 'location': 'unclear'})
            
            return validated
            
        except Exception as e:
            logger.error(f"Error parsing classification response: {e}")
            # Return all unclear as fallback
            return [{'box_id': i, 'location': 'unclear'} for i in range(expected_count)]


if __name__ == '__main__':
    """Test the location classifier with a sample image."""
    
    # Create a dummy test image
    test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Create some dummy bounding boxes
    dummy_boxes = [
        {'xyxy': [100, 100, 200, 300]},  # Person 1
        {'xyxy': [500, 200, 600, 400]},  # Person 2
        {'xyxy': [1000, 150, 1100, 350]},  # Person 3
    ]
    
    try:
        classifier = LocationClassifier()
        result = classifier.classify_locations(
            test_image, 
            dummy_boxes,
            save_annotated=True,
            annotated_path="test_annotated.jpg"
        )
        
        print("\n--- Classification Result ---")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
