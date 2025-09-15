"""
Beach monitoring tools for the agent
"""
import logging
from typing import Dict, Any
from langchain_core.tools import tool
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pathlib import Path
import sys
import os
import cv2

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from beach_cv_tool.capture import BeachCapture
from beach_cv_tool.detection import BeachDetector

logger = logging.getLogger(__name__)

class BeachMonitoringTool:
    """Tool for capturing and analyzing beach activity"""
    
    def __init__(self):
        self.capture = BeachCapture()
        self.detector = BeachDetector()
        self.vision_llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)
    
    def get_current_beach_status(self) -> Dict[str, Any]:
        """
        Capture current beach snapshot and analyze activity
        
        Returns:
            Dictionary with beach status information
        """
        try:
            # Capture current snapshot
            logger.info("Capturing current beach snapshot...")
            snapshot_path = self.capture.capture_snapshot()
            
            # Analyze the snapshot
            logger.info("Analyzing beach activity...")
            analysis = self.detector.analyze_beach_activity(snapshot_path)
            
            # Format response for agent
            response = {
                'success': True,
                'snapshot_path': snapshot_path,
                'annotated_image_path': analysis['annotated_image_path'],
                'people_count': analysis['people_count'],
                'boat_count': analysis['boat_count'],
                'activity_level': analysis['activity_level'],
                'summary': analysis['summary']
            }
            
            logger.info(f"Beach analysis complete: {response['summary']}")
            return response
            
        except Exception as e:
            logger.error(f"Error getting beach status: {e}")
            return {
                'success': False,
                'error': str(e),
                'people_count': 0,
                'boat_count': 0,
                'activity_level': 'unknown',
                'summary': 'Unable to get current beach status'
            }

    def resize_image(self, image_path: str, max_width: int = 800) -> str:
        """Resize an image to reduce file size while maintaining aspect ratio.
        
        Args:
            image_path: Path to the original image
            max_width: Maximum width for the resized image
            
        Returns:
            Path to the resized image
        """
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Could not read image: {image_path}")
                return image_path
                
            # Calculate new dimensions
            height, width = img.shape[:2]
            if width <= max_width:
                return image_path  # No need to resize
                
            # Maintain aspect ratio
            new_height = int(height * (max_width / width))
            resized_img = cv2.resize(img, (max_width, new_height))
            
            # Save resized image
            original_path = Path(image_path)
            resized_path = original_path.parent / f"{original_path.stem}_resized{original_path.suffix}"
            cv2.imwrite(str(resized_path), resized_img)
            
            logger.info(f"Resized image saved to: {resized_path}")
            return str(resized_path)
            
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return image_path  # Return original path if resize fails
    
    def get_weather_from_image(self, image_path: str) -> str:
        """Analyze the weather from an image using a multimodal model."""
        try:
            # Resize the image to reduce API costs
            resized_image_path = self.resize_image(image_path)
            
            with open(resized_image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            response = self.vision_llm.invoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": "Describe the weather conditions in this image. Is it sunny, cloudy, partly cloudy, or overcast? Is the water calm or choppy?"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ]
                    )
                ]
            )
            return response.content
        except Exception as e:
            logger.error(f"Error getting weather from image: {e}")
            return "Unable to determine weather conditions from the image."

@tool
def beach_status_tool() -> str:
    """
    Tool function for LangGraph agent to get current beach status.
    This tool captures a new image and analyzes it for people and boat counts.
    
    Returns:
        String description of current beach activity.
    """
    tool = BeachMonitoringTool()
    result = tool.get_current_beach_status()
    
    if result['success']:
        return (f"Current beach status at Kaanapali Beach: {result['summary']}. "
                f"Detected {result['people_count']} people and {result['boat_count']} boats. "
                f"Annotated image is available at: {result['annotated_image_path']}")
    else:
        return f"Unable to get beach status: {result['error']}"


@tool
def get_weather_tool() -> str:
    """
    Tool function for LangGraph agent to get the current weather conditions from the latest snapshot.
    This tool does not capture a new image; it uses the most recent one.

    Returns:
        String description of the weather conditions.
    """
    tool = BeachMonitoringTool()
    try:
        # Capture current snapshot
        logger.info("Capturing current beach snapshot...")
        snapshot_path = tool.capture.capture_snapshot()
        return tool.get_weather_from_image(str(snapshot_path))
    except Exception as e:
        logger.error(f"Error getting weather from image: {e}")
        return "Unable to determine weather conditions from the image."
