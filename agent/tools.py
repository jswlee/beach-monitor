"""
Beach monitoring tools for the agent

Modular tools that can be composed:
1. capture_snapshot_tool - Just captures and returns the image path
2. analyze_beach_tool - Analyzes a snapshot (uses cached if available)
3. get_weather_tool - Gets weather from snapshot
"""
import logging
from typing import Dict, Any, Optional
from langchain_core.tools import tool
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pathlib import Path
import sys
import os
import cv2
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from beach_cv_tool.capture import BeachCapture
from beach_cv_tool.detection import BeachDetector

logger = logging.getLogger(__name__)

# Global cache for snapshot (so tools can share the same snapshot)
_snapshot_cache = {
    'path': None,
    'timestamp': None,
    'cache_duration': timedelta(minutes=5)  # Cache for 5 minutes
}

class BeachMonitoringTool:
    """Shared tool instance for beach monitoring operations"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.capture = BeachCapture()
            cls._instance.detector = BeachDetector()
            cls._instance.vision_llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)
        return cls._instance
    
    def capture_snapshot(self, force_new: bool = False) -> str:
        """
        Capture a beach snapshot (with caching)
        
        Args:
            force_new: If True, always capture a new snapshot
        
        Returns:
            Path to the snapshot image
        """
        global _snapshot_cache
        
        # Check if we have a valid cached snapshot
        if not force_new and _snapshot_cache['path'] and _snapshot_cache['timestamp']:
            age = datetime.now() - _snapshot_cache['timestamp']
            if age < _snapshot_cache['cache_duration']:
                logger.info(f"Using cached snapshot from {age.seconds}s ago: {_snapshot_cache['path']}")
                return _snapshot_cache['path']
        
        # Capture new snapshot
        logger.info("Capturing new beach snapshot...")
        snapshot_path = self.capture.capture_snapshot()
        
        # Update cache
        _snapshot_cache['path'] = snapshot_path
        _snapshot_cache['timestamp'] = datetime.now()
        
        logger.info(f"Snapshot captured: {snapshot_path}")
        return snapshot_path
    
    def analyze_snapshot(self, snapshot_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a beach snapshot
        
        Args:
            snapshot_path: Path to snapshot. If None, uses cached snapshot.
        
        Returns:
            Dictionary with analysis results
        """
        try:
            # Use cached snapshot if no path provided
            if snapshot_path is None:
                snapshot_path = _snapshot_cache.get('path')
                if not snapshot_path:
                    raise ValueError("No snapshot available. Capture a snapshot first.")
            
            logger.info(f"Analyzing snapshot: {snapshot_path}")
            analysis = self.detector.analyze_beach_activity(snapshot_path)
            
            response = {
                'success': True,
                'snapshot_path': snapshot_path,
                'annotated_image_path': analysis['annotated_image_path'],
                'people_count': analysis['people_count'],
                'boat_count': analysis['boat_count'],
                'beach_count': analysis.get('beach_count', 0),
                'water_count': analysis.get('water_count', 0),
                'other_count': analysis.get('other_count', 0),
                'activity_level': analysis['activity_level'],
                'summary': analysis['summary']
            }
            
            logger.info(f"Analysis complete: {response['summary']}")
            return response
            
        except Exception as e:
            logger.error(f"Error analyzing snapshot: {e}")
            return {
                'success': False,
                'error': str(e),
                'summary': f'Unable to analyze snapshot: {str(e)}'
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
def get_original_image_tool() -> str:
    """
    Get the most recent original beach snapshot without capturing a new one.
    
    Use this tool when the user asks to see the "original image" or "raw image"
    after an analysis has been done. This returns the unmodified snapshot that
    was used for the most recent analysis.
    
    This is efficient because it doesn't require capturing a new snapshot from YouTube.
    
    Returns:
        Path to the most recent snapshot image.
    """
    global _snapshot_cache
    
    try:
        # Check if we have a cached snapshot
        if _snapshot_cache.get('path'):
            snapshot_path = _snapshot_cache['path']
            if Path(snapshot_path).exists():
                return f"Here is the original beach image. Image saved at: {snapshot_path}"
            else:
                return "No recent snapshot found. Please ask me to analyze the beach first."
        else:
            return "No recent snapshot available. Please ask me to analyze the beach first, or capture a new snapshot."
    except Exception as e:
        logger.error(f"Error getting original image: {e}")
        return f"Failed to get original image: {str(e)}"


@tool
def capture_snapshot_tool() -> str:
    """
    Capture a NEW snapshot from the beach livestream.
    
    Use this tool when the user explicitly wants to see what the beach looks like RIGHT NOW,
    or when they ask "show me the beach" or "what does it look like now".
    
    This captures a fresh image from the YouTube livestream.
    
    Returns:
        Path to the newly captured snapshot image.
    """
    tool = BeachMonitoringTool()
    try:
        snapshot_path = tool.capture_snapshot(force_new=True)
        return f"Successfully captured fresh beach snapshot. Image saved at: {snapshot_path}"
    except Exception as e:
        logger.error(f"Error capturing snapshot: {e}")
        return f"Failed to capture snapshot: {str(e)}"


@tool
def analyze_beach_tool() -> str:
    """
    Analyze the beach for people and boat counts.
    
    This tool will use a recently captured snapshot if available (within 5 minutes),
    or capture a new one if needed. It runs YOLO detection and classifies people
    as being on beach, in water, or other (boats, etc.).
    
    Use this tool when the user asks about beach activity, crowding, or counts.
    
    Returns:
        Detailed analysis of beach activity including counts and annotated image path.
    """
    tool = BeachMonitoringTool()
    try:
        # Capture snapshot if needed
        snapshot_path = tool.capture_snapshot()
        
        # Analyze it
        result = tool.analyze_snapshot(snapshot_path)
        
        if result['success']:
            beach_count = result.get('beach_count', 0)
            water_count = result.get('water_count', 0)
            other_count = result.get('other_count', 0)
            
            response = f"Current beach status at Kaanapali Beach: {result['summary']}. "
            response += f"Detected {result['people_count']} people total "
            
            if beach_count > 0 or water_count > 0:
                response += f"({beach_count} on beach, {water_count} in water"
                if other_count > 0:
                    response += f", {other_count} other"
                response += ") "
            
            response += f"and {result['boat_count']} boats. "
            response += f"Annotated image is available at: {result['annotated_image_path']}"
            
            return response
        else:
            return f"Unable to analyze beach: {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        logger.error(f"Error in analyze_beach_tool: {e}")
        return f"Failed to analyze beach: {str(e)}"


@tool
def get_weather_tool() -> str:
    """
    Get current weather conditions from the beach camera.
    
    This tool captures a snapshot and uses vision AI to describe the weather.
    
    Returns:
        Description of weather conditions (sunny, cloudy, water conditions, etc.).
    """
    tool = BeachMonitoringTool()
    try:
        # Use cached snapshot or capture new one
        snapshot_path = tool.capture_snapshot()
        return tool.get_weather_from_image(str(snapshot_path))
    except Exception as e:
        logger.error(f"Error getting weather: {e}")
        return f"Unable to determine weather conditions: {str(e)}"


# Legacy tool for backward compatibility
@tool
def beach_status_tool() -> str:
    """
    DEPRECATED: Use analyze_beach_tool instead.
    
    Get current beach status (captures and analyzes in one step).
    """
    return analyze_beach_tool.invoke({})
