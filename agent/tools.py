"""
Beach monitoring tools for the agent - API client version

These tools interact with the Beach Monitor API service via HTTP,
with no direct model imports. This allows the agent to run independently
from the CV models, enabling containerization and separation of concerns.

Modular tools that can be composed:
1. capture_snapshot_tool - Captures and returns the image path
2. analyze_beach_tool - Analyzes a snapshot (uses cached if available)
3. get_weather_tool - Gets weather from snapshot using vision model
"""
import logging
from typing import Dict, Any, Optional
from langchain_core.tools import tool
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pathlib import Path
from datetime import datetime, timedelta

from agent.api_client import get_api_client

logger = logging.getLogger(__name__)

# Global cache for snapshot (so tools can share the same snapshot)
_snapshot_cache = {
    'path': None,
    'url': None,
    'timestamp': None,
    'cache_duration': timedelta(minutes=5),  # Cache for 5 minutes
    'last_analysis': {
        'people_count': None,
        'boat_count': None,
        'beach_count': None,
        'water_count': None,
        'other_count': None,
        'summary': None,
        'annotated_url': None,
        'segmented_url': None,
    }
}


class BeachMonitoringTool:
    """Shared tool instance for beach monitoring operations via API"""
    
    def __init__(self):
        """Initialize beach monitoring tool with API client"""
        self.api_client = get_api_client()
        self.vision_llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)
        logger.info("Initialized BeachMonitoringTool with API client")
    
    def capture_snapshot(self, force_new: bool = False) -> Dict[str, str]:
        """
        Capture a beach snapshot via API (with caching)
        
        Args:
            force_new: If True, always capture a new snapshot
        
        Returns:
            Dictionary with 'path' and 'url' keys
        """
        global _snapshot_cache
        
        # Check if we have a valid cached snapshot
        if not force_new and _snapshot_cache['path'] and _snapshot_cache['timestamp']:
            age = datetime.now() - _snapshot_cache['timestamp']
            if age < _snapshot_cache['cache_duration']:
                logger.info(f"Using cached snapshot from {age.seconds}s ago")
                return {
                    'path': _snapshot_cache['path'],
                    'url': _snapshot_cache['url']
                }
        
        # Capture new snapshot via API
        logger.info("Capturing new beach snapshot via API...")
        try:
            result = self.api_client.capture_snapshot()
            snapshot_url = result['snapshot_url']
            
            # Download the snapshot
            snapshot_path = self.api_client.download_image(snapshot_url)
            
            # Update cache
            _snapshot_cache['path'] = snapshot_path
            _snapshot_cache['url'] = snapshot_url
            _snapshot_cache['timestamp'] = datetime.now()
            
            logger.info(f"Snapshot captured: {snapshot_path}")
            return {
                'path': snapshot_path,
                'url': snapshot_url
            }
            
        except Exception as e:
            logger.error(f"Failed to capture snapshot: {e}")
            raise
    
    def analyze_snapshot(self, snapshot_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a beach snapshot via API
        
        Args:
            snapshot_path: Path to snapshot. If None, uses cached snapshot.
        
        Returns:
            Dictionary with analysis results
        """
        global _snapshot_cache
        
        # Use cached snapshot if no path provided, or capture new one
        if snapshot_path is None:
            if not _snapshot_cache['path']:
                # No cached snapshot, capture a fresh one
                logger.info("No cached snapshot, capturing fresh one...")
                capture_result = self.capture_snapshot(force_new=True)
                snapshot_path = capture_result['path']
            else:
                snapshot_path = _snapshot_cache['path']
        
        logger.info(f"Analyzing snapshot via API: {snapshot_path}")
        
        try:
            # Call API for analysis
            analysis = self.api_client.analyze_beach(snapshot_path)
            
            # Update cache
            _snapshot_cache['last_analysis'] = {
                'people_count': analysis['people_count'],
                'boat_count': analysis['boat_count'],
                'beach_count': analysis['beach_count'],
                'water_count': analysis['water_count'],
                'other_count': analysis['other_count'],
                'summary': analysis['summary'],
                'annotated_url': analysis['annotated_image_url'],
                'segmented_url': analysis.get('segmented_image_url'),
            }
            
            return {
                'success': True,
                'snapshot_path': snapshot_path,
                'people_count': analysis['people_count'],
                'boat_count': analysis['boat_count'],
                'beach_count': analysis['beach_count'],
                'water_count': analysis['water_count'],
                'other_count': analysis['other_count'],
                'activity_level': analysis['activity_level'],
                'summary': analysis['summary'],
                'annotated_image_url': analysis['annotated_image_url'],
                'segmented_image_url': analysis.get('segmented_image_url'),
                'processing_time_ms': analysis['processing_time_ms']
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'snapshot_path': snapshot_path
            }
    
    def get_weather_from_snapshot(self, snapshot_path: Optional[str] = None) -> str:
        """
        Use vision model to describe weather conditions from snapshot
        
        Args:
            snapshot_path: Path to snapshot. If None, uses cached snapshot.
        
        Returns:
            Weather description string
        """
        # Use cached snapshot if no path provided
        if snapshot_path is None:
            if not _snapshot_cache['path']:
                raise ValueError("No snapshot available. Capture a snapshot first.")
            snapshot_path = _snapshot_cache['path']
        
        logger.info(f"Getting weather from snapshot: {snapshot_path}")
        
        try:
            # Encode image to base64
            with open(snapshot_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Create message with image
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Describe the weather conditions visible in this beach image. "
                               "Focus on: sky conditions (clear/cloudy/overcast), "
                               "wave conditions (calm/moderate/rough), "
                               "visibility, and any notable weather features. "
                               "Keep it concise (2-3 sentences)."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ]
            )
            
            # Get response from vision model
            response = self.vision_llm.invoke([message])
            weather_description = response.content
            
            logger.info(f"Weather description: {weather_description}")
            return weather_description
            
        except Exception as e:
            logger.error(f"Failed to get weather: {e}")
            return f"Unable to determine weather conditions: {str(e)}"


# Singleton instance
_beach_monitoring_tool = None


def get_beach_monitoring_tool():
    """Get the shared BeachMonitoringTool instance"""
    global _beach_monitoring_tool
    if _beach_monitoring_tool is None:
        _beach_monitoring_tool = BeachMonitoringTool()
    return _beach_monitoring_tool


# LangChain tool definitions

@tool
def capture_snapshot_tool(force_new: bool = False) -> str:
    """
    Capture a snapshot from the beach livestream.
    
    Args:
        force_new: Set to True to force a new capture (use when user says "now", "current", "fresh")
    
    Returns:
        A message indicating the snapshot was captured successfully with the image path
    """
    try:
        client = get_api_client()
        
        if force_new:
            # Use /capture/fresh for guaranteed fresh snapshot
            result = client.capture_fresh_snapshot()
            snapshot_path = client.download_image(result['snapshot_url'])
            return f"✅ Fresh snapshot captured. Image saved at: {snapshot_path}"
        else:
            # Use regular /capture which may use cache
            result = client.capture_snapshot()
            snapshot_path = client.download_image(result['snapshot_url'])
            return f"✅ Snapshot captured successfully. Image saved at: {snapshot_path}"
    except Exception as e:
        logger.error(f"Snapshot capture failed: {e}")
        return f"❌ Failed to capture snapshot: {str(e)}"


@tool
def analyze_beach_tool(force_fresh: bool = True) -> str:
    """
    Analyze the beach snapshot to count people, boats, and determine locations.
    
    Args:
        force_fresh: Set to True to force a fresh capture and analysis (DEFAULT: True for current data)
                     Set to False only if user explicitly wants cached/historical data
    
    Returns:
        A formatted string with the analysis results including people count, boat count, and location breakdown
    """
    try:
        client = get_api_client()
        
        # If user wants fresh data, force new capture
        if force_fresh:
            logger.info("Using /analyze/fresh for guaranteed fresh analysis")
            # Call fresh analysis endpoint (does not download images by default)
            result = client.analyze_beach_fresh(download_images=False)
            result['success'] = True
        else:
            tool_instance = get_beach_monitoring_tool()
            result = tool_instance.analyze_snapshot(snapshot_path=None)
        
        if not result.get('success', False):
            return f"❌ Analysis failed: {result.get('error', 'Unknown error')}"
        
        # Format the response
        response = f"""✅ Beach Analysis Complete:

👥 People: {result['people_count']} total
   - On beach: {result['beach_count']}
   - In water: {result['water_count']}
   - Other areas: {result['other_count']}

🚤 Boats: {result['boat_count']}

📊 Activity Level: {result['activity_level']}

📝 Summary: {result['summary']}

Processing time: {result['processing_time_ms']:.0f}ms
"""

        # If an annotated image URL is available, download it and append the local path
        try:
            annotated_path = None
            annotated_url = result.get('annotated_image_url')
            if annotated_url:
                annotated_path = client.download_image(annotated_url)
            if annotated_path:
                response += f"\n\nImage saved at: {annotated_path}"
        except Exception as e:
            logger.warning(f"Failed to download annotated image: {e}")

        return response
        
    except Exception as e:
        logger.error(f"Beach analysis failed: {e}")
        return f"❌ Failed to analyze beach: {str(e)}"


@tool
def get_weather_tool(snapshot_path: Optional[str] = None) -> str:
    """
    Analyze weather conditions from the beach snapshot using a vision model.
    
    Args:
        snapshot_path: Optional path to a specific snapshot. If not provided, uses the most recent cached snapshot.
    
    Returns:
        A description of the weather conditions visible in the image
    """
    try:
        tool_instance = get_beach_monitoring_tool()
        weather_description = tool_instance.get_weather_from_snapshot(snapshot_path)
        return f"🌤️ Weather Conditions:\n{weather_description}"
    except Exception as e:
        logger.error(f"Weather analysis failed: {e}")
        return f"❌ Failed to analyze weather: {str(e)}"


@tool
def get_original_image_tool() -> str:
    """
    Get the most recent original beach snapshot without capturing a new one.
    
    Use this tool when the user asks to see the "original image" or "raw image"
    after an analysis has been done.
    
    Returns:
        Path to the most recent snapshot image
    """
    try:
        client = get_api_client()
        image_path = client.get_latest_original_image()
        return f"Here is the original beach image. Image saved at: {image_path}"
    except Exception as e:
        logger.error(f"Failed to get original image: {e}")
        return f"❌ Failed to get original image: {str(e)}. Please run analyze_beach_tool first."


@tool
def get_annotated_image_tool() -> str:
    """
    Get the most recent annotated detection image (with bounding boxes).
    
    This shows the image with bounding boxes around detected people and boats.
    
    Returns:
        Path to the annotated image
    """
    try:
        client = get_api_client()
        image_path = client.get_latest_annotated_image()
        return f"Annotated image is available at: {image_path}"
    except Exception as e:
        logger.error(f"Failed to get annotated image: {e}")
        return f"❌ Failed to get annotated image: {str(e)}. Please run analyze_beach_tool first."


@tool
def get_regions_image_tool() -> str:
    """
    Get the most recent regions classification image (beach vs water per person).
    
    This shows people color-coded by their location (beach, water, or other).
    
    Returns:
        Path to the regions/segmented image
    """
    try:
        client = get_api_client()
        image_path = client.get_latest_segmented_image()
        return f"Regions image is available at: {image_path}"
    except Exception as e:
        logger.error(f"Failed to get regions image: {e}")
        return f"❌ Failed to get regions image: {str(e)}. Please run analyze_beach_tool first."


# Export tools for the agent
BEACH_MONITORING_TOOLS = [
    capture_snapshot_tool,
    analyze_beach_tool,
    get_weather_tool,
    get_original_image_tool,
    get_annotated_image_tool,
    get_regions_image_tool,
]


if __name__ == "__main__":
    # Test the tools
    import sys
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Testing Beach Monitoring Tools (API Version) ===\n")
    
    try:
        # Test capture
        print("1. Testing capture_snapshot_tool...")
        result = capture_snapshot_tool.invoke({"force_new": True})
        print(result)
        
        # Test analysis
        print("\n2. Testing analyze_beach_tool...")
        result = analyze_beach_tool.invoke({})
        print(result)
        
        # Test weather
        print("\n3. Testing get_weather_tool...")
        result = get_weather_tool.invoke({})
        print(result)
        
        print("\n✅ All tools tested successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        sys.exit(1)
