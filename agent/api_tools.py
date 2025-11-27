"""
LangGraph tools that use the FastAPI inference service.

These tools call the FastAPI endpoints instead of directly invoking the models,
allowing the agent to use the inference service like any other application.
"""
import logging
from typing import Dict, Any
from langchain_core.tools import tool
import requests
from pathlib import Path
import os

logger = logging.getLogger(__name__)

# API endpoint configuration
API_BASE_URL = "http://localhost:8000"


@tool
def capture_snapshot_api_tool() -> str:
    """
    Capture a NEW snapshot from the beach livestream using the API.
    
    Use this tool when the user explicitly wants to see what the beach looks like RIGHT NOW.
    
    Returns:
        Path to the newly captured snapshot image.
    """
    try:
        response = requests.post(f"{API_BASE_URL}/capture", timeout=30)
        response.raise_for_status()
        
        result = response.json()
        snapshot_url = result['snapshot_url']
        
        # Convert URL to local path (assumes API serves from same filesystem)
        # In production, you might download the image or use the URL directly
        return f"Successfully captured fresh beach snapshot. Processing time: {result['processing_time_ms']:.0f}ms. Image URL: {snapshot_url}"
    except Exception as e:
        logger.error(f"Error capturing snapshot via API: {e}")
        return f"Failed to capture snapshot: {str(e)}"


@tool
def analyze_beach_api_tool(image_path: str = None) -> str:
    """
    Analyze the beach for people and boat counts using the API.
    
    This tool calls the FastAPI inference service to run detection and segmentation.
    If no image_path is provided, it will first capture a snapshot.
    
    Args:
        image_path: Optional path to image to analyze. If None, captures new snapshot.
    
    Returns:
        Detailed analysis of beach activity including counts.
    """
    try:
        # If no image provided, capture one first
        if not image_path:
            capture_response = requests.post(f"{API_BASE_URL}/capture", timeout=30)
            capture_response.raise_for_status()
            capture_result = capture_response.json()
            
            # Get the actual file path from the URL
            # This assumes the API and agent run on the same machine
            snapshot_url = capture_result['snapshot_url']
            # Extract filename and construct local path
            filename = snapshot_url.split('/')[-1]
            image_path = str(Path("data/snapshots") / filename)
        
        # Open and send the image to the analyze endpoint
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f, 'image/jpeg')}
            response = requests.post(f"{API_BASE_URL}/analyze", files=files, timeout=60)
            response.raise_for_status()
        
        result = response.json()
        
        # Format response
        beach_count = result['beach_count']
        water_count = result['water_count']
        other_count = result['other_count']
        people_count = result['people_count']
        boat_count = result['boat_count']
        
        response_text = f"Current beach status at Kaanapali Beach: {result['summary']}. "
        response_text += f"Detected {people_count} people total "
        
        if beach_count > 0 or water_count > 0:
            response_text += f"({beach_count} on beach, {water_count} in water"
            if other_count > 0:
                response_text += f", {other_count} other"
            response_text += ") "
        
        response_text += f"and {boat_count} boats. "
        response_text += f"Processing time: {result['processing_time_ms']:.0f}ms"
        
        return response_text
        
    except Exception as e:
        logger.error(f"Error analyzing beach via API: {e}")
        return f"Failed to analyze beach: {str(e)}"


@tool
def detect_objects_api_tool(image_path: str) -> str:
    """
    Run object detection on an image using the API.
    
    Args:
        image_path: Path to the image to analyze
        
    Returns:
        Detection results with people and boat counts
    """
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f, 'image/jpeg')}
            response = requests.post(
                f"{API_BASE_URL}/detect",
                files=files,
                params={'save_annotated': True},
                timeout=60
            )
            response.raise_for_status()
        
        result = response.json()
        
        return f"Detected {result['people_count']} people and {result['boat_count']} boats. " \
               f"Activity level: {result['activity_level']}. " \
               f"Processing time: {result['processing_time_ms']:.0f}ms"
        
    except Exception as e:
        logger.error(f"Error detecting objects via API: {e}")
        return f"Failed to detect objects: {str(e)}"


@tool
def segment_beach_water_api_tool(image_path: str) -> str:
    """
    Run beach/water segmentation on an image using the API.
    
    Args:
        image_path: Path to the image to segment
        
    Returns:
        Segmentation results with beach/water counts
    """
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f, 'image/jpeg')}
            response = requests.post(
                f"{API_BASE_URL}/segment",
                files=files,
                params={'save_segmented': True},
                timeout=60
            )
            response.raise_for_status()
        
        result = response.json()
        
        return f"Segmentation complete: {result['beach_count']} on beach, " \
               f"{result['water_count']} in water, {result['other_count']} other. " \
               f"Total: {result['total_count']} people. " \
               f"Processing time: {result['processing_time_ms']:.0f}ms"
        
    except Exception as e:
        logger.error(f"Error segmenting via API: {e}")
        return f"Failed to segment: {str(e)}"


@tool
def check_api_health() -> str:
    """
    Check if the inference API is healthy and available.
    
    Returns:
        Health status of the API
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        result = response.json()
        return f"API is healthy. Status: {result['status']}, Timestamp: {result['timestamp']}"
    except Exception as e:
        logger.error(f"API health check failed: {e}")
        return f"API is not available: {str(e)}"
