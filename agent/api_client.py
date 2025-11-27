"""
HTTP client for Beach Monitor API service.

This client provides a clean interface for the agent to interact with
the API service without direct model imports. All CV operations are
delegated to the API service via REST endpoints.
"""
import os
import logging
import requests
from typing import Dict, Any, Optional
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)


class BeachMonitorAPIClient:
    """
    Client for interacting with the Beach Monitor API service.
    
    This client handles all communication with the API service,
    allowing the agent to remain stateless and model-free.
    """
    
    def __init__(self, base_url: str = None):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API service (e.g., http://localhost:8000)
                     If None, reads from API_BASE_URL environment variable
        """
        self.base_url = base_url or os.getenv("API_BASE_URL", "http://localhost:8000")
        self.base_url = self.base_url.rstrip('/')
        logger.info(f"Initialized API client with base URL: {self.base_url}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if the API service is healthy.
        
        Returns:
            Dictionary with health status
            
        Raises:
            requests.RequestException: If API is unreachable
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Health check failed: {e}")
            raise
    
    def capture_snapshot(self) -> Dict[str, Any]:
        """
        Capture a fresh snapshot from the beach livestream.
        
        Returns:
            Dictionary with:
                - snapshot_url: URL to the captured image
                - timestamp: Capture timestamp
                - processing_time_ms: Processing time in milliseconds
                
        Raises:
            requests.RequestException: If capture fails
        """
        try:
            response = requests.post(f"{self.base_url}/capture", timeout=30)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Captured snapshot: {result['snapshot_url']}")
            return result
        except requests.RequestException as e:
            logger.error(f"Snapshot capture failed: {e}")
            raise
    
    def download_image(self, image_url: str, save_path: str = None) -> str:
        """
        Download an image from the API service.
        
        Args:
            image_url: URL path to the image (e.g., /images/snapshot.jpg)
            save_path: Local path to save the image. If None, uses temp file.
            
        Returns:
            Path to the downloaded image
            
        Raises:
            requests.RequestException: If download fails
        """
        try:
            # Construct full URL
            if not image_url.startswith('http'):
                image_url = f"{self.base_url}{image_url}"
            
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Save to file
            if save_path is None:
                suffix = Path(image_url).suffix or '.jpg'
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    save_path = tmp.name
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded image to: {save_path}")
            return save_path
            
        except requests.RequestException as e:
            logger.error(f"Image download failed: {e}")
            raise
    
    def detect_objects(self, image_path: str) -> Dict[str, Any]:
        """
        Detect people and boats in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with:
                - people_count: Number of people detected
                - boat_count: Number of boats detected
                - activity_level: Activity level (low/moderate/high)
                - annotated_image_url: URL to annotated image
                - processing_time_ms: Processing time
                
        Raises:
            requests.RequestException: If detection fails
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(
                    f"{self.base_url}/detect",
                    files=files,
                    timeout=60
                )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Detection: {result['people_count']} people, {result['boat_count']} boats")
            return result
        except requests.RequestException as e:
            logger.error(f"Object detection failed: {e}")
            raise
    
    def segment_regions(self, image_path: str) -> Dict[str, Any]:
        """
        Classify beach vs water regions and person locations.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with:
                - beach_count: People on beach
                - water_count: People in water
                - other_count: People in other areas
                - total_count: Total people
                - segmented_image_url: URL to segmented image
                - processing_time_ms: Processing time
                
        Raises:
            requests.RequestException: If segmentation fails
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(
                    f"{self.base_url}/segment",
                    files=files,
                    timeout=60
                )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Segmentation: {result['beach_count']} beach, {result['water_count']} water")
            return result
        except requests.RequestException as e:
            logger.error(f"Region segmentation failed: {e}")
            raise
    
    def analyze_beach(self, image_path: str) -> Dict[str, Any]:
        """
        Complete beach analysis: detection + segmentation.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with complete analysis:
                - people_count: Total people detected
                - boat_count: Total boats detected
                - beach_count: People on beach
                - water_count: People in water
                - other_count: People in other areas
                - activity_level: Activity level (low/moderate/high)
                - summary: Human-readable summary
                - annotated_image_url: URL to annotated image
                - segmented_image_url: URL to segmented image
                - processing_time_ms: Processing time
                
        Raises:
            requests.RequestException: If analysis fails
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(
                    f"{self.base_url}/analyze",
                    files=files,
                    timeout=90
                )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Analysis complete: {result['summary']}")
            return result
        except requests.RequestException as e:
            logger.error(f"Beach analysis failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary with model information
            
        Raises:
            requests.RequestException: If request fails
        """
        try:
            response = requests.get(f"{self.base_url}/models/info", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get model info: {e}")
            raise
    
    def get_latest_original_image(self, save_path: str = None) -> str:
        """
        Get the most recent original snapshot from the API.
        
        Args:
            save_path: Local path to save the image. If None, uses temp file.
            
        Returns:
            Path to the downloaded image
            
        Raises:
            requests.RequestException: If request fails
        """
        try:
            response = requests.get(f"{self.base_url}/images/latest/original", timeout=30)
            response.raise_for_status()
            
            # Save to file
            if save_path is None:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    save_path = tmp.name
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded latest original image to: {save_path}")
            return save_path
            
        except requests.RequestException as e:
            logger.error(f"Failed to get latest original image: {e}")
            raise
    
    def get_latest_annotated_image(self, save_path: str = None) -> str:
        """
        Get the most recent annotated image (with bounding boxes) from the API.
        
        Args:
            save_path: Local path to save the image. If None, uses temp file.
            
        Returns:
            Path to the downloaded image
            
        Raises:
            requests.RequestException: If request fails
        """
        try:
            response = requests.get(f"{self.base_url}/images/latest/annotated", timeout=30)
            response.raise_for_status()
            
            # Save to file
            if save_path is None:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    save_path = tmp.name
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded latest annotated image to: {save_path}")
            return save_path
            
        except requests.RequestException as e:
            logger.error(f"Failed to get latest annotated image: {e}")
            raise
    
    def get_latest_segmented_image(self, save_path: str = None) -> str:
        """
        Get the most recent segmented image (beach vs water) from the API.
        
        Args:
            save_path: Local path to save the image. If None, uses temp file.
            
        Returns:
            Path to the downloaded image
            
        Raises:
            requests.RequestException: If request fails
        """
        try:
            response = requests.get(f"{self.base_url}/images/latest/segmented", timeout=30)
            response.raise_for_status()
            
            # Save to file
            if save_path is None:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    save_path = tmp.name
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded latest segmented image to: {save_path}")
            return save_path
            
        except requests.RequestException as e:
            logger.error(f"Failed to get latest segmented image: {e}")
            raise

    def capture_fresh_snapshot(self) -> Dict[str, Any]:
        """
        Force capture a FRESH snapshot (no cache).
        
        Returns:
            Dictionary with snapshot_url, timestamp, processing_time_ms
        """
        try:
            response = requests.post(f"{self.base_url}/capture/fresh", timeout=30)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Captured FRESH snapshot: {result['snapshot_url']}")
            return result
        except requests.RequestException as e:
            logger.error(f"Fresh snapshot capture failed: {e}")
            raise

    def analyze_beach_fresh(self, download_images: bool = True) -> Dict[str, Any]:
        """
        Capture fresh snapshot and analyze (no cache).
        
        Args:
            download_images: Whether to download annotated/segmented images
        
        Returns:
            Dictionary with complete analysis
        """
        try:
            response = requests.post(f"{self.base_url}/analyze/fresh", timeout=90)
            response.raise_for_status()
            result = response.json()
            
            # Optionally download images
            if download_images:
                if result.get('annotated_image_url'):
                    result['annotated_local_path'] = self.download_image(result['annotated_image_url'])
                if result.get('segmented_image_url'):
                    result['segmented_local_path'] = self.download_image(result['segmented_image_url'])
            
            logger.info(f"Fresh analysis complete: {result['summary']}")
            return result
        except requests.RequestException as e:
            logger.error(f"Fresh analysis failed: {e}")
            raise
    
    # Convenience methods for common workflows
    
    def capture_and_analyze(self) -> Dict[str, Any]:
        """
        Capture a snapshot and perform complete analysis.
        
        Returns:
            Dictionary with analysis results and image paths
            
        Raises:
            requests.RequestException: If any step fails
        """
        # Capture snapshot
        capture_result = self.capture_snapshot()
        snapshot_url = capture_result['snapshot_url']
        
        # Download snapshot
        snapshot_path = self.download_image(snapshot_url)
        
        try:
            # Analyze
            analysis_result = self.analyze_beach(snapshot_path)
            
            # Add snapshot path to result
            analysis_result['snapshot_path'] = snapshot_path
            analysis_result['snapshot_url'] = snapshot_url
            
            return analysis_result
            
        except Exception as e:
            # Clean up temp file on error
            if Path(snapshot_path).exists():
                Path(snapshot_path).unlink()
            raise
    
    def capture_and_detect(self) -> Dict[str, Any]:
        """
        Capture a snapshot and detect objects only (faster than full analysis).
        
        Returns:
            Dictionary with detection results
            
        Raises:
            requests.RequestException: If any step fails
        """
        # Capture snapshot
        capture_result = self.capture_snapshot()
        snapshot_url = capture_result['snapshot_url']
        
        # Download snapshot
        snapshot_path = self.download_image(snapshot_url)
        
        try:
            # Detect
            detection_result = self.detect_objects(snapshot_path)
            
            # Add snapshot path to result
            detection_result['snapshot_path'] = snapshot_path
            detection_result['snapshot_url'] = snapshot_url
            
            return detection_result
            
        except Exception as e:
            # Clean up temp file on error
            if Path(snapshot_path).exists():
                Path(snapshot_path).unlink()
            raise


# Singleton instance
_client: Optional[BeachMonitorAPIClient] = None


def get_api_client() -> BeachMonitorAPIClient:
    """
    Get or create the singleton API client instance.
    
    Returns:
        BeachMonitorAPIClient instance
    """
    global _client
    if _client is None:
        _client = BeachMonitorAPIClient()
    return _client


if __name__ == "__main__":
    # Test the API client
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        client = BeachMonitorAPIClient()
        
        # Health check
        print("\n=== Health Check ===")
        health = client.health_check()
        print(f"Status: {health['status']}")
        
        # Model info
        print("\n=== Model Info ===")
        info = client.get_model_info()
        print(f"Models loaded: {info.get('models_loaded', 'Unknown')}")
        
        # Capture and analyze
        print("\n=== Capture and Analyze ===")
        result = client.capture_and_analyze()
        print(f"People: {result['people_count']} (Beach: {result['beach_count']}, Water: {result['water_count']})")
        print(f"Boats: {result['boat_count']}")
        print(f"Summary: {result['summary']}")
        
        # Clean up
        if 'snapshot_path' in result and Path(result['snapshot_path']).exists():
            Path(result['snapshot_path']).unlink()
            print(f"\n✅ Cleaned up temp file")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
