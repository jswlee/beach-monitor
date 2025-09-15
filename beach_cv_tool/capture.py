"""
Beach snapshot capture module - simplified for MVP
"""
import logging
import yaml

from beach_cv_tool.livestream_capture_youtube import YouTubeCapture
from beach_cv_tool.extract_url import get_youtube_livestream_url

logger = logging.getLogger(__name__)

class BeachCapture:
    """Simplified beach snapshot capture for MVP"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize capture with configuration"""
        self.config = self._load_config(config_path)
        self.camera_config = self.config['camera']
        self.paths_config = self.config['paths']
        
        # Initialize YouTube capture
        self.youtube_capture = YouTubeCapture(
            url=get_youtube_livestream_url(self.camera_config['stream_url']),
            output_dir=self.paths_config['snapshots'],
            interval=self.camera_config['capture_interval']
        )
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def capture_snapshot(self) -> str:
        """
        Capture a single snapshot from the beach livestream
        
        Returns:
            str: Path to the captured image file
        """
        try:
            logger.info("Capturing beach snapshot...")
            
            snapshot_path = self.youtube_capture.capture_snapshot(
                max_retries=self.camera_config['max_retries']
            )
            
            if snapshot_path:
                logger.info(f"Successfully captured: {snapshot_path}")
                return snapshot_path
            else:
                raise Exception("Failed to capture snapshot")
                
        except Exception as e:
            logger.error(f"Error capturing snapshot: {e}")
            raise
