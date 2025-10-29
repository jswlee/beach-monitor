"""
Beach snapshot capture module - simplified for MVP
"""
import logging
import yaml
from datetime import datetime
from pathlib import Path
import pytz

from beach_cv_tool.livestream_capture_youtube import YouTubeCapture
from beach_cv_tool.extract_url import get_youtube_livestream_url, validate_video_title
from beach_cv_tool.backup_photo_selector import select_closest_photo

logger = logging.getLogger(__name__)

class BeachCapture:
    """Simplified beach snapshot capture for MVP"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize capture with configuration"""
        self.config = self._load_config(config_path)
        self.camera_config = self.config['camera']
        self.paths_config = self.config['paths']
        self.config_path_obj = Path(config_path).parent if config_path != "config.yaml" else Path.cwd()
        
        # Try to get YouTube livestream URL and validate title
        self.youtube_capture = None
        self.use_backup = False
        
        stream_url, video_title = get_youtube_livestream_url(self.camera_config['stream_url'])
        
        # Check if video title contains all search query words
        if stream_url and video_title and validate_video_title(video_title, self.camera_config['stream_url']):
            logger.info(f"Valid livestream found: {video_title}")
            self.youtube_capture = YouTubeCapture(
                url=stream_url,
                output_dir=self.paths_config['snapshots'],
                interval=self.camera_config['capture_interval']
            )
        else:
            logger.warning(f"Livestream not available or title mismatch. Will use backup photos.")
            self.use_backup = True
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            # Store the directory of the config file
            self.config_dir = Path(config_path).parent.resolve()
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def capture_snapshot(self) -> str:
        """
        Capture a single snapshot from the beach livestream or select backup photo.
        
        Returns:
            str: Path to the captured/selected image file
        """
        try:
            # If using backup photos, select one based on current time
            if self.use_backup:
                logger.info("Using backup photo based on time of day...")
                return self._get_backup_photo()
            
            # Try to capture from livestream
            logger.info("Capturing beach snapshot from livestream...")
            
            try:
                snapshot_path = self.youtube_capture.capture_snapshot(
                    max_retries=self.camera_config['max_retries']
                )
                
                if snapshot_path:
                    logger.info(f"Successfully captured: {snapshot_path}")
                    return snapshot_path
                else:
                    # Fallback to backup photo
                    logger.warning("Livestream capture failed, falling back to backup photo")
                    return self._get_backup_photo()
                    
            except Exception as e:
                logger.error(f"Error capturing from livestream: {e}. Using backup photo.")
                return self._get_backup_photo()
                
        except Exception as e:
            logger.error(f"Error in capture_snapshot: {e}")
            raise
    
    def _get_backup_photo(self) -> str:
        """
        Select a backup photo from demo-photos based on current HST time.
        
        Returns:
            str: Path to selected backup photo
        """
        # Get current time in HST
        hst = pytz.timezone('Pacific/Honolulu')
        current_time = datetime.now(hst)
        
        # Demo photos directory
        demo_dir = self.config_path_obj / "demo-photos"
        
        selected_photo = select_closest_photo(str(demo_dir), current_time)
        
        if selected_photo:
            logger.info(f"Selected backup photo: {selected_photo}")
            return selected_photo
        else:
            raise Exception("No backup photos available")
