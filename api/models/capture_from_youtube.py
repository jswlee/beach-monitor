"""
Simplified YouTube livestream capture for single snapshots
"""
import os
import logging
from datetime import datetime
import cv2
import random
import time

logger = logging.getLogger(__name__)

class YouTubeCapture:
    """Simplified YouTube capture for single snapshot functionality"""
    
    def __init__(self, url: str, output_dir: str = None, interval: int = 5):
        """Initialize YouTube capture
        
        Args:
            url: YouTube livestream URL
            output_dir: Directory to save snapshots
            interval: Time interval between captures (unused in single capture mode)
        """
        self.url = url
        self.output_dir = output_dir or "images/youtube_snapshots"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def get_stream_url(self) -> str:
        """Get direct stream URL using yt-dlp
        
        Based on logs, this livestream consistently provides HLS m3u8 formats with avc1 codec.
        We simply pick the highest resolution available.
        """
        try:
            import yt_dlp
            logger.info(f"Getting stream URL for: {self.url}")
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'format': 'best',
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=False)
                formats = info.get("formats") or []
                
                if not formats:
                    # Fallback to yt-dlp's best URL if no formats list
                    fallback_url = info.get("url")
                    if fallback_url:
                        logger.info("No formats list, using yt-dlp best URL")
                        return fallback_url
                    raise ValueError("No formats available")
                
                # Filter to playable video formats (vcodec exists and not 'none')
                playable = [f for f in formats if f.get("vcodec") and f.get("vcodec") != "none"]
                
                if not playable:
                    raise ValueError("No playable video formats found")
                
                # Pick the format with highest resolution (height)
                # Livestream consistently provides HLS m3u8 with heights: 144, 240, 360, 480, 720, 1080
                chosen = max(playable, key=lambda f: f.get("height") or 0)
                
                chosen_url = chosen.get("url")
                if not chosen_url:
                    raise ValueError("Chosen format has no URL")
                
                logger.info(
                    f"Selected format id={chosen.get('format_id')} height={chosen.get('height')}p "
                    f"proto={chosen.get('protocol')} vcodec={chosen.get('vcodec')}"
                )
                
                return chosen_url
                
        except Exception as e:
            logger.error(f"Error getting stream URL: {e}")
            return None
    
    def capture_snapshot(self, max_retries: int = 3) -> str:
        """Capture a single snapshot from the livestream
        
        Args:
            max_retries: Maximum number of retry attempts
            
        Returns:
            Path to saved image file, or None if failed
        """
        attempt = 0
        
        def _retry_backoff_and_continue(reason: str):
            nonlocal attempt
            attempt += 1
            if attempt < max_retries:
                backoff = (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"{reason} (attempt {attempt}/{max_retries}). Retrying in {backoff:.1f} seconds...")
                time.sleep(backoff)
                return True
            return False
        
        while attempt < max_retries:
            try:
                stream_url = self.get_stream_url()
                if not stream_url:
                    if _retry_backoff_and_continue("Failed to get stream URL"):
                        continue
                    logger.error("Failed to get stream URL after retries")
                    break
                
                cap = cv2.VideoCapture(stream_url)
                if not cap.isOpened():
                    cap.release()
                    if _retry_backoff_and_continue("Failed to open video stream"):
                        continue
                    logger.error("Failed to open video stream after retries")
                    break
                
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    if _retry_backoff_and_continue("Failed to read frame"):
                        continue
                    logger.error("Failed to read frame after retries")
                    break
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"youtube_snapshot_{timestamp}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                
                cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                
                file_size = os.path.getsize(filepath) / (1024 * 1024)
                logger.info(f"Saved: {filename} | Size: {file_size:.1f}MB")
                
                return filepath
                
            except Exception as e:
                logger.error(f"Error capturing snapshot: {e}")
                if _retry_backoff_and_continue("Transient error during capture"):
                    continue
                logger.error("Giving up after retries due to repeated errors")
                
        logger.error(f"Failed to capture snapshot after {max_retries} attempts")
        return None
