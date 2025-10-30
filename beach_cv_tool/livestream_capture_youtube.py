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
        """Get direct stream URL using yt-dlp"""
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
                fallback_url = info.get("url")
                
                formats = info.get("formats") or []
                if not formats:
                    if fallback_url:
                        logger.info("Using fallback best URL from yt-dlp")
                        return fallback_url
                    raise ValueError("No formats available")
                
                def is_h264(fmt):
                    v = (fmt.get("vcodec") or "").lower()
                    return ("avc" in v) or ("h264" in v)
                
                def is_hls(fmt):
                    p = (fmt.get("protocol") or "").lower()
                    ext = (fmt.get("ext") or "").lower()
                    return ("m3u8" in p) or (ext == "m3u8")
                
                def playable(fmt):
                    vcodec = fmt.get("vcodec")
                    return vcodec and vcodec != "none"
                
                candidates = [f for f in formats if playable(f)]
                h264_candidates = [f for f in candidates if is_h264(f)]
                
                def key(fmt):
                    height = fmt.get("height") or 0
                    fps = fmt.get("fps") or 0
                    hls_boost = 1 if is_hls(fmt) else 0
                    return (height, fps, hls_boost)
                
                chosen = None
                if h264_candidates:
                    chosen = sorted(h264_candidates, key=key, reverse=True)[0]
                elif candidates:
                    chosen = sorted(candidates, key=key, reverse=True)[0]
                
                if chosen and chosen.get("url"):
                    return chosen.get("url")
                
                if fallback_url:
                    logger.info("Falling back to yt-dlp best URL")
                    return fallback_url
                
                raise ValueError("No suitable stream URL found")
                
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
        while attempt < max_retries:
            try:
                stream_url = self.get_stream_url()
                if not stream_url:
                    logger.error(f"Failed to get stream URL (attempt {attempt+1}/{max_retries})")
                    attempt += 1
                    if attempt < max_retries:
                        backoff = (2 ** attempt) + random.uniform(0, 1)
                        logger.info(f"Retrying in {backoff:.1f} seconds...")
                        time.sleep(backoff)
                    continue
                
                cap = cv2.VideoCapture(stream_url)
                if not cap.isOpened():
                    logger.error(f"Failed to open video stream (attempt {attempt+1}/{max_retries})")
                    cap.release()
                    attempt += 1
                    if attempt < max_retries:
                        backoff = (2 ** attempt) + random.uniform(0, 1)
                        logger.info(f"Retrying in {backoff:.1f} seconds...")
                        time.sleep(backoff)
                    continue
                
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    logger.error(f"Failed to read frame (attempt {attempt+1}/{max_retries})")
                    attempt += 1
                    if attempt < max_retries:
                        backoff = (2 ** attempt) + random.uniform(0, 1)
                        logger.info(f"Retrying in {backoff:.1f} seconds...")
                        time.sleep(backoff)
                    continue
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"youtube_snapshot_{timestamp}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                
                cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                
                file_size = os.path.getsize(filepath) / (1024 * 1024)
                logger.info(f"Saved: {filename} | Size: {file_size:.1f}MB")
                
                return filepath
                
            except Exception as e:
                logger.error(f"Error capturing snapshot (attempt {attempt+1}/{max_retries}): {e}")
                attempt += 1
                if attempt < max_retries:
                    backoff = (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retrying in {backoff:.1f} seconds...")
                    time.sleep(backoff)
        
        logger.error(f"Failed to capture snapshot after {max_retries} attempts")
        return None
