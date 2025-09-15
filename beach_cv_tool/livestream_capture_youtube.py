#!/usr/bin/env python3
"""
YouTube Livestream Capture for macOS
Captures snapshots from a YouTube livestream at specified intervals
"""

import os
import argparse
import time
import logging
from datetime import datetime
import cv2
from urllib.parse import urlparse
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("youtube_capture.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class YouTubeCapture:
    def __init__(self, url, output_dir=None, interval=5, max_runtime=None):
        """Initialize YouTube capture with a single URL"""
        self.url = url
        
        # Set default output directory based on URL domain
        if output_dir is None:
            domain = urlparse(url).netloc
            if domain == "youtu.be" or domain == "www.youtube.com" or domain == "youtube.com":
                video_id = self._get_video_id(url)
                output_dir = f"images/youtube_{video_id}"
            else:
                output_dir = f"images/{domain}"
                
        self.output_dir = output_dir
        self.interval = interval
        self.max_runtime = max_runtime
        self.cap = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def _get_video_id(self, url):
        """Extract video ID from YouTube URL"""
        if "youtu.be" in url:
            return url.split("/")[-1]
        elif "youtube.com" in url:
            if "v=" in url:
                return url.split("v=")[1].split("&")[0]
            else:
                return url.split("/")[-1]
        return "unknown"
        
    def get_stream_url(self):
        """Get the direct stream URL using yt-dlp Python API, picking highest-res playable stream.

        Strategy:
        - Use yt-dlp Python API instead of subprocess calls
        - Prefer H.264/AVC ("avc"/"h264") video codecs when available
        - Among candidates, pick the highest height
        - Prefer HLS (m3u8) protocols which OpenCV can read
        - Fallback to yt-dlp's best URL
        """
        try:
            import yt_dlp
            logger.info(f"Getting stream URL for: {self.url}")
            
            # Configure yt-dlp options
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'format': 'best',  # Default format
            }
            
            # Use a context manager to ensure proper cleanup
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info
                info = ydl.extract_info(self.url, download=False)
                
                # If yt-dlp already gave a direct URL (best), keep it as fallback
                fallback_url = info.get("url")
                
                formats = info.get("formats") or []
                if not formats:
                    if fallback_url:
                        logger.info("No formats list; using fallback best URL from yt-dlp")
                        return fallback_url
                    raise ValueError("No formats available in yt-dlp output")
                
                def is_h264(fmt):
                    v = (fmt.get("vcodec") or "").lower()
                    return ("avc" in v) or ("h264" in v)
                
                def is_hls(fmt):
                    p = (fmt.get("protocol") or "").lower()
                    ext = (fmt.get("ext") or "").lower()
                    return ("m3u8" in p) or (ext == "m3u8")
                
                def playable(fmt):
                    # Must have video
                    vcodec = fmt.get("vcodec")
                    return vcodec and vcodec != "none"
                
                # Build candidate lists
                candidates = [f for f in formats if playable(f)]
                h264_candidates = [f for f in candidates if is_h264(f)]
                
                def key(fmt):
                    # Sort by height desc, fps desc, hls preferred
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
                    logger.info(
                        "Chosen format: %sx%s @ %sfps | vcodec=%s acodec=%s prot=%s ext=%s",
                        chosen.get("width"), chosen.get("height"), chosen.get("fps"),
                        chosen.get("vcodec"), chosen.get("acodec"), chosen.get("protocol"), chosen.get("ext")
                    )
                    return chosen.get("url")
                
                if fallback_url:
                    logger.info("Falling back to yt-dlp best URL")
                    return fallback_url
                
                raise ValueError("No suitable stream URL found")
                
        except Exception as e:
            logger.error(f"Error getting stream URL: {e}")
            return None
    
    def setup_capture(self):
        """Setup video capture from YouTube stream"""
        try:
            stream_url = self.get_stream_url()
            if not stream_url:
                raise ValueError("Failed to get stream URL")
            
            # Open video capture
            self.cap = cv2.VideoCapture(stream_url)
            
            if not self.cap.isOpened():
                raise ValueError("Failed to open video stream")
                
            # Get video properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"Stream opened: {width}x{height} at {fps:.1f} fps")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup capture: {e}")
            return False
            
    def capture_snapshot(self, max_retries=3):
        """Capture a snapshot from the stream by opening a fresh connection each time
        
        Args:
            max_retries: Maximum number of retry attempts if capture fails
        
        Returns:
            str: Path to the saved image file, or None if capture failed
        """
        attempt = 0
        while attempt < max_retries:
            try:
                # Get a fresh stream URL each time
                stream_url = self.get_stream_url()
                if not stream_url:
                    logger.error(f"Failed to get stream URL (attempt {attempt+1}/{max_retries})")
                    attempt += 1
                    if attempt < max_retries:
                        # Add exponential backoff with jitter
                        backoff = (2 ** attempt) + random.uniform(0, 1)
                        logger.info(f"Retrying in {backoff:.1f} seconds...")
                        time.sleep(backoff)
                    continue
                
                # Open a new capture for this snapshot
                cap = cv2.VideoCapture(stream_url)
                if not cap.isOpened():
                    logger.error(f"Failed to open video stream (attempt {attempt+1}/{max_retries})")
                    cap.release()  # Ensure release even if not opened
                    attempt += 1
                    if attempt < max_retries:
                        # Add exponential backoff with jitter
                        backoff = (2 ** attempt) + random.uniform(0, 1)
                        logger.info(f"Retrying in {backoff:.1f} seconds...")
                        time.sleep(backoff)
                    continue
                
                # Read frame
                ret, frame = cap.read()
                
                # Always close the capture immediately
                cap.release()
                
                if not ret:
                    logger.error(f"Failed to read frame (attempt {attempt+1}/{max_retries})")
                    attempt += 1
                    if attempt < max_retries:
                        # Add exponential backoff with jitter
                        backoff = (2 ** attempt) + random.uniform(0, 1)
                        logger.info(f"Retrying in {backoff:.1f} seconds...")
                        time.sleep(backoff)
                    continue
                
                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"youtube_snapshot_{timestamp}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                
                # Save with higher JPEG quality
                cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                
                # Get file size for logging
                file_size = os.path.getsize(filepath) / (1024 * 1024)  # Size in MB
                logger.info(f"Saved: {filename} | Size: {file_size:.1f}MB")
                
                return filepath
                
            except Exception as e:
                logger.error(f"Error capturing snapshot (attempt {attempt+1}/{max_retries}): {e}")
                attempt += 1
                if attempt < max_retries:
                    # Add exponential backoff with jitter
                    backoff = (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retrying in {backoff:.1f} seconds...")
                    time.sleep(backoff)
        
        logger.error(f"Failed to capture snapshot after {max_retries} attempts")
        return None
        
    def run(self):
        """Run the capture process with error recovery for long-running sessions"""
        start_time = time.time()
        capture_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        try:
            # Verify we can get a stream URL before starting
            if not self.get_stream_url():
                logger.error("Failed to get initial stream URL. Exiting.")
                return
                
            # Show info
            logger.info(f"Capturing every {self.interval} seconds")
            if self.max_runtime:
                logger.info(f"Will run for {self.max_runtime} seconds")
            else:
                logger.info("Press Ctrl+C to stop")
            logger.info(f"Saving to: {os.path.abspath(self.output_dir)}")
            logger.info("Using fresh connection for each capture to prevent freezing")
            logger.info("Added error recovery with automatic retries")
            
            # Main loop
            while True:
                # Check max runtime
                if self.max_runtime and (time.time() - start_time) >= self.max_runtime:
                    logger.info(f"Maximum runtime reached. Captured {capture_count} images.")
                    break
                
                try:    
                    # Take snapshot with fresh connection each time (with retries)
                    if self.capture_snapshot():
                        capture_count += 1
                        consecutive_failures = 0  # Reset failure counter on success
                        
                        if self.max_runtime:
                            remaining = max(0, self.max_runtime - (time.time() - start_time))
                            logger.info(f"Captures: {capture_count} | Remaining: {remaining:.0f}s")
                    else:
                        consecutive_failures += 1
                        logger.warning(f"Snapshot capture failed. Consecutive failures: {consecutive_failures}")
                        
                        # If too many consecutive failures, try a longer cooldown period
                        if consecutive_failures >= max_consecutive_failures:
                            cooldown = 30 + random.uniform(0, 10)  # 30-40 second cooldown
                            logger.warning(f"Too many consecutive failures. Cooling down for {cooldown:.1f}s before retrying")
                            time.sleep(cooldown)
                            consecutive_failures = 0  # Reset after cooldown
                            
                            # Force Python garbage collection to help clean up resources
                            import gc
                            gc.collect()
                            
                except Exception as e:
                    consecutive_failures += 1
                    logger.error(f"Error in capture cycle: {e}")
                    logger.warning(f"Continuing to next cycle. Consecutive failures: {consecutive_failures}")
                
                # Wait for next interval
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            elapsed = time.time() - start_time
            logger.info(f"Stopped after {elapsed:.0f}s with {capture_count} captures")
        except Exception as e:
            logger.error(f"Critical error in main loop: {e}")
        finally:
            if self.cap:
                self.cap.release()
                logger.info("Video capture released")


def main():
    """Parse arguments and start capture"""
    parser = argparse.ArgumentParser(description="YouTube livestream snapshot capture")
    parser.add_argument(
        "--url", type=str,
        default="https://www.youtube.com/watch?v=DNnj_9bVWGI",
        help="URL of the YouTube livestream to capture"
    )
    parser.add_argument(
        "--interval", type=float, default=5,
        help="Seconds between snapshots (default: 5)"
    )
    parser.add_argument(
        "--max-runtime", type=int, default=30,
        help="Total runtime in seconds; 0 = run until Ctrl+C (default: 30)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save images (default: images/youtube_{video_id})"
    )
    args = parser.parse_args()

    # Handle max_runtime=0 meaning run indefinitely
    max_runtime = args.max_runtime if args.max_runtime > 0 else None

    logger.info(f"Starting YouTube capture for {args.url}")
    
    try:
        # Create and run the capture
        capture = YouTubeCapture(
            url=args.url,
            output_dir=args.output_dir,
            interval=args.interval,
            max_runtime=max_runtime
        )
        capture.run()
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
