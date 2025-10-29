"""
Helper module to select backup photos from demo-photos based on time of day.
"""
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def parse_photo_timestamp(filename: str) -> Optional[datetime]:
    """
    Parse timestamp from demo photo filename.
    
    Expected format: youtube_snapshot_YYYYMMDD_HHMMSS.jpg
    Example: youtube_snapshot_20250902_130003.jpg
    
    Args:
        filename: Photo filename
        
    Returns:
        datetime object or None if parsing fails
    """
    pattern = r'youtube_snapshot_(\d{8})_(\d{6})\.jpg'
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    date_str = match.group(1)  # YYYYMMDD
    time_str = match.group(2)  # HHMMSS
    
    try:
        dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        return dt
    except ValueError:
        return None


def get_time_difference_minutes(time1: datetime, time2: datetime) -> int:
    """
    Calculate time difference in minutes, considering only time of day (not date).
    
    Args:
        time1: First datetime
        time2: Second datetime
        
    Returns:
        Absolute difference in minutes
    """
    # Extract time components
    t1_minutes = time1.hour * 60 + time1.minute
    t2_minutes = time2.hour * 60 + time2.minute
    
    # Calculate difference
    diff = abs(t1_minutes - t2_minutes)
    
    # Handle wrap-around (e.g., 23:00 vs 01:00 should be 2 hours, not 22 hours)
    if diff > 720:  # More than 12 hours
        diff = 1440 - diff  # 1440 minutes in a day
    
    return diff


def select_closest_photo(demo_photos_dir: str, target_time: datetime) -> Optional[str]:
    """
    Select the photo from demo-photos with time of day closest to target time.
    
    Args:
        demo_photos_dir: Path to demo-photos directory
        target_time: Target datetime (typically current time in HST)
        
    Returns:
        Full path to selected photo, or None if no valid photos found
    """
    demo_path = Path(demo_photos_dir)
    
    if not demo_path.exists():
        logger.error(f"Demo photos directory not found: {demo_photos_dir}")
        return None
    
    # Get all photo files
    photo_files = [f for f in os.listdir(demo_path) if f.endswith('.jpg')]
    
    if not photo_files:
        logger.error(f"No photos found in {demo_photos_dir}")
        return None
    
    # Parse timestamps and find closest match
    best_photo = None
    min_diff = float('inf')
    
    for filename in photo_files:
        photo_time = parse_photo_timestamp(filename)
        
        if photo_time is None:
            logger.debug(f"Could not parse timestamp from {filename}")
            continue
        
        diff = get_time_difference_minutes(photo_time, target_time)
        
        if diff < min_diff:
            min_diff = diff
            best_photo = filename
    
    if best_photo:
        photo_path = str(demo_path / best_photo)
        logger.info(f"Selected backup photo: {best_photo} (time diff: {min_diff} minutes)")
        return photo_path
    else:
        logger.error("No valid photos found with parseable timestamps")
        return None


if __name__ == "__main__":
    # Test the selector
    import pytz
    
    # Get current time in HST
    hst = pytz.timezone('Pacific/Honolulu')
    current_time = datetime.now(hst)
    
    print(f"Current time (HST): {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test with demo-photos directory
    demo_dir = Path(__file__).parent.parent / "demo-photos"
    selected = select_closest_photo(str(demo_dir), current_time)
    
    if selected:
        print(f"Selected photo: {selected}")
    else:
        print("No photo selected")
