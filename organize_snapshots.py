"""
Script to organize snapshot images.
Moves all images that are NOT raw snapshots (e.g. annotated, segmented) into a subfolder.
"""
import os
import re
import shutil
from pathlib import Path

def organize_snapshots():
    # Source directory
    snapshots_dir = Path("data/snapshots")
    target_dir = snapshots_dir / "annotated"
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(exist_ok=True)
    
    # Regex for raw snapshot: youtube_snapshot_YYYYMMDD_HHMMSS.jpg
    # Strict start (^) and end ($) anchors
    raw_pattern = re.compile(r'^youtube_snapshot_\d{8}_\d{6}\.(jpg|jpeg)$', re.IGNORECASE)
    
    # Get all files
    files = [f for f in snapshots_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg']]
    
    print(f"Found {len(files)} image files in {snapshots_dir}")
    
    moved_count = 0
    for file in files:
        if not raw_pattern.match(file.name):
            # This file does NOT match the raw pattern -> Move it
            dest = target_dir / file.name
            try:
                shutil.move(str(file), str(dest))
                # print(f"Moved: {file.name}")
                moved_count += 1
            except Exception as e:
                print(f"Error moving {file.name}: {e}")
                
    print(f"\nDone! Moved {moved_count} files to {target_dir}")
    print(f"Remaining files in {snapshots_dir}: {len(list(snapshots_dir.glob('*.jpg')))}")

if __name__ == "__main__":
    organize_snapshots()
