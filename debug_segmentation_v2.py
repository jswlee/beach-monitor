"""
Comprehensive debug script for segmentation pipeline.
Shows what image is fed to the model and what it returns.
"""
import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from api.models.classify_regions import RegionClassifier

# Load the test image
image_path = "demo-photos/youtube_snapshot_20250902_135358.jpg"
image = cv2.imread(image_path)

print("="*80)
print("SEGMENTATION PIPELINE DEBUG")
print("="*80)
print(f"\n1. Original image shape: {image.shape}")

# Initialize classifier
classifier = RegionClassifier()

# Step 1: Apply beach polygon mask on full image
print("\n" + "="*80)
print("STEP 1: Apply beach polygon mask (from config.yaml)")
print("="*80)
masked_full_image = classifier._apply_mask(image)
print(f"After beach mask shape: {masked_full_image.shape}")
print(f"Non-zero pixels: {np.count_nonzero(masked_full_image)} / {masked_full_image.size}")
cv2.imwrite("debug_step1_beach_masked.jpg", masked_full_image)
print("Saved: debug_step1_beach_masked.jpg")

# Step 2: Crop to detection ROI
print("\n" + "="*80)
print("STEP 2: Crop to detection ROI bounding box")
print("="*80)
detection_roi_image, x_offset, y_offset = classifier._crop_to_detection_roi(masked_full_image)
print(f"After crop shape: {detection_roi_image.shape}")
print(f"Crop offsets: x={x_offset}, y={y_offset}")
print(f"Non-zero pixels: {np.count_nonzero(detection_roi_image)} / {detection_roi_image.size}")
cv2.imwrite("debug_step2_cropped.jpg", detection_roi_image)
print("Saved: debug_step2_cropped.jpg")

# Step 3: Run segmentation
print("\n" + "="*80)
print("STEP 3: Run segmentation model (resizes to 512x512 internally)")
print("="*80)
segmentation = classifier._segment_image(detection_roi_image)
print(f"Segmentation output shape: {segmentation.shape}")
print(f"Unique class IDs: {np.unique(segmentation)}")

# Count pixels per class
print("\nClass distribution:")
for class_id in np.unique(segmentation):
    count = np.sum(segmentation == class_id)
    percentage = (count / segmentation.size) * 100
    print(f"  Class {class_id}: {count:,} pixels ({percentage:.1f}%)")

# Create color-coded visualization
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

color_map = {
    0: [0, 0, 0],       # Background - Black
    1: [255, 0, 0],     # Class 1 - Blue (BGR)
    2: [0, 165, 255],   # Class 2 - Orange (BGR)
    3: [0, 255, 0],     # Class 3 - Green (BGR)
}

h, w = segmentation.shape
colored = np.zeros((h, w, 3), dtype=np.uint8)

for class_id, color in color_map.items():
    mask = segmentation == class_id
    colored[mask] = color

cv2.imwrite("debug_step3_segmentation_colored.jpg", colored)
print("Saved: debug_step3_segmentation_colored.jpg")

# Create overlay
overlay = detection_roi_image.copy()
overlay = cv2.addWeighted(overlay, 0.6, colored, 0.4, 0)
cv2.imwrite("debug_step4_overlay.jpg", overlay)
print("Saved: debug_step4_overlay.jpg")

# Now simulate what happens with person boxes
print("\n" + "="*80)
print("SIMULATING PERSON CLASSIFICATION")
print("="*80)

# Create a few test points across the image
test_points = [
    (detection_roi_image.shape[1] // 4, detection_roi_image.shape[0] // 2),  # Left side
    (detection_roi_image.shape[1] // 2, detection_roi_image.shape[0] // 2),  # Center
    (3 * detection_roi_image.shape[1] // 4, detection_roi_image.shape[0] // 2),  # Right side
    (detection_roi_image.shape[1] // 2, detection_roi_image.shape[0] // 4),  # Top
    (detection_roi_image.shape[1] // 2, 3 * detection_roi_image.shape[0] // 4),  # Bottom
]

print("\nSampling segmentation at test points:")
for i, (x, y) in enumerate(test_points):
    x_clamped = max(0, min(x, segmentation.shape[1] - 1))
    y_clamped = max(0, min(y, segmentation.shape[0] - 1))
    class_id = int(segmentation[y_clamped, x_clamped])
    print(f"  Point {i+1} at ({x}, {y}): class_id = {class_id}")

# Check if the cropped image actually has any beach/water content
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

if np.count_nonzero(detection_roi_image) == 0:
    print("⚠️  WARNING: Cropped image is completely black!")
    print("   The beach mask may have zeroed out the entire ROI.")
elif len(np.unique(segmentation)) == 1:
    print("⚠️  WARNING: Segmentation returned only one class!")
    print("   The model may not be seeing any beach/water features.")
    print("\n   Possible causes:")
    print("   1. The beach mask polygon doesn't overlap with the detection ROI")
    print("   2. The model expects different preprocessing")
    print("   3. The cropped region is too different from training data")
else:
    print("✓ Segmentation returned multiple classes")
    print(f"  Classes found: {sorted(np.unique(segmentation))}")

print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  - debug_step1_beach_masked.jpg (full image with beach mask)")
print("  - debug_step2_cropped.jpg (cropped to detection ROI)")
print("  - debug_step3_segmentation_colored.jpg (color-coded segmentation)")
print("  - debug_step4_overlay.jpg (segmentation overlaid on cropped image)")
