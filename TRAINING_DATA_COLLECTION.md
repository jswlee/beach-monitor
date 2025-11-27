# Automatic Training Data Collection

The beach monitoring system now automatically collects training data during inference for model fine-tuning.

## Overview

Every time the API analyzes a beach image, it saves:
1. **Object Detection Data** - Bounding boxes for people and boats
2. **Segmentation Data** - Location classifications (beach/water/other)

Both datasets are saved in YOLO format, ready for training.

## Directory Structure

```
training_data/
├── README.md                          # Detailed format documentation
├── train_object_detection/
│   ├── images/                        # Original beach images
│   │   ├── youtube_snapshot_20251115_085601.jpg
│   │   ├── youtube_snapshot_20251115_090123.jpg
│   │   └── ...
│   └── labels/                        # YOLO bounding box labels
│       ├── youtube_snapshot_20251115_085601.txt
│       ├── youtube_snapshot_20251115_090123.txt
│       └── ...
└── train_segmentation/
    ├── images/                        # Original beach images (same as above)
    │   ├── youtube_snapshot_20251115_085601.jpg
    │   ├── youtube_snapshot_20251115_090123.jpg
    │   └── ...
    └── labels/                        # YOLO segmentation labels
        ├── youtube_snapshot_20251115_085601.txt
        ├── youtube_snapshot_20251115_090123.txt
        └── ...
```

## How It Works

### 1. During Inference

When `BeachAnalyzer.analyze_beach_activity()` runs:

```python
# In api/models/analyze_beach.py
analyzer = BeachAnalyzer(save_training_data=True)  # Default: True
result = analyzer.analyze_beach_activity(image_path)
```

The system automatically:
1. Runs object detection → saves bounding boxes to `train_object_detection/`
2. Runs segmentation → saves location labels to `train_segmentation/`
3. Copies the original image to both `images/` folders
4. Creates matching `.txt` label files

### 2. Label Format

**Object Detection (`train_object_detection/labels/*.txt`):**
```
1 0.523456 0.678901 0.045678 0.123456  # person
1 0.234567 0.456789 0.067890 0.089012  # person
0 0.789012 0.345678 0.123456 0.234567  # boat
```
Format: `class_id x_center y_center width height` (all normalized 0-1)

**Segmentation (`train_segmentation/labels/*.txt`):**
```
0 0.123 0.456 0.234 0.445 0.345 0.467 0.456 0.489 0.567 0.512  # beach polygon
1 0.789 0.234 0.890 0.245 0.901 0.256 0.912 0.267 0.823 0.278  # water polygon
0 0.012 0.789 0.023 0.790 0.034 0.801 0.045 0.812              # another beach region
```
Format: `class_id x1 y1 x2 y2 ... xn yn` (polygon points from segmentation mask, normalized 0-1)
- Variable number of points per polygon
- Multiple polygons per class possible
- Extracted from SegFormer waterline segmentation model

## Configuration

### Enable/Disable Collection

```python
# Enable (default)
analyzer = BeachAnalyzer(save_training_data=True)

# Disable
analyzer = BeachAnalyzer(save_training_data=False)
```

### Custom Base Directory

```python
from api.models.save_training_data import get_training_data_saver

# Default: "training_data/"
saver = get_training_data_saver(base_dir="custom_training_data/")
```

## Checking Dataset Statistics

### Using Python

```python
from api.models.save_training_data import get_training_data_saver

saver = get_training_data_saver()
stats = saver.get_dataset_stats()

print(stats)
# {
#     'detection': {'images': 150, 'labels': 150},
#     'segmentation': {'images': 150, 'labels': 150}
# }
```

### Using CLI Script

```bash
python check_training_data.py
```

Output:
```
============================================================
Beach Monitor - Training Data Statistics
============================================================

📊 Dataset Statistics:
------------------------------------------------------------

🎯 Object Detection (Bounding Boxes):
   Images: 150
   Labels: 150
   Path:   training_data/train_object_detection

🗺️  Segmentation (Location Classification):
   Images: 150
   Labels: 150
   Path:   training_data/train_segmentation

✅ All datasets have matching image/label counts

📝 Sample Labels:
------------------------------------------------------------
...
```

## Using for Model Training

### Object Detection

```python
from ultralytics import YOLO

# Create dataset.yaml
dataset_config = """
path: training_data/train_object_detection
train: images
val: images  # Split manually for better validation

nc: 2
names: ['boat', 'person']
"""

# Train
model = YOLO('models_cache/beach_detection_100e30p20we1e-5lr10f.pt')  # Your existing model
results = model.train(
    data='dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='beach_detection_finetuned'
)
```

### Segmentation

```python
from ultralytics import YOLO

# Create dataset.yaml
dataset_config = """
path: training_data/train_segmentation
train: images
val: images  # Split manually for better validation

nc: 2
names: ['beach', 'water']
"""

# Train
model = YOLO('yolov8n-seg.pt')  # Or your existing segmentation model
results = model.train(
    data='dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='waterline_segmentation_finetuned'
)
```

## Best Practices

### 1. Review Labels Before Training
- Model predictions may have errors
- Use tools like [LabelImg](https://github.com/heartexlabs/labelImg) or [CVAT](https://github.com/opencv/cvat)
- Correct obvious mistakes

### 2. Balance Your Dataset
Ensure good representation of:
- ✅ Different times of day (morning, noon, afternoon, evening)
- ✅ Various crowd densities (empty, quiet, moderate, busy)
- ✅ Different weather conditions (sunny, cloudy, rainy)
- ✅ Different camera angles/positions

### 3. Split Train/Val Properly
```python
import shutil
from pathlib import Path
import random

def split_dataset(base_dir, val_ratio=0.2):
    """Split images into train and val sets."""
    images_dir = Path(base_dir) / "images"
    labels_dir = Path(base_dir) / "labels"
    
    # Create val directories
    val_images = Path(base_dir) / "val" / "images"
    val_labels = Path(base_dir) / "val" / "labels"
    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    all_images = list(images_dir.glob("*.jpg"))
    random.shuffle(all_images)
    
    # Split
    val_count = int(len(all_images) * val_ratio)
    val_images_list = all_images[:val_count]
    
    # Move to val
    for img_path in val_images_list:
        label_path = labels_dir / f"{img_path.stem}.txt"
        shutil.move(str(img_path), str(val_images / img_path.name))
        shutil.move(str(label_path), str(val_labels / label_path.name))
    
    print(f"Split complete: {len(all_images) - val_count} train, {val_count} val")

# Usage
split_dataset("training_data/train_object_detection", val_ratio=0.2)
split_dataset("training_data/train_segmentation", val_ratio=0.2)
```

### 4. Remove Duplicates
```python
import imagehash
from PIL import Image
from pathlib import Path

def find_duplicates(images_dir, threshold=5):
    """Find near-duplicate images using perceptual hashing."""
    hashes = {}
    duplicates = []
    
    for img_path in Path(images_dir).glob("*.jpg"):
        img_hash = imagehash.average_hash(Image.open(img_path))
        
        # Check for similar hashes
        for existing_hash, existing_path in hashes.items():
            if abs(img_hash - existing_hash) < threshold:
                duplicates.append((img_path, existing_path))
                break
        else:
            hashes[img_hash] = img_path
    
    return duplicates

# Usage
dupes = find_duplicates("training_data/train_object_detection/images")
print(f"Found {len(dupes)} duplicate pairs")
```

### 5. Version Control
- Keep track of dataset versions
- Tag model checkpoints with dataset version
- Document changes and improvements

## Implementation Details

### Files Modified
- `api/models/save_training_data.py` - New module for saving training data
- `api/models/analyze_beach.py` - Integrated training data saving
- `api/models/__init__.py` - Added exports
- `.gitignore` - Excluded training data from git

### Classes
- `TrainingDataSaver` - Main class for saving YOLO format data
- `get_training_data_saver()` - Singleton getter

### Methods
- `save_detection_data()` - Save object detection labels
- `save_segmentation_data()` - Save segmentation labels
- `get_dataset_stats()` - Get dataset statistics

## Troubleshooting

### Images saved but no labels
- Check if detections were successful
- Verify `save_training_data=True` in analyzer
- Check logs for errors

### Label format errors
- Ensure coordinates are normalized (0-1)
- Verify class IDs match your model
- Check for NaN or invalid values

### Disk space issues
- Training data can grow large over time
- Periodically archive old data
- Consider cloud storage for large datasets

## Future Enhancements

Potential improvements:
- [ ] Active learning - prioritize uncertain predictions
- [ ] Automatic label correction using ensemble models
- [ ] Data augmentation pipeline
- [ ] Cloud sync for distributed collection
- [ ] Automatic train/val splitting
- [ ] Dataset versioning and tracking
