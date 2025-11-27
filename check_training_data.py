"""
Utility script to check training data statistics and validate YOLO format.
"""
import sys
from pathlib import Path

# Direct import without going through api.models.__init__
# This avoids loading transformers and other heavy dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "save_training_data",
    Path(__file__).parent / "api" / "models" / "save_training_data.py"
)
save_training_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(save_training_data)

TrainingDataSaver = save_training_data.TrainingDataSaver


def main():
    """Display training data statistics."""
    print("=" * 60)
    print("Beach Monitor - Training Data Statistics")
    print("=" * 60)
    
    saver = TrainingDataSaver()
    stats = saver.get_dataset_stats()
    
    print("\n📊 Dataset Statistics:")
    print("-" * 60)
    
    print("\n🎯 Object Detection (Bounding Boxes):")
    print(f"   Images: {stats['detection']['images']}")
    print(f"   Labels: {stats['detection']['labels']}")
    print(f"   Path:   {saver.detection_dir}")
    
    print("\n🗺️  Segmentation (Location Classification):")
    print(f"   Images: {stats['segmentation']['images']}")
    print(f"   Labels: {stats['segmentation']['labels']}")
    print(f"   Path:   {saver.segmentation_dir}")
    
    # Check for mismatches
    detection_mismatch = stats['detection']['images'] != stats['detection']['labels']
    segmentation_mismatch = stats['segmentation']['images'] != stats['segmentation']['labels']
    
    if detection_mismatch or segmentation_mismatch:
        print("\n⚠️  Warnings:")
        if detection_mismatch:
            print(f"   Detection: Image/label count mismatch!")
        if segmentation_mismatch:
            print(f"   Segmentation: Image/label count mismatch!")
    else:
        print("\n✅ All datasets have matching image/label counts")
    
    # Sample a few labels to show format
    print("\n📝 Sample Labels:")
    print("-" * 60)
    
    # Detection sample
    detection_labels = list((saver.detection_dir / "labels").glob("*.txt"))
    if detection_labels:
        sample_label = detection_labels[0]
        print(f"\n🎯 Detection Label: {sample_label.name}")
        with open(sample_label, 'r') as f:
            lines = f.readlines()[:3]  # First 3 lines
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id, x, y, w, h = parts
                    cls_name = "boat" if cls_id == "0" else "person"
                    print(f"   {cls_name}: center=({x}, {y}) size=({w}, {h})")
    
    # Segmentation sample
    segmentation_labels = list((saver.segmentation_dir / "labels").glob("*.txt"))
    if segmentation_labels:
        sample_label = segmentation_labels[0]
        print(f"\n🗺️  Segmentation Label: {sample_label.name}")
        with open(sample_label, 'r') as f:
            lines = f.readlines()[:3]  # First 3 lines
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = parts[0]
                    cls_name = ["beach", "water"][int(cls_id)]
                    print(f"   {cls_name}: polygon with {(len(parts)-1)//2} points")
    
    print("\n" + "=" * 60)
    print("💡 Tip: Review and correct labels before fine-tuning!")
    print("=" * 60)


if __name__ == "__main__":
    main()
