"""
Convert YOLO PyTorch model to INT8 TFLite format for edge deployment.

This script:
1. Loads the YOLO .pt model
2. Exports to TFLite format with INT8 quantization
3. Uses representative dataset for calibration
4. Saves the quantized .tflite model

Usage:
    python convert_yolo_to_tflite.py --model yolo11n_enlarged_boxes_1920x1080.pt
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_representative_dataset(image_dir: str = "data", num_samples: int = 100):
    """
    Generate a representative dataset for INT8 quantization calibration.
    
    Args:
        image_dir: Directory containing sample images
        num_samples: Number of samples to use for calibration
        
    Yields:
        Preprocessed image batches for calibration
    """
    # Find all .jpg files in the directory
    all_jpg_files = list(Path(image_dir).glob("**/*.jpg"))
    
    # Filter to only include files that match the pattern: end with timestamp + .jpg
    # Pattern: youtube_snapshot_YYYYMMDD_HHMMSS.jpg
    import re
    timestamp_pattern = re.compile(r'_\d{6,8}_\d{6}\.jpg$')
    image_paths = [p for p in all_jpg_files if timestamp_pattern.search(str(p))]
    
    # Exclude any files with suffixes like _segmented, _annotated, _det_annotated, etc.
    excluded_suffixes = ['_segmented.jpg', '_annotated.jpg', '_det_annotated.jpg', '_raw_seg.jpg']
    image_paths = [p for p in image_paths if not any(str(p).endswith(suffix) for suffix in excluded_suffixes)]
    
    if not image_paths:
        logger.warning(f"No images found in {image_dir}, using synthetic data")
        # Generate synthetic images if no real images available
        for _ in range(num_samples):
            # Create random image (1920x1080 RGB)
            img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            # Normalize to [0, 1] and convert to float32
            img = img.astype(np.float32) / 255.0
            # Add batch dimension and transpose to (1, 3, H, W)
            img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
            yield [img]
    else:
        logger.info(f"Found {len(image_paths)} images for calibration")
        
        # Use available images, cycling if needed
        for i in range(num_samples):
            img_path = image_paths[i % len(image_paths)]
            img = cv2.imread(str(img_path))
            
            if img is None:
                continue
                
            # Resize to model input size (1920x1080 or whatever your model expects)
            img = cv2.resize(img, (1920, 1080))
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Transpose to (C, H, W) and add batch dimension
            img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
            
            yield [img]


def convert_to_tflite(
    model_path: str,
    output_path: str = None,
    int8: bool = True,
    image_dir: str = "demo-photos"
):
    """
    Convert YOLO model to TFLite format with optional INT8 quantization.
    
    Args:
        model_path: Path to YOLO .pt model
        output_path: Output path for .tflite model (default: same name as input)
        int8: Whether to use INT8 quantization (default: True)
        image_dir: Directory with sample images for calibration
    """
    logger.info(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    
    # Determine output path
    if output_path is None:
        output_path = str(Path(model_path).with_suffix('.tflite'))
    
    logger.info(f"Output path: {output_path}")
    
    # Export to TFLite
    # Ultralytics YOLO export supports TFLite with int8 quantization
    try:
        logger.info("Starting export to TFLite format...")
        
        if int8:
            logger.info("Using INT8 quantization for smaller model size and faster inference")
            logger.info("Generating representative dataset for calibration...")
            
            # Export with INT8 quantization
            # The YOLO export will handle the quantization internally
            model.export(
                format='tflite',
                int8=True,
                data='data.yaml',  # Uses your data.yaml for class info
                imgsz=1920,  # Input image size
            )
            
            # The exported file will be saved as {model_name}_saved_model/{model_name}_int8.tflite
            # Let's find and rename it
            model_stem = Path(model_path).stem
            exported_dir = Path(f"{model_stem}_saved_model")
            
            if exported_dir.exists():
                # Find the int8 tflite file
                tflite_files = list(exported_dir.glob("*_int8.tflite"))
                if tflite_files:
                    src_file = tflite_files[0]
                    # Move to desired output path
                    import shutil
                    shutil.move(str(src_file), output_path)
                    logger.info(f"✓ INT8 TFLite model saved to: {output_path}")
                    
                    # Clean up the saved_model directory
                    shutil.rmtree(exported_dir)
                else:
                    logger.warning("Could not find exported INT8 TFLite file")
            else:
                logger.warning(f"Export directory {exported_dir} not found")
        else:
            logger.info("Using FP32 (no quantization)")
            model.export(
                format='tflite',
                int8=False,
                imgsz=1920,
            )
            
            # Similar file handling for FP32
            model_stem = Path(model_path).stem
            exported_dir = Path(f"{model_stem}_saved_model")
            
            if exported_dir.exists():
                tflite_files = list(exported_dir.glob("*.tflite"))
                if tflite_files:
                    src_file = tflite_files[0]
                    import shutil
                    shutil.move(str(src_file), output_path)
                    logger.info(f"✓ FP32 TFLite model saved to: {output_path}")
                    shutil.rmtree(exported_dir)
        
        # Print model info
        import os
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"Model size: {size_mb:.2f} MB")
            
            # Compare with original
            orig_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            logger.info(f"Original model size: {orig_size_mb:.2f} MB")
            logger.info(f"Size reduction: {((orig_size_mb - size_mb) / orig_size_mb * 100):.1f}%")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO model to TFLite INT8")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLO .pt model file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for .tflite model (default: same name as input)"
    )
    parser.add_argument(
        "--no-int8",
        action="store_true",
        help="Disable INT8 quantization (use FP32 instead)"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data",
        help="Directory with sample images for INT8 calibration (default: data/)"
    )
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not Path(args.model).exists():
        logger.error(f"Model file not found: {args.model}")
        return
    
    # Convert
    convert_to_tflite(
        model_path=args.model,
        output_path=args.output,
        int8=not args.no_int8,
        image_dir=args.image_dir
    )
    
    logger.info("✓ Conversion complete!")


if __name__ == "__main__":
    main()
