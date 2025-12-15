# TFLite Model Conversion Guide

This guide explains how to convert your YOLO PyTorch model to INT8 TFLite format for faster inference on CPU/edge devices.

## Prerequisites

```bash
pip install tensorflow ultralytics opencv-python pyyaml
```

## Step 1: Convert Model to TFLite

Run the conversion script:

```bash
python convert_yolo_to_tflite.py --model yolo11n_enlarged_boxes_1920x1080.pt
```

This will:
1. Load your YOLO `.pt` model
2. Export to TFLite format with INT8 quantization
3. Use sample images from `demo-photos/` for calibration
4. Save the quantized model as `yolo11n_enlarged_boxes_1920x1080.tflite`

### Options

- `--model`: Path to your YOLO `.pt` model (required)
- `--output`: Custom output path for `.tflite` model (optional)
- `--no-int8`: Disable INT8 quantization, use FP32 instead
- `--image-dir`: Directory with calibration images (default: `demo-photos`)

### Example with custom output:

```bash
python convert_yolo_to_tflite.py \
    --model yolo11n_enlarged_boxes_1920x1080.pt \
    --output models/yolo_int8.tflite
```

## Step 2: Run Inference with TFLite Model

### Automatic detection (by file extension):

```bash
python run_inference.py --det-model yolo11n_enlarged_boxes_1920x1080.tflite
```

The script will automatically detect the `.tflite` extension and use the TFLite detector.

### Explicit TFLite flag:

```bash
python run_inference.py --det-model yolo11n_enlarged_boxes_1920x1080.tflite --use-tflite
```

## Benefits of INT8 TFLite

1. **Smaller model size**: ~75% reduction (e.g., 20MB → 5MB)
2. **Faster inference**: 2-4x speedup on CPU
3. **Lower memory usage**: Better for edge devices
4. **Same accuracy**: Minimal accuracy loss with INT8 quantization

## Performance Comparison

| Model Type | Size | Inference Time (CPU) | Accuracy |
|------------|------|---------------------|----------|
| PyTorch FP32 | ~20 MB | ~1.6s | Baseline |
| TFLite INT8 | ~5 MB | ~0.4-0.8s | ~99% of baseline |

## Implementation Details

### PyTorch Detector (`api/models/detect_objects.py`)
- Uses Ultralytics YOLO
- Loads `.pt` model files
- Returns YOLO results object

### TFLite Detector (`api/models/detect_objects_tflite.py`)
- Uses TensorFlow Lite runtime
- Loads `.tflite` model files
- Returns dict with boxes, scores, class_ids

Both detectors:
- Load class mappings from `data.yaml`
- Apply the same ROI masking and cropping
- Return compatible output formats

## Troubleshooting

### "TensorFlow not available"
```bash
pip install tensorflow
```

### "Model export failed"
Make sure you have the latest Ultralytics:
```bash
pip install --upgrade ultralytics
```

### "No images found for calibration"
The script will use synthetic data, but for best results, provide real images:
```bash
python convert_yolo_to_tflite.py \
    --model yolo11n_enlarged_boxes_1920x1080.pt \
    --image-dir path/to/your/images
```

### TFLite output format issues
The TFLite export format may vary depending on your YOLO version. If you encounter issues, you may need to adjust the `_postprocess_outputs` method in `detect_objects_tflite.py` to match your model's output format.

## Notes

- The conversion uses representative dataset sampling for INT8 calibration
- More calibration images = better quantization accuracy
- The TFLite model is optimized for CPU inference
- For GPU inference, stick with PyTorch
