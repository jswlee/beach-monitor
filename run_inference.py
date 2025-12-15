#!/usr/bin/env python3
"""
Simple inference runner for the two trained models (object detection + segmentation).

Usage:
  python run_inference.py [--image PATH] [--seg-model PATH] [--skip-models]

By default picks a random image from demo-photos/ and attempts to load models
using the project's model-loading code. If model loading fails, the script will
handle errors and exit gracefully.
"""
import argparse
import random
import sys
from pathlib import Path
import cv2
import logging
import time # <-- Import the time module


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pick_random_image(demo_dir: Path) -> Path:
    images = list(demo_dir.glob("**/*.*"))
    images = [p for p in images if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    if not images:
        raise FileNotFoundError(f"No images found in {demo_dir}")
    return random.choice(images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="demo-photos/youtube_snapshot_20250902_135358.jpg", help="Path to image to run inference on")
    parser.add_argument("--seg-model", help="Optional path to segmentation model directory")
    parser.add_argument("--skip-models", action="store_true", help="Do not attempt to load models (dry run)")
    parser.add_argument("--det-model", help="Optional path to local YOLO model file (.pt or .tflite)")
    parser.add_argument("--use-tflite", action="store_true", help="Use TFLite model for detection (faster on CPU)")
    args = parser.parse_args()

    repo_root = Path(__file__).parent
    demo_dir = repo_root / "demo-photos"

    if args.image:
        image_path = Path(args.image)
    else:
        try:
            image_path = pick_random_image(demo_dir)
        except Exception as e:
            logger.error(f"Failed to pick random image: {e}")
            sys.exit(1)

    logger.info(f"Using image: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        logger.error(f"Could not read image: {image_path}")
        sys.exit(1)

    # Try to import and run detector + classifier
    if args.skip_models:
        logger.info("Skipping model load (--skip-models). Exiting after basic checks.")
        print("Image load OK. Models skipped.")
        sys.exit(0)

    # Import model classes from api.models
    try:
        from api.models.classify_regions import RegionClassifier
        
        # Import appropriate detector based on model type
        if args.use_tflite or (args.det_model and args.det_model.endswith('.tflite')):
            from api.models.detect_objects_tflite import BeachDetectorTFLite as BeachDetector
            logger.info("Using TFLite detector")
        else:
            from api.models.detect_objects import BeachDetector
            logger.info("Using PyTorch detector")
    except Exception as e:
        logger.error(f"Failed to import model classes: {e}")
        sys.exit(1)

    # Initialize detector
    try:
        if args.use_tflite or (args.det_model and args.det_model.endswith('.tflite')):
            # TFLite detector requires model_path
            if not args.det_model:
                logger.error("TFLite detector requires --det-model argument")
                sys.exit(1)
            detector = BeachDetector(model_path=args.det_model)
        else:
            # PyTorch detector
            detector = BeachDetector(model_path=args.det_model) if args.det_model else BeachDetector()

        # --- Start Detection Timing ---
        start_time_det = time.time()
        detection_result, raw_results = detector.detect_objects(img)
        end_time_det = time.time()
        # --- End Detection Timing ---

        logger.info(f"Object Detection Inference Time: {end_time_det - start_time_det:.4f} seconds")

        print("--- Detection Result ---")
        print(f"People: {detection_result['people_count']}")
        print(f"Boats: {detection_result['boat_count']}")

        # Save annotated if possible
        try:
            annotated = raw_results[0].plot()
            out_path = image_path.parent / f"{image_path.stem}_det_annotated.jpg"
            cv2.imwrite(str(out_path), annotated)
            logger.info(f"Saved detection annotated image to: {out_path}")
        except Exception as e:
            logger.warning(f"Could not save detection annotated image: {e}")

    except Exception as e:
        logger.error(f"Object detector failed: {e}")
        sys.exit(1)

    # Build person boxes for segmentation classifier
    # Use the person_class_id from the detector (loaded from data.yaml)
    person_boxes = []
    try:
        if args.use_tflite or (args.det_model and args.det_model.endswith('.tflite')):
            # TFLite returns dict with boxes, scores, class_ids
            boxes = raw_results.get('boxes', [])
            class_ids = raw_results.get('class_ids', [])
            
            for i, cls in enumerate(class_ids):
                if int(cls) == detector.person_class_id:
                    box = boxes[i]
                    person_boxes.append({'xyxy': box.tolist()})
        else:
            # PyTorch YOLO returns results object
            result = raw_results[0]
            if result.boxes is not None:
                for i, cls in enumerate(result.boxes.cls):
                    # Use dynamic person class ID from detector
                    if int(cls) == detector.person_class_id:
                        box = result.boxes.xyxy[i].cpu().numpy()
                        person_boxes.append({'xyxy': box.tolist()})
    except Exception as e:
        logger.warning(f"Failed to extract person boxes from detector results: {e}")

    # Initialize region classifier (allow passing a local model path)
    try:
        seg_model_path = args.seg_model if args.seg_model else None
        classifier = RegionClassifier(model_path=seg_model_path)

        regions_out = image_path.parent / f"{image_path.stem}_segmented.jpg"
        
        # --- Start Classification Timing ---
        start_time_seg = time.time()
        loc_results = classifier.classify_locations(img, person_boxes, save_annotated=True, annotated_path=str(regions_out))
        end_time_seg = time.time()
        # --- End Classification Timing ---
        
        logger.info(f"Region Classification Inference Time: {end_time_seg - start_time_seg:.4f} seconds") # <-- Log time

        print("--- Location Classification ---")
        print(f"Beach: {loc_results.get('beach_count', 0)}")
        print(f"Water: {loc_results.get('water_count', 0)}")
        print(f"Other: {loc_results.get('other_count', 0)}")
        logger.info(f"Saved segmentation annotated image to: {regions_out}")

    except Exception as e:
        logger.error(f"Region classifier failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()