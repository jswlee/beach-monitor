"""
FastAPI service for beach monitoring model inference.

This service exposes the YOLO detection and segmentation models as REST endpoints,
allowing any application to use the beach monitoring capabilities independently
of the LangGraph agent.

Endpoints:
- POST /detect - Run object detection on an image
- POST /segment - Run beach/water segmentation on an image
- POST /analyze - Complete analysis (detection + segmentation)
- GET /health - Health check
- GET /models/info - Get model information
"""
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import cv2
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
import sys

# Add parent directory to path and set working directory
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Change to project root so config.yaml can be found
import os
os.chdir(PROJECT_ROOT)

from api.models.detect_objects import BeachDetector
from api.models.classify_regions import RegionClassifier
from api.models.capture_snapshot import BeachCapture

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Beach Monitor Inference API",
    description="REST API for beach monitoring computer vision models",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances (lazy loaded)
_detector: Optional[BeachDetector] = None
_region_classifier: Optional[RegionClassifier] = None
_beach_capture: Optional[BeachCapture] = None


def get_detector() -> BeachDetector:
    """Get or initialize the beach detector."""
    global _detector
    if _detector is None:
        logger.info("Initializing BeachDetector...")
        _detector = BeachDetector()
    return _detector


def get_region_classifier() -> RegionClassifier:
    """Get or initialize the region classifier."""
    global _region_classifier
    if _region_classifier is None:
        logger.info("Initializing RegionClassifier...")
        _region_classifier = RegionClassifier()
    return _region_classifier


def get_beach_capture() -> BeachCapture:
    """Get or initialize the beach capture."""
    global _beach_capture
    if _beach_capture is None:
        logger.info("Initializing BeachCapture...")
        _beach_capture = BeachCapture()
    return _beach_capture


# Pydantic models for request/response
class DetectionResult(BaseModel):
    """Object detection result."""
    people_count: int = Field(..., description="Number of people detected")
    boat_count: int = Field(..., description="Number of boats detected")
    activity_level: str = Field(..., description="Activity level: empty, quiet, moderate, busy")
    summary: str = Field(..., description="Human-readable summary")
    annotated_image_url: Optional[str] = Field(None, description="URL to annotated image")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class SegmentationResult(BaseModel):
    """Beach/water segmentation result."""
    beach_count: int = Field(..., description="Number of people on beach")
    water_count: int = Field(..., description="Number of people in water")
    other_count: int = Field(..., description="Number of people in other areas")
    total_count: int = Field(..., description="Total number of people")
    classifications: List[Dict[str, Any]] = Field(..., description="Per-person classifications")
    segmented_image_url: Optional[str] = Field(None, description="URL to segmented image")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class AnalysisResult(BaseModel):
    """Complete beach analysis result."""
    people_count: int
    boat_count: int
    beach_count: int
    water_count: int
    other_count: int
    activity_level: str
    summary: str
    annotated_image_url: Optional[str] = None
    segmented_image_url: Optional[str] = None
    processing_time_ms: float


class SnapshotResult(BaseModel):
    """Beach snapshot capture result."""
    snapshot_url: str = Field(..., description="URL to captured snapshot")
    timestamp: str = Field(..., description="Capture timestamp")
    processing_time_ms: float


class ModelInfo(BaseModel):
    """Model information."""
    detector_loaded: bool
    segmentation_loaded: bool
    detector_type: str = "YOLOv8"
    segmentation_type: str = "SegFormer"


def cleanup_temp_file(path: str):
    """Background task to cleanup temporary files."""
    try:
        Path(path).unlink(missing_ok=True)
        logger.info(f"Cleaned up temp file: {path}")
    except Exception as e:
        logger.error(f"Error cleaning up {path}: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/models/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about loaded models."""
    return ModelInfo(
        detector_loaded=_detector is not None,
        segmentation_loaded=_region_classifier is not None
    )


@app.post("/capture", response_model=SnapshotResult)
async def capture_snapshot():
    """
    Capture a snapshot from the beach livestream.
    May return cached snapshot if one was captured recently (within 5 minutes).
    For guaranteed fresh snapshot, use /capture/fresh instead.
    
    Returns:
        SnapshotResult with URL to the captured image
    """
    start_time = datetime.now()
    
    try:
        # This uses BeachCapture which may have internal caching
        snapshot_path = get_beach_capture().capture_snapshot()
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return SnapshotResult(
            snapshot_url=f"/images/{Path(snapshot_path).name}",
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time
        )
    except Exception as e:
        logger.error(f"Error capturing snapshot: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to capture snapshot: {str(e)}")


@app.post("/detect", response_model=DetectionResult)
async def detect_objects(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    save_annotated: bool = True
):
    """
    Run object detection on an uploaded image.
    
    Args:
        file: Image file (JPEG, PNG)
        save_annotated: Whether to save and return annotated image
        
    Returns:
        DetectionResult with counts and activity level
    """
    start_time = datetime.now()
    temp_path = None
    
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name
        
        # Load image
        image = cv2.imread(temp_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run detection
        detector = get_detector()
        detection_results, raw_results = detector.detect_objects(image)
        
        # Generate annotated image if requested
        annotated_url = None
        if save_annotated:
            annotated_frame = raw_results[0].plot()
            annotated_path = Path(temp_path).parent / f"{Path(temp_path).stem}_annotated.jpg"
            cv2.imwrite(str(annotated_path), annotated_frame)
            annotated_url = f"/images/{annotated_path.name}"
            
            # Schedule cleanup
            if background_tasks:
                background_tasks.add_task(cleanup_temp_file, str(annotated_path))
        
        # Determine activity level
        people_count = detection_results['people_count']
        boat_count = detection_results['boat_count']
        
        if people_count == 0 and boat_count == 0:
            activity_level = "empty"
        elif people_count <= 5:
            activity_level = "quiet"
        elif people_count <= 15:
            activity_level = "moderate"
        else:
            activity_level = "busy"
        
        summary = f"Detected {people_count} people and {boat_count} boats. Beach is {activity_level}."
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Schedule cleanup of temp file
        if background_tasks:
            background_tasks.add_task(cleanup_temp_file, temp_path)
        
        return DetectionResult(
            people_count=people_count,
            boat_count=boat_count,
            activity_level=activity_level,
            summary=summary,
            annotated_image_url=annotated_url,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/segment", response_model=SegmentationResult)
async def segment_beach_water(
    file: UploadFile = File(...),
    person_boxes: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
    save_segmented: bool = True
):
    """
    Run beach/water segmentation on an uploaded image with person bounding boxes.
    
    Args:
        file: Image file (JPEG, PNG)
        person_boxes: JSON string of person bounding boxes (optional, will run detection if not provided)
        save_segmented: Whether to save and return segmented image
        
    Returns:
        SegmentationResult with beach/water counts
    """
    start_time = datetime.now()
    temp_path = None
    
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name
        
        # Load image
        image = cv2.imread(temp_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Get person boxes (either from parameter or run detection)
        if person_boxes:
            import json
            boxes = json.loads(person_boxes)
        else:
            # Run detection to get person boxes
            detector = get_detector()
            _, raw_results = detector.detect_objects(image)
            result = raw_results[0]
            boxes = []
            if result.boxes is not None:
                for i, cls in enumerate(result.boxes.cls):
                    if int(cls) == 1:  # person class
                        box = result.boxes.xyxy[i].cpu().numpy()
                        boxes.append({'xyxy': box.tolist()})
        
        # Run segmentation
        classifier = get_region_classifier()
        segmented_path = None
        if save_segmented:
            segmented_path = Path(temp_path).parent / f"{Path(temp_path).stem}_segmented.jpg"
        
        result = classifier.classify_locations(
            image,
            boxes,
            save_annotated=save_segmented,
            annotated_path=str(segmented_path) if segmented_path else None
        )
        
        segmented_url = None
        if save_segmented and segmented_path and segmented_path.exists():
            segmented_url = f"/images/{segmented_path.name}"
            if background_tasks:
                background_tasks.add_task(cleanup_temp_file, str(segmented_path))
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Cleanup temp file
        if background_tasks:
            background_tasks.add_task(cleanup_temp_file, temp_path)
        
        return SegmentationResult(
            beach_count=result['beach_count'],
            water_count=result['water_count'],
            other_count=result['other_count'],
            total_count=result['total_count'],
            classifications=result['classifications'],
            segmented_image_url=segmented_url,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error during segmentation: {e}")
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_beach(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Complete beach analysis: detection + segmentation.
    
    Args:
        file: Image file (JPEG, PNG)
        
    Returns:
        AnalysisResult with complete analysis
    """
    start_time = datetime.now()
    temp_path = None
    
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name
        
        # Run full analysis using BeachAnalyzer
        from api.models.analyze_beach import BeachAnalyzer
        analyzer = BeachAnalyzer(detector=get_detector(), region_classifier=get_region_classifier())
        analysis = analyzer.analyze_beach_activity(temp_path)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Convert paths to URLs
        annotated_url = f"/images/{Path(analysis['annotated_image_path']).name}"
        segmented_url = None
        if 'regions_image_path' in analysis:
            segmented_url = f"/images/{Path(analysis['regions_image_path']).name}"
        
        # Cleanup temp file
        if background_tasks:
            background_tasks.add_task(cleanup_temp_file, temp_path)
        
        return AnalysisResult(
            people_count=analysis['people_count'],
            boat_count=analysis['boat_count'],
            beach_count=analysis.get('beach_count', 0),
            water_count=analysis.get('water_count', 0),
            other_count=analysis.get('other_count', 0),
            activity_level=analysis['activity_level'],
            summary=analysis['summary'],
            annotated_image_url=annotated_url,
            segmented_image_url=segmented_url,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/images/{filename}")
async def get_image(filename: str):
    """
    Serve generated images.
    
    Args:
        filename: Image filename
        
    Returns:
        Image file
    """
    # Look in temp directory and snapshots directory
    temp_dir = Path(tempfile.gettempdir())
    snapshots_dir = Path("data/snapshots")
    
    for directory in [temp_dir, snapshots_dir]:
        image_path = directory / filename
        if image_path.exists():
            return FileResponse(image_path)
    
    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/images/latest/original")
async def get_latest_original():
    """
    Get the most recent original snapshot.
    
    Returns:
        The latest snapshot image file
    """
    import re
    
    snapshots_dir = Path("data/snapshots")
    
    if not snapshots_dir.exists():
        raise HTTPException(status_code=404, detail="No snapshots directory found")
    
    # Find the most recent snapshot - only original files (6 digits before .jpg)
    # Pattern: youtube_snapshot_YYYYMMDD_HHMMSS.jpg
    pattern = re.compile(r'youtube_snapshot_\d{8}_\d{6}\.jpg$')
    snapshots = [p for p in snapshots_dir.glob("*.jpg") if pattern.match(p.name)]
    
    if not snapshots:
        raise HTTPException(status_code=404, detail="No snapshots found")
    
    latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
    return FileResponse(latest_snapshot)


@app.get("/images/latest/annotated")
async def get_latest_annotated():
    """
    Get the most recent annotated image (with detection bounding boxes).
    
    Returns:
        The latest annotated image file
    """
    temp_dir = Path(tempfile.gettempdir())
    snapshots_dir = Path("data/snapshots")
    
    # Look for annotated images in both directories
    annotated_images = []
    for directory in [temp_dir, snapshots_dir]:
        if directory.exists():
            annotated_images.extend(directory.glob("*_annotated.jpg"))
    
    if not annotated_images:
        raise HTTPException(status_code=404, detail="No annotated images found. Run /analyze first.")
    
    latest_annotated = max(annotated_images, key=lambda p: p.stat().st_mtime)
    return FileResponse(latest_annotated)


@app.get("/images/latest/segmented")
async def get_latest_segmented():
    """
    Get the most recent segmented image (beach vs water classification).
    
    Returns:
        The latest segmented image file
    """
    temp_dir = Path(tempfile.gettempdir())
    snapshots_dir = Path("data/snapshots")
    
    # Look for segmented images in both directories
    segmented_images = []
    for directory in [temp_dir, snapshots_dir]:
        if directory.exists():
            segmented_images.extend(directory.glob("*_segmented.jpg"))
    
    if not segmented_images:
        raise HTTPException(status_code=404, detail="No segmented images found. Run /analyze first.")
    
    latest_segmented = max(segmented_images, key=lambda p: p.stat().st_mtime)
    return FileResponse(latest_segmented)


@app.post("/capture/fresh", response_model=SnapshotResult)
async def capture_fresh_snapshot():
    """
    Force capture a FRESH snapshot, bypassing any cache.
    Use this when user explicitly wants current/live beach view.
    
    Returns:
        SnapshotResult with URL to the newly captured image
    """
    start_time = datetime.now()
    
    try:
        # Force fresh capture
        snapshot_path = get_beach_capture().capture_snapshot()
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return SnapshotResult(
            snapshot_url=f"/images/{Path(snapshot_path).name}",
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time
        )
    except Exception as e:
        logger.error(f"Error capturing fresh snapshot: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to capture snapshot: {str(e)}")


@app.post("/analyze/fresh", response_model=AnalysisResult)
async def analyze_beach_fresh(background_tasks: BackgroundTasks = None):
    """
    Capture a FRESH snapshot and analyze it.
    Use this when user explicitly wants current beach conditions ("now", "right now", "currently").
    
    Returns:
        AnalysisResult with complete fresh analysis
    """
    start_time = datetime.now()
    temp_path = None
    
    try:
        # Force capture fresh snapshot
        snapshot_path = get_beach_capture().capture_snapshot()
        temp_path = snapshot_path
        
        # Run full analysis
        from api.models.analyze_beach import BeachAnalyzer
        analyzer = BeachAnalyzer(detector=get_detector(), region_classifier=get_region_classifier())
        analysis = analyzer.analyze_beach_activity(temp_path)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Convert paths to URLs
        annotated_url = f"/images/{Path(analysis['annotated_image_path']).name}"
        segmented_url = None
        if 'regions_image_path' in analysis:
            segmented_url = f"/images/{Path(analysis['regions_image_path']).name}"
        
        return AnalysisResult(
            people_count=analysis['people_count'],
            boat_count=analysis['boat_count'],
            beach_count=analysis.get('beach_count', 0),
            water_count=analysis.get('water_count', 0),
            other_count=analysis.get('other_count', 0),
            activity_level=analysis['activity_level'],
            summary=analysis['summary'],
            annotated_image_url=annotated_url,
            segmented_image_url=segmented_url,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error during fresh analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

