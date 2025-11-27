"""
Beach monitoring models package.

This package contains all the computer vision models and utilities for beach monitoring:
- Object detection (people and boats)
- Region classification (beach vs water)
- Beach snapshot capture
- Beach analysis
- Training data collection
"""

from api.models.detect_objects import BeachDetector
from api.models.classify_regions import RegionClassifier
from api.models.capture_snapshot import BeachCapture
from api.models.analyze_beach import BeachAnalyzer
from api.models.save_training_data import TrainingDataSaver, get_training_data_saver

__all__ = [
    'BeachDetector', 
    'RegionClassifier', 
    'BeachCapture', 
    'BeachAnalyzer',
    'TrainingDataSaver',
    'get_training_data_saver'
]
