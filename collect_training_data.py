"""
Simple script to continuously collect training data by running beach analysis on a timer.
"""
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from api.models import BeachAnalyzer, get_training_data_saver
from api.models.detect_objects import BeachDetector
from api.models.classify_regions import RegionClassifier
from api.models.capture_snapshot import BeachCapture

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_analysis_loop(interval_minutes: int = 2):
    """
    Run beach analysis on a timer to collect training data.
    
    Args:
        interval_minutes: Time between analyses in minutes (default: 2)
    """
    logger.info("=" * 60)
    logger.info("Training Data Collection - Starting")
    logger.info(f"Interval: {interval_minutes} minutes")
    logger.info("=" * 60)
    
    # Initialize components (lazy-loaded, will download models if needed)
    logger.info("Initializing components...")
    detector = BeachDetector()
    region_classifier = RegionClassifier()
    beach_capture = BeachCapture()
    analyzer = BeachAnalyzer(
        detector=detector,
        region_classifier=region_classifier,
        save_training_data=True
    )
    training_saver = get_training_data_saver()
    
    logger.info("✅ Initialization complete")
    logger.info("Press Ctrl+C to stop\n")
    
    iteration = 0
    interval_seconds = interval_minutes * 60
    
    try:
        while True:
            iteration += 1
            start_time = time.time()
            
            logger.info(f"{'='*60}")
            logger.info(f"Iteration #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*60}")
            
            try:
                # Step 1: Capture snapshot
                logger.info("📸 Capturing snapshot...")
                snapshot_path = beach_capture.capture_snapshot()
                
                if not snapshot_path:
                    logger.error("❌ Failed to capture snapshot, skipping this iteration")
                    time.sleep(interval_seconds)
                    continue
                
                logger.info(f"✅ Snapshot saved: {snapshot_path}")
                
                # Step 2: Run analysis (automatically saves training data)
                logger.info("🔍 Running analysis...")
                result = analyzer.analyze_beach_activity(
                    image_path=snapshot_path,
                    classify_locations=True,
                    save_annotated=True
                )
                
                # Log results
                logger.info(f"✅ Analysis complete:")
                logger.info(f"   👥 People: {result['people_count']} (beach: {result['beach_count']}, water: {result['water_count']}, other: {result['other_count']})")
                logger.info(f"   🚤 Boats: {result['boat_count']}")
                logger.info(f"   📊 Activity: {result['activity_level']}")
                
                # Show dataset stats
                stats = training_saver.get_dataset_stats()
                logger.info(f"📦 Training data collected:")
                logger.info(f"   Detection: {stats['detection']['images']} images")
                logger.info(f"   Segmentation: {stats['segmentation']['images']} images")
                
            except Exception as e:
                logger.error(f"❌ Error during iteration: {e}", exc_info=True)
            
            # Calculate sleep time
            elapsed = time.time() - start_time
            sleep_time = max(0, interval_seconds - elapsed)
            
            if sleep_time > 0:
                logger.info(f"⏳ Waiting {sleep_time:.1f} seconds until next iteration...\n")
                time.sleep(sleep_time)
            else:
                logger.info(f"⚠️  Analysis took longer than interval ({elapsed:.1f}s), starting next iteration immediately\n")
    
    except KeyboardInterrupt:
        logger.info("\n" + "="*60)
        logger.info("🛑 Stopping training data collection")
        logger.info("="*60)
        
        # Final stats
        stats = training_saver.get_dataset_stats()
        logger.info(f"\n📊 Final Statistics:")
        logger.info(f"   Total iterations: {iteration}")
        logger.info(f"   Detection dataset: {stats['detection']['images']} images")
        logger.info(f"   Segmentation dataset: {stats['segmentation']['images']} images")
        logger.info("\n✅ Training data saved to: training_data/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect training data by running beach analysis on a timer")
    parser.add_argument(
        "--interval",
        type=int,
        default=2,
        help="Time between analyses in minutes (default: 2)"
    )
    
    args = parser.parse_args()
    
    run_analysis_loop(interval_minutes=args.interval)
