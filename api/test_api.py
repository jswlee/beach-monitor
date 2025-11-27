"""
Test script for the Beach Monitor API

This script tests all API endpoints in sequence:
1. Health check - verify API is running
2. Capture - get a fresh snapshot from the livestream
3. Download - retrieve the captured snapshot
4. Detect - run object detection on the captured image
5. Analyze - run full analysis on the captured image

Falls back to demo images if capture fails.
"""
import requests
import sys
from pathlib import Path

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_capture():
    """Test capture endpoint and return the snapshot path"""
    print("\n=== Testing Capture Endpoint ===")
    try:
        response = requests.post(f"{API_URL}/capture", timeout=30)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Snapshot URL: {result.get('snapshot_url')}")
        print(f"Timestamp: {result.get('timestamp')}")
        print(f"Processing time: {result.get('processing_time_ms'):.0f}ms")
        
        # Return the snapshot URL so we can use it for testing
        if response.status_code == 200:
            return result.get('snapshot_url')
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_detect(image_path):
    """Test detect endpoint"""
    print(f"\n=== Testing Detect Endpoint ===")
    print(f"Image: {image_path}")
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f, 'image/jpeg')}
            response = requests.post(f"{API_URL}/detect", files=files, timeout=60)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"People: {result.get('people_count')}")
        print(f"Boats: {result.get('boat_count')}")
        print(f"Activity: {result.get('activity_level')}")
        print(f"Processing time: {result.get('processing_time_ms'):.0f}ms")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_analyze(image_path):
    """Test analyze endpoint"""
    print(f"\n=== Testing Analyze Endpoint ===")
    print(f"Image: {image_path}")
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f, 'image/jpeg')}
            response = requests.post(f"{API_URL}/analyze", files=files, timeout=60)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"People: {result.get('people_count')} (Beach: {result.get('beach_count')}, Water: {result.get('water_count')})")
        print(f"Boats: {result.get('boat_count')}")
        print(f"Summary: {result.get('summary')}")
        print(f"Processing time: {result.get('processing_time_ms'):.0f}ms")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def download_snapshot(snapshot_url):
    """Download a snapshot from the API to test with"""
    print(f"\n=== Downloading Snapshot ===")
    print(f"URL: {API_URL}{snapshot_url}")
    try:
        response = requests.get(f"{API_URL}{snapshot_url}", timeout=10)
        if response.status_code == 200:
            # Save to temp file
            temp_path = Path(f"temp_snapshot_{Path(snapshot_url).name}")
            temp_path.write_bytes(response.content)
            print(f"✅ Downloaded to: {temp_path}")
            return str(temp_path)
        else:
            print(f"❌ Failed to download: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading: {e}")
        return None

def main():
    """Run all tests"""
    print("Beach Monitor API Test Suite")
    print("=" * 50)
    
    # Test health
    if not test_health():
        print("\n❌ API is not running. Start it with: python api/inference_service.py")
        sys.exit(1)
    
    # Test capture and get the snapshot
    snapshot_url = test_capture()
    
    # Use the captured snapshot for testing
    test_image = None
    if snapshot_url:
        # Download the snapshot to test with
        test_image = download_snapshot(snapshot_url)
    
    # Fallback to demo images if capture failed
    if not test_image:
        print("\n⚠️ Capture failed, trying demo images...")
        demo_dir = Path("demo-photos")
        if demo_dir.exists():
            images = list(demo_dir.glob("youtube_snapshot_*.jpg"))
            if images:
                # Filter out annotated/segmented images
                original_images = [img for img in images if not any(x in img.name for x in ['_annotated', '_segmented', '_regions', '_resized'])]
                if original_images:
                    test_image = original_images[0]
                else:
                    print("\n⚠️ No original demo images found")
            else:
                print("\n⚠️ No demo images found")
        else:
            print("\n⚠️ demo-photos directory not found")
    
    # Run detection and analysis tests
    if test_image:
        test_detect(test_image)
        test_analyze(test_image)
        
        # Cleanup temp file if we downloaded it
        if str(test_image).startswith("temp_snapshot_"):
            try:
                Path(test_image).unlink()
                print(f"\n🧹 Cleaned up temp file: {test_image}")
            except:
                pass
    else:
        print("\n❌ No image available for testing detect/analyze endpoints")
    
    print("\n" + "=" * 50)
    print("✅ Tests complete!")

if __name__ == "__main__":
    main()
