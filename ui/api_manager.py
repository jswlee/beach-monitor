"""
Manages the FastAPI subprocess for Streamlit

This module allows Streamlit to automatically start and manage the FastAPI
inference service as a subprocess.
"""
import subprocess
import time
import requests
import logging
from pathlib import Path
import atexit
import sys
import os

logger = logging.getLogger(__name__)

class APIManager:
    """Manages the FastAPI inference service as a subprocess"""
    
    def __init__(self, port=8000):
        self.port = port
        self.process = None
        self.api_url = f"http://localhost:{port}"
        
    def start(self):
        """Start the API server if not already running"""
        # Check if already running
        if self.is_running():
            logger.info(f"API already running at {self.api_url}")
            return True
        
        # Start the API server
        logger.info("Starting API server...")
        api_script = Path(__file__).parent.parent / "api" / "inference_service.py"
        
        if not api_script.exists():
            logger.error(f"API script not found at {api_script}")
            return False
        
        try:
            # Start subprocess with output suppression (or redirect to log file)
            self.process = subprocess.Popen(
                [sys.executable, str(api_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(Path(__file__).parent.parent)  # Set working directory to project root
            )
            
            # Wait for server to start (max 30 seconds)
            logger.info("Waiting for API server to start...")
            for i in range(30):
                if self.is_running():
                    logger.info(f"✅ API server started successfully at {self.api_url}")
                    return True
                time.sleep(1)
                if i % 5 == 0:
                    logger.info(f"Still waiting... ({i}s)")
            
            logger.error("❌ API server failed to start within 30 seconds")
            # Try to get error output
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                logger.error(f"Process stdout: {stdout}")
                logger.error(f"Process stderr: {stderr}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            return False
    
    def is_running(self):
        """Check if the API server is running"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def stop(self):
        """Stop the API server"""
        if self.process:
            logger.info("Stopping API server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Process didn't terminate, killing...")
                self.process.kill()
            self.process = None
            logger.info("API server stopped")
    
    def restart(self):
        """Restart the API server"""
        self.stop()
        time.sleep(2)
        return self.start()
    
    def get_status(self):
        """Get the status of the API server"""
        if self.is_running():
            return {
                "status": "running",
                "url": self.api_url,
                "healthy": True
            }
        elif self.process and self.process.poll() is None:
            return {
                "status": "starting",
                "url": self.api_url,
                "healthy": False
            }
        else:
            return {
                "status": "stopped",
                "url": self.api_url,
                "healthy": False
            }

# Global API manager instance
_api_manager = None

def get_api_manager(port=8000):
    """Get or create the global API manager"""
    global _api_manager
    if _api_manager is None:
        _api_manager = APIManager(port=port)
        # Register cleanup on exit
        atexit.register(_api_manager.stop)
        # Set environment variable for API tools
        os.environ["INFERENCE_API_URL"] = _api_manager.api_url
    return _api_manager

def ensure_api_running(port=8000):
    """Ensure the API is running, start it if not"""
    manager = get_api_manager(port=port)
    if not manager.is_running():
        return manager.start()
    return True

if __name__ == "__main__":
    # Test the API manager
    logging.basicConfig(level=logging.INFO)
    
    print("Testing API Manager...")
    manager = get_api_manager()
    
    print("\n1. Starting API...")
    if manager.start():
        print("✅ API started successfully")
    else:
        print("❌ Failed to start API")
        sys.exit(1)
    
    print("\n2. Checking status...")
    status = manager.get_status()
    print(f"Status: {status}")
    
    print("\n3. Testing health endpoint...")
    try:
        response = requests.get(f"{manager.api_url}/health")
        print(f"Health check: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    print("\n4. Stopping API...")
    manager.stop()
    print("✅ API stopped")
    
    print("\nAPI Manager test complete!")
