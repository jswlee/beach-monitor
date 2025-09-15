"""
Beach Monitor MVP - Main entry point
"""
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_ui():
    """Launch the Streamlit UI"""
    import subprocess
    ui_path = Path(__file__).parent / "ui" / "chat.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_path)])

if __name__ == "__main__":
    run_ui()
