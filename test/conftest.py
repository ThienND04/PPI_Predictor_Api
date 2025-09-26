import os
import sys
from pathlib import Path

# Ensure project root (the directory containing `src/`) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))




