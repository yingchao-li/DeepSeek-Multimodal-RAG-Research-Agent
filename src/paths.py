"""Paths anchored to the real app root, independent of the launch directory."""
from pathlib import Path
import os

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get('DEEPSEEK_DATA_DIR', str(APP_ROOT / 'data'))).expanduser().resolve()
