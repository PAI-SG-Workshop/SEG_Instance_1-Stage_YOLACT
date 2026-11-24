"""Evaluate seg YOLACT on the test split."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.runner import main
from detectors.yolact.build import build_model

if __name__ == "__main__":
    main("evaluate", task="seg", version="yolact", build_model=build_model)
