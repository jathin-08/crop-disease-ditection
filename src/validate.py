import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.validate import run_validation

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate YOLO26 Model")
    parser.add_argument("--model", type=str, default="", help="Model weight path")
    parser.add_argument("--split", type=str, default="val", help="Split: 'val' or 'test'")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="IoU threshold")
    args = parser.parse_args()
    run_validation(args.model, args.split, args.conf, args.iou)
