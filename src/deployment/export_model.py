"""
Tomato Disease AI - Model Exporter
Exports trained YOLO26 checkpoint to deployment formats (ONNX, TorchScript, TFLite).
"""

import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import resolve_model_path


def export_checkpoint(
    weights_path: str = "",
    export_format: str = "torchscript",
    imgsz: int = 320
):
    target_weights = weights_path if weights_path else str(resolve_model_path())
    print("\n" + "=" * 75)
    print(" TOMATO LEAF DISEASE AI - MODEL EXPORT")
    print(f" Source Weights : {target_weights}")
    print(f" Target Format  : {export_format.upper()}")
    print("=" * 75)

    model = YOLO(target_weights)
    try:
        exported_path = model.export(format=export_format, imgsz=imgsz)
        print(f"\n[+] Successfully exported model to: {exported_path}")
        return exported_path
    except Exception as e:
        print(f"\n[x] Export encountered an issue: {e}")
        print("[!] Retaining optimal PyTorch checkpoint (.pt) for native execution.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Export Trained YOLO26 Model")
    parser.add_argument("--model", type=str, default="", help="Path to weights (default: best.pt)")
    parser.add_argument("--format", type=str, default="torchscript", choices=["torchscript", "onnx", "tflite"], help="Target format")
    parser.add_argument("--imgsz", type=int, default=320, help="Image size")
    args = parser.parse_args()

    export_checkpoint(args.model, args.format, args.imgsz)


if __name__ == "__main__":
    main()
