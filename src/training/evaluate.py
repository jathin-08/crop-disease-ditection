"""
Tomato Disease AI - Independent Test Evaluation Pipeline
==============================================================================
Evaluates YOLO26 on unseen test data across all disease classes.
Dynamically calculates:
  - 1-to-1 Ground-Truth Matching
  - Real TP, FP, FN
  - IoU values and Mean IoU
  - Precision = TP / (TP + FP)
  - Recall = TP / (TP + FN)
  - F1 Score = 2 * P * R / (P + R)
  - mAP@50 and mAP@50-95
Saves results dynamically to results/reports/metrics.json.
==============================================================================
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, resolve_model_path, get_classes
from src.metrics.evaluation import evaluate_dataset


def main(cli_args=None):
    if cli_args is not None:
        model_arg = getattr(cli_args, "model", "")
        conf_arg = getattr(cli_args, "conf", 0.001)
        iou_arg = getattr(cli_args, "iou", 0.50)
    else:
        parser = argparse.ArgumentParser(description="Evaluate Tomato Disease AI on Independent Test Dataset")
        parser.add_argument("--model", type=str, default="", help="Path to best.pt")
        parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
        parser.add_argument("--iou", type=float, default=0.50, help="IoU threshold")
        filtered_argv = [a for a in sys.argv[1:] if a != "evaluate"]
        args = parser.parse_args(filtered_argv)
        model_arg = args.model
        conf_arg = args.conf
        iou_arg = args.iou

    print("\n" + "=" * 75)
    print(" TOMATO LEAF DISEASE AI - INDEPENDENT TEST BENCHMARK")
    print("=" * 75)

    cfg = load_config()
    target_model = model_arg if model_arg else str(resolve_model_path())

    # Execute dataset evaluation
    results = evaluate_dataset(
        model=target_model,
        dataset_dir=PROJECT_ROOT / "dataset",
        split="test",
        conf=conf_arg,
        iou_nms=iou_arg,
        iou_match_thresh=iou_arg,
        class_names=get_classes()
    )

    threshold = cfg.get("deployment", {}).get("min_map50_readiness", 0.40)
    achieved = results.get("map50", 0.0)

    print("\n" + "=" * 75)
    print(" EVALUATION SUMMARY (DYNAMICALLY CALCULATED)")
    print("=" * 75)
    print(f"  Precision    : {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"  Recall       : {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"  F1 Score     : {results['f1']:.4f} ({results['f1']*100:.2f}%)")
    print(f"  Mean IoU     : {results['mean_iou']:.4f} ({results['mean_iou']*100:.2f}%)")
    print(f"  mAP@50       : {results['map50']:.4f} ({results['map50']*100:.2f}%)")
    print(f"  mAP@50-95    : {results['map50_95']:.4f} ({results['map50_95']*100:.2f}%)")
    print(f"  True Pos (TP): {results['tp']}")
    print(f"  False Pos(FP): {results['fp']}")
    print(f"  False Neg(FN): {results['fn']}")
    print(f"  Report Saved : {PROJECT_ROOT / 'results' / 'reports' / 'metrics.json'}")
    print("-" * 75)

    if achieved >= threshold:
        print(f" MODEL READINESS: READY (Test mAP@50 = {achieved*100:.2f}% >= {threshold*100:.1f}%)")
    else:
        print(f" MODEL READINESS: NOT READY (Test mAP@50 = {achieved*100:.2f}% < {threshold*100:.1f}%)")
    print("=" * 75 + "\n")

    return results


if __name__ == "__main__":
    main()
