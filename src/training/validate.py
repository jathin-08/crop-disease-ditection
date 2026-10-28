"""
Tomato Disease AI - Validation & Benchmarking Pipeline
Validates YOLO26 on validation set, computes precision/recall/F1/mAP,
exports confusion matrix, and performs zero-metric debugging if anomalies occur.
"""

import sys
import shutil
import argparse
from pathlib import Path
from typing import Any, Optional, Dict, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ultralytics import YOLO

from src.utils.config import load_config, resolve_model_path, get_classes
from src.utils.logger import setup_logger
from src.metrics.calculate_metrics import extract_overall_metrics
from src.metrics.per_class_metrics import compute_per_class_metrics
from src.metrics.confusion_matrix import plot_confusion_matrix
from src.metrics.generate_report import export_model_evaluation_reports
from src.metrics.iou_metrics import (
    evaluate_iou,
    export_iou_csv,
    generate_iou_bar_chart
)


logger = setup_logger("validate_yolo26")


def diagnose_zero_metrics(model_path: str, data_yaml_path: str, val_results: Any) -> str:
    """Diagnoses root cause when validation yields 0 precision, recall, or mAP."""
    lines = [
        "=" * 75,
        " ZERO-METRIC AUTOMATIC DIAGNOSTICS REPORT",
        "=" * 75,
        f"Model Evaluated : {model_path}",
        f"Dataset Config  : {data_yaml_path}",
        "-" * 75,
        "DIAGNOSTIC CHECKS EXECUTED:"
    ]

    import yaml
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        dy = yaml.safe_load(f)

    # 1. Class count check
    nc = dy.get("nc", 0)
    lines.append(f"  [1] Class count configured: nc={nc}")

    # 2. Validation set check
    val_rel = dy.get("val", "images/val")
    val_dir = PROJECT_ROOT / dy.get("path", "dataset") / val_rel
    val_imgs = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png"))
    lines.append(f"  [2] Validation images found: {len(val_imgs)} in {val_dir}")

    # 3. Label check
    lbl_dir = PROJECT_ROOT / dy.get("path", "dataset") / "labels" / "val"
    val_lbls = list(lbl_dir.glob("*.txt"))
    lines.append(f"  [3] Validation label files: {len(val_lbls)} in {lbl_dir}")

    # 4. Check detections count
    lines.append(f"  [4] Model Task: detect (YOLO26)")
    lines.append("  [5] Likely causes:")
    lines.append("      - Confidence threshold too high for early-stage training.")
    lines.append("      - Model under-trained (epochs too low or learning rate too low).")
    lines.append("      - Annotation format or bounding box mismatch.")
    lines.append("=" * 75 + "\n")

    report_str = "\n".join(lines)
    debug_path = PROJECT_ROOT / "results" / "reports" / "debug_report.txt"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    with open(debug_path, "w", encoding="utf-8") as f:
        f.write(report_str)
    return report_str


def run_validation(
    model_path: Optional[str] = None,
    split: str = "val",
    conf: float = 0.001,
    iou: float = 0.50
):
    print("\n" + "=" * 75)
    print(f" TOMATO LEAF DISEASE AI - YOLO26 VALIDATION ({split.upper()})")
    print("=" * 75)

    cfg = load_config()
    target_model = model_path if model_path else str(resolve_model_path())
    print(f"[*] Loading Weights : {target_model}")
    model = YOLO(target_model)

    device = "0" if torch.cuda.is_available() else "cpu"
    data_yaml = str(PROJECT_ROOT / "dataset" / "data.yaml")

    val_run_dir = PROJECT_ROOT / "results" / "validation"
    val_run_dir.mkdir(parents=True, exist_ok=True)

    # Run validation
    val_results = model.val(
        data=data_yaml,
        split=split,
        conf=conf,
        iou=iou,
        device=device,
        plots=True,
        project=str(val_run_dir),
        name="val_run",
        exist_ok=True,
        verbose=True
    )

    # Extract Metrics
    overall = extract_overall_metrics(val_results)
    classes = get_classes()
    per_class = compute_per_class_metrics(val_results, classes)

    # Execute Dataset IoU Evaluation (Ground Truth vs Model Predictions)
    print(f"\n[*] Computing IoU (Intersection over Union) on {split.upper()} dataset...")
    iou_summary = evaluate_iou(
        model=model,
        dataset_dir=PROJECT_ROOT / "dataset",
        split=split,
        conf=conf,
        iou_nms=iou,
        class_names=classes,
        device=device
    )

    # Connect IoU metrics with existing YOLO evaluation metrics
    overall["mean_iou"] = iou_summary["mean_iou"]
    overall["mean_iou_pct"] = iou_summary["mean_iou_pct"]
    overall["min_iou"] = iou_summary["min_iou"]
    overall["max_iou"] = iou_summary["max_iou"]
    overall["evaluated_detections"] = iou_summary["evaluated_detections"]
    overall["iou_gte_50_count"] = iou_summary["iou_gte_50_count"]
    overall["iou_gte_50_pct"] = iou_summary["iou_gte_50_pct"]
    overall["iou_gte_75_count"] = iou_summary["iou_gte_75_count"]
    overall["iou_gte_75_pct"] = iou_summary["iou_gte_75_pct"]

    # Merge per-class IoU metrics into per_class dictionary
    iou_by_cid = {c["class_id"]: c for c in iou_summary.get("per_class_iou", [])}
    for pc in per_class:
        cid = pc["class_id"]
        if cid in iou_by_cid:
            pc["mean_iou"] = iou_by_cid[cid]["mean_iou"]
            pc["min_iou"] = iou_by_cid[cid]["min_iou"]
            pc["max_iou"] = iou_by_cid[cid]["max_iou"]
            pc["iou_gte_50_count"] = iou_by_cid[cid]["iou_gte_50_count"]
            pc["iou_gte_50_pct"] = iou_by_cid[cid]["iou_gte_50_pct"]
            pc["iou_gte_75_count"] = iou_by_cid[cid]["iou_gte_75_count"]
            pc["iou_gte_75_pct"] = iou_by_cid[cid]["iou_gte_75_pct"]
            pc["evaluated_detections"] = iou_by_cid[cid]["evaluated_detections"]

    print("\n" + "=" * 50)
    print(" MODEL EVALUATION")
    print("-" * 50)
    print(f"Precision       : {overall['precision']*100:.2f}%")
    print(f"Recall          : {overall['recall']*100:.2f}%")
    print(f"mAP@50          : {overall['map50']*100:.2f}%")
    print(f"mAP@50-95       : {overall['map50_95']*100:.2f}%")
    print(f"F1 Score        : {overall['f1']*100:.2f}%")
    print(f"Mean IoU        : {overall['mean_iou']*100:.2f}%")
    print(f"IoU >= 0.50     : {overall['iou_gte_50_pct']:.2f}% detections")
    print(f"IoU >= 0.75     : {overall['iou_gte_75_pct']:.2f}% detections")
    print("=" * 50)

    print("\n" + "-" * 75)
    print("PER-CLASS IoU BREAKDOWN:")
    print(f"  {'Disease Name':<30} | {'Detections':<10} | {'Mean IoU':<10} | {'IoU >= 0.50':<12} | {'IoU >= 0.75'}")
    print("  " + "-" * 75)
    for pc in per_class:
        m_iou = pc.get("mean_iou", 0.0)
        p50 = pc.get("iou_gte_50_pct", 0.0)
        p75 = pc.get("iou_gte_75_pct", 0.0)
        n_eval = pc.get("evaluated_detections", 0)
        print(f"  {pc['disease_name']:<30} | {n_eval:<10} | {m_iou:<10.4f} | {p50:<11.1f}% | {p75:.1f}%")
    print("-" * 75)

    # Zero-metric anomaly check
    if overall["precision"] == 0 or overall["recall"] == 0 or overall["map50"] == 0:
        print("\n[!] WARNING: Zero-metric detected. Performing automatic diagnostics...")
        diag = diagnose_zero_metrics(target_model, data_yaml, val_results)
        print(diag)

    # Target metrics directories
    metrics_primary_dir = PROJECT_ROOT / "metrics"
    metrics_results_dir = PROJECT_ROOT / "results" / "metrics"
    metrics_primary_dir.mkdir(parents=True, exist_ok=True)
    metrics_results_dir.mkdir(parents=True, exist_ok=True)

    # Export iou_results.csv to results/metrics/, metrics/, and results/
    matches = iou_summary.get("matches", [])
    export_iou_csv(matches, metrics_primary_dir / "iou_results.csv")
    export_iou_csv(matches, metrics_results_dir / "iou_results.csv")
    export_iou_csv(matches, PROJECT_ROOT / "results" / "iou_results.csv")

    # Generate and export IoU per-class bar chart
    chart_p1 = PROJECT_ROOT / "results" / "graphs" / "iou_per_class.png"
    chart_p2 = metrics_results_dir / "iou_per_class.png"
    generate_iou_bar_chart(iou_summary.get("per_class_iou", []), chart_p1)
    if chart_p1.exists():
        shutil.copy2(chart_p1, chart_p2)
        shutil.copy2(chart_p1, metrics_primary_dir / "iou_per_class.png")

    # Save confusion matrix
    cm_dir = PROJECT_ROOT / "results" / "confusion_matrix"
    cm_dir.mkdir(parents=True, exist_ok=True)
    
    if hasattr(val_results, "confusion_matrix") and val_results.confusion_matrix is not None:
        raw_cm = val_results.confusion_matrix.matrix
        cnames = [classes.get(i, f"c_{i}") for i in range(len(classes))]
        plot_confusion_matrix(raw_cm[:len(classes), :len(classes)], cnames, cm_dir / "confusion_matrix_normalized.png", normalize=True)
        plot_confusion_matrix(raw_cm[:len(classes), :len(classes)], cnames, cm_dir / "confusion_matrix_raw.png", normalize=False)
        # Also copy to root results and metrics directories
        shutil.copy2(cm_dir / "confusion_matrix_normalized.png", PROJECT_ROOT / "results" / "confusion_matrix.png")
        for m_dir in [metrics_primary_dir, metrics_results_dir]:
            shutil.copy2(cm_dir / "confusion_matrix_normalized.png", m_dir / "confusion_matrix_normalized.png")
            shutil.copy2(cm_dir / "confusion_matrix_raw.png", m_dir / "confusion_matrix_raw.png")

    # Copy curves to results/graphs/ and metrics directories
    graphs_dir = PROJECT_ROOT / "results" / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    val_out_dir = val_run_dir / "val_run"
    for img_curve in val_out_dir.glob("*.png"):
        shutil.copy2(img_curve, graphs_dir / img_curve.name)
        for m_dir in [metrics_primary_dir, metrics_results_dir]:
            shutil.copy2(img_curve, m_dir / img_curve.name)

    # Generate Reports across all metrics directories
    dataset_info = {
        "path": "dataset/",
        "num_classes": len(classes),
        "val_images": len(list((PROJECT_ROOT / "dataset" / "images" / split).glob("*.*")))
    }
    rep_txt = PROJECT_ROOT / "results" / "reports" / "model_report.txt"
    rep_json = PROJECT_ROOT / "results" / "reports" / "metrics.json"
    threshold = cfg.get("deployment", {}).get("min_map50_readiness", 0.40)
    export_model_evaluation_reports(
        model_name=target_model,
        dataset_info=dataset_info,
        overall_metrics=overall,
        per_class_metrics=per_class,
        output_txt=rep_txt,
        output_json=rep_json,
        readiness_threshold=threshold,
        target_metric_dirs=[metrics_primary_dir, metrics_results_dir]
    )

    print(f"\n[+] Validation metrics successfully exported to:")
    print(f"     - metrics/ (overall_metrics.csv, per_class_metrics.csv, iou_results.csv, metrics.json, evaluation_summary.md, plots)")
    print(f"     - results/metrics/ (iou_results.csv, metrics.json, model_report.txt, plots)")
    print(f"     - results/iou_results.csv & results/evaluation_report.txt")
    print(f"     - {rep_txt.relative_to(PROJECT_ROOT)}")
    print(f"     - {rep_json.relative_to(PROJECT_ROOT)}")
    print("=" * 75 + "\n")
    return overall



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate YOLO26 Model")
    parser.add_argument("--model", type=str, default="", help="Model weight path")
    parser.add_argument("--split", type=str, default="val", help="Split: 'val' or 'test'")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="IoU threshold")
    args = parser.parse_args()

    run_validation(args.model, args.split, args.conf, args.iou)
