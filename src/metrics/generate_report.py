import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd


def export_model_evaluation_reports(
    model_name: str,
    dataset_info: Dict[str, Any],
    overall_metrics: Dict[str, float],
    per_class_metrics: List[Dict[str, Any]],
    output_txt: Path,
    output_json: Path,
    readiness_threshold: float = 0.40,
    target_metric_dirs: Optional[List[Path]] = None
) -> Dict[str, Any]:
    """Generates structured JSON, formatted plain-text, Markdown, and CSV reports across metrics directories."""
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    map50 = overall_metrics.get("map50", 0.0)
    is_ready = map50 >= readiness_threshold
    readiness_status = "READY" if is_ready else "NOT READY"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    iou_50_pct = overall_metrics.get("iou_gte_50_pct", 0.0)
    iou_75_pct = overall_metrics.get("iou_gte_75_pct", 0.0)

    data = {
        "precision": float(overall_metrics.get("precision", 0.0)),
        "recall": float(overall_metrics.get("recall", 0.0)),
        "f1": float(overall_metrics.get("f1", 0.0)),
        "mean_iou": float(overall_metrics.get("mean_iou", 0.0)),
        "map50": float(overall_metrics.get("map50", 0.0)),
        "map50_95": float(overall_metrics.get("map50_95", 0.0)),
        "iou_50_rate": float(iou_50_pct / 100.0 if iou_50_pct > 1.0 else iou_50_pct),
        "iou_75_rate": float(iou_75_pct / 100.0 if iou_75_pct > 1.0 else iou_75_pct),
        "tp": int(overall_metrics.get("tp", overall_metrics.get("evaluated_detections", 0))),
        "fp": int(overall_metrics.get("fp", 0)),
        "fn": int(overall_metrics.get("fn", 0)),
        "timestamp": timestamp,
        "model": {
            "name": model_name,
            "architecture": "YOLO26",
            "task": "detect"
        },
        "dataset": dataset_info,
        "overall_performance": overall_metrics,
        "per_class_performance": per_class_metrics,
        "model_status": readiness_status,
        "readiness_threshold_map50": readiness_threshold
    }

    has_iou = "mean_iou" in overall_metrics

    # Save primary JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Format text report
    lines = [
        "=" * 75,
        " TOMATO LEAF DISEASE AI - COMPREHENSIVE MODEL EVALUATION REPORT",
        "=" * 75,
        f"Evaluation Timestamp : {timestamp}",
        "",
        "MODEL INFORMATION",
        f"  Model Architecture : YOLO26 (Transfer Learning)",
        f"  Weights Evaluated  : {model_name}",
        f"  Primary Task       : YOLO26 Object Detection (Localization + Classification)",
        "",
        "DATASET INFORMATION",
        f"  Dataset Directory  : {dataset_info.get('path', 'dataset/')}",
        f"  Number of Classes  : {dataset_info.get('num_classes', 10)}",
        f"  Training Images    : {dataset_info.get('train_images', 0)}",
        f"  Validation Images  : {dataset_info.get('val_images', 0)}",
        f"  Test Images        : {dataset_info.get('test_images', 0)}",
        "-" * 75,
        "OVERALL PERFORMANCE (ACTUAL MEASURED VALUES)",
        f"  Precision (P)      : {overall_metrics.get('precision', 0.0):.4f} ({overall_metrics.get('precision', 0.0)*100:.2f}%)",
        f"  Recall (R)         : {overall_metrics.get('recall', 0.0):.4f} ({overall_metrics.get('recall', 0.0)*100:.2f}%)",
        f"  F1-Score           : {overall_metrics.get('f1', 0.0):.4f} ({overall_metrics.get('f1', 0.0)*100:.2f}%)",
        f"  mAP @ 0.50         : {overall_metrics.get('map50', 0.0):.4f} ({overall_metrics.get('map50', 0.0)*100:.2f}%)",
        f"  mAP @ 0.50:0.95    : {overall_metrics.get('map50_95', 0.0):.4f} ({overall_metrics.get('map50_95', 0.0)*100:.2f}%)",
    ]

    if has_iou:
        lines.extend([
            f"  Mean IoU           : {overall_metrics.get('mean_iou', 0.0):.4f} ({overall_metrics.get('mean_iou', 0.0)*100:.2f}%)",
            f"  IoU >= 0.50        : {overall_metrics.get('iou_gte_50_pct', 0.0):.2f}% detections",
            f"  IoU >= 0.75        : {overall_metrics.get('iou_gte_75_pct', 0.0):.2f}% detections",
            f"  Min / Max IoU      : {overall_metrics.get('min_iou', 0.0):.4f} / {overall_metrics.get('max_iou', 0.0):.4f}",
            f"  Evaluated Detections: {overall_metrics.get('evaluated_detections', 0)}",
        ])

    lines.append("-" * 75)
    lines.append("PER-CLASS PERFORMANCE BREAKDOWN")
    if has_iou:
        lines.append(f"  {'Disease Name':<30} | {'P':<6} | {'R':<6} | {'F1':<6} | {'AP50':<6} | {'AP95':<6} | {'Mean IoU':<9} | {'IoU>=.50'}")
        lines.append("  " + "-" * 88)
        for c in per_class_metrics:
            m_iou_str = f"{c.get('mean_iou', 0.0):.3f}" if "mean_iou" in c else "N/A"
            iou50_str = f"{c.get('iou_gte_50_pct', 0.0):.1f}%" if "iou_gte_50_pct" in c else "N/A"
            lines.append(
                f"  {c['disease_name']:<30} | {c['precision']:<6.3f} | {c['recall']:<6.3f} | {c['f1']:<6.3f} | {c['ap50']:<6.3f} | {c.get('ap50_95', 0.0):<6.3f} | {m_iou_str:<9} | {iou50_str}"
            )
    else:
        lines.append(f"  {'Disease Name':<32} | {'P':<7} | {'R':<7} | {'F1':<7} | {'AP50':<7} | {'AP50-95'}")
        lines.append("  " + "-" * 71)
        for c in per_class_metrics:
            lines.append(
                f"  {c['disease_name']:<32} | {c['precision']:<7.3f} | {c['recall']:<7.3f} | {c['f1']:<7.3f} | {c['ap50']:<7.3f} | {c['ap50_95']:.3f}"
            )

    lines.extend([
        "-" * 75,
        "MODEL READINESS ASSESSMENT",
        f"  Current Model Status : {readiness_status}",
        f"  Readiness Criteria   : Validation mAP@50 >= {readiness_threshold*100:.1f}%",
        f"  Achieved mAP@50      : {overall_metrics.get('map50', 0.0)*100:.2f}%",
        "=" * 75 + "\n"
    ])

    report_text = "\n".join(lines)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Format Markdown summary
    md_lines = [
        "# 🍅 Tomato Leaf Disease Model Evaluation Summary",
        f"**Timestamp:** `{timestamp}` | **Model:** `{Path(model_name).name}` | **Status:** `{readiness_status}`",
        "",
        "## Overall Model Performance",
        "| Metric | Score | Percentage |",
        "| :--- | :---: | :---: |",
        f"| **mAP @ 0.50** | `{overall_metrics.get('map50', 0.0):.4f}` | **{overall_metrics.get('map50', 0.0)*100:.2f}%** |",
        f"| **mAP @ 0.50:0.95** | `{overall_metrics.get('map50_95', 0.0):.4f}` | **{overall_metrics.get('map50_95', 0.0)*100:.2f}%** |",
        f"| **Precision (P)** | `{overall_metrics.get('precision', 0.0):.4f}` | **{overall_metrics.get('precision', 0.0)*100:.2f}%** |",
        f"| **Recall (R)** | `{overall_metrics.get('recall', 0.0):.4f}` | **{overall_metrics.get('recall', 0.0)*100:.2f}%** |",
        f"| **F1-Score** | `{overall_metrics.get('f1', 0.0):.4f}` | **{overall_metrics.get('f1', 0.0)*100:.2f}%** |",
    ]
    if has_iou:
        md_lines.extend([
            f"| **Mean IoU** | `{overall_metrics.get('mean_iou', 0.0):.4f}` | **{overall_metrics.get('mean_iou', 0.0)*100:.2f}%** |",
            f"| **IoU >= 0.50** | `{overall_metrics.get('iou_gte_50_pct', 0.0):.2f}%` | **{overall_metrics.get('iou_gte_50_pct', 0.0):.2f}%** |",
            f"| **IoU >= 0.75** | `{overall_metrics.get('iou_gte_75_pct', 0.0):.2f}%` | **{overall_metrics.get('iou_gte_75_pct', 0.0):.2f}%** |",
            f"| **Evaluated Detections** | `{overall_metrics.get('evaluated_detections', 0)}` | - |",
        ])

    md_lines.extend([
        "",
        "## Per-Class Disease Performance",
    ])
    if has_iou:
        md_lines.append("| Class ID | Disease Name | Precision | Recall | F1-Score | mAP@50 | mAP@50:95 | Mean IoU | IoU>=0.50 |")
        md_lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for c in per_class_metrics:
            m_iou_str = f"{c.get('mean_iou', 0.0):.4f}" if "mean_iou" in c else "-"
            iou50_str = f"{c.get('iou_gte_50_pct', 0.0):.1f}%" if "iou_gte_50_pct" in c else "-"
            md_lines.append(
                f"| {c['class_id']} | **{c['disease_name']}** | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} | {c['ap50']:.4f} | {c.get('ap50_95', 0.0):.4f} | {m_iou_str} | {iou50_str} |"
            )
    else:
        md_lines.append("| Class ID | Disease Name | Precision | Recall | F1-Score | mAP@50 | mAP@50:95 |")
        md_lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |")
        for c in per_class_metrics:
            md_lines.append(
                f"| {c['class_id']} | **{c['disease_name']}** | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} | {c['ap50']:.4f} | {c['ap50_95']:.4f} |"
            )
    md_report = "\n".join(md_lines) + "\n"

    # Export to additional metrics directories (e.g. metrics/ and results/metrics/)
    all_target_dirs = []
    if target_metric_dirs:
        all_target_dirs.extend(target_metric_dirs)

    # Per-class DataFrame
    df_per_class = pd.DataFrame(per_class_metrics)
    # Overall DataFrame
    df_overall = pd.DataFrame([overall_metrics])

    for m_dir in all_target_dirs:
        m_dir.mkdir(parents=True, exist_ok=True)
        # 1. JSON
        with open(m_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        # 2. TXT Report
        with open(m_dir / "model_report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)
        # 3. Markdown
        with open(m_dir / "evaluation_summary.md", "w", encoding="utf-8") as f:
            f.write(md_report)
        # 4. CSVs
        df_per_class.to_csv(m_dir / "per_class_metrics.csv", index=False)
        df_overall.to_csv(m_dir / "overall_metrics.csv", index=False)

    # Also save evaluation_report.txt and metrics.json directly in results/ for standard access
    project_results_dir = output_txt.parent.parent
    if project_results_dir.name == "results":
        with open(project_results_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)
        with open(project_results_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    return data

