"""
Tomato Disease AI - Central Metric Engine
==============================================================================
Consolidated metric engine providing:
  - calculate_iou(pred_box, gt_box)
  - calculate_precision(tp, fp)
  - calculate_recall(tp, fn)
  - calculate_f1(precision, recall)
  - match_predictions_to_ground_truth(predictions, ground_truths)
  - evaluate_image(predictions, ground_truths)
  - evaluate_dataset(predictions, ground_truths)
  - load_metrics_report()
==============================================================================
"""

from typing import Dict, Any, List, Optional, Sequence, Union
from pathlib import Path

from src.metrics.iou_metrics import (
    calculate_iou,
    calculate_iou_matrix,
    calculate_precision,
    calculate_recall,
    calculate_f1,
    match_predictions_to_ground_truth,
    calculate_mean_iou,
    calculate_per_class_iou,
    calculate_iou_thresholds,
    calculate_iou_summary,
    load_ground_truths_from_label_file,
    export_iou_csv,
    generate_iou_bar_chart,
    xywh_to_xyxy,
    xyxy_to_xywh
)
from src.metrics.evaluation import (
    evaluate_image,
    evaluate_dataset
)
from src.metrics.metrics_loader import (
    load_metrics_report,
    get_default_model_metrics
)

__all__ = [
    "calculate_iou",
    "calculate_iou_matrix",
    "calculate_precision",
    "calculate_recall",
    "calculate_f1",
    "match_predictions_to_ground_truth",
    "evaluate_image",
    "evaluate_dataset",
    "calculate_mean_iou",
    "calculate_per_class_iou",
    "calculate_iou_thresholds",
    "calculate_iou_summary",
    "load_ground_truths_from_label_file",
    "export_iou_csv",
    "generate_iou_bar_chart",
    "load_metrics_report",
    "get_default_model_metrics",
    "xywh_to_xyxy",
    "xyxy_to_xywh"
]
