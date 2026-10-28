"""Tomato Disease AI - Metrics Calculation & Reporting Package"""

from src.metrics.calculate_metrics import extract_overall_metrics
from src.metrics.per_class_metrics import compute_per_class_metrics
from src.metrics.confusion_matrix import plot_confusion_matrix
from src.metrics.generate_report import export_model_evaluation_reports
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
    evaluate_iou,
    export_iou_csv,
    generate_iou_bar_chart
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
    "extract_overall_metrics",
    "compute_per_class_metrics",
    "plot_confusion_matrix",
    "export_model_evaluation_reports",
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
    "load_metrics_report",
    "get_default_model_metrics",
    "evaluate_iou",
    "export_iou_csv",
    "generate_iou_bar_chart"
]
