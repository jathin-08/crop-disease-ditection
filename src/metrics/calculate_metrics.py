"""
Tomato Disease AI - Core Metrics Calculator
Computes overall Precision, Recall, F1, mAP@50, and mAP@50-95.
"""

from typing import Dict, Any


def extract_overall_metrics(val_results: Any) -> Dict[str, float]:
    """Extracts overall validation metrics from Ultralytics results."""
    metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "map50": 0.0,
        "map50_95": 0.0
    }

    if hasattr(val_results, "results_dict"):
        rd = val_results.results_dict
        p = float(rd.get("metrics/precision(B)", 0.0))
        r = float(rd.get("metrics/recall(B)", 0.0))
        m50 = float(rd.get("metrics/mAP50(B)", 0.0))
        m50_95 = float(rd.get("metrics/mAP50-95(B)", 0.0))

        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        metrics["precision"] = p
        metrics["recall"] = r
        metrics["f1"] = f1
        metrics["map50"] = m50
        metrics["map50_95"] = m50_95
    elif hasattr(val_results, "box"):
        box = val_results.box
        p = float(box.mp) if hasattr(box, "mp") else 0.0
        r = float(box.mr) if hasattr(box, "mr") else 0.0
        m50 = float(box.map50) if hasattr(box, "map50") else 0.0
        m50_95 = float(box.map) if hasattr(box, "map") else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        metrics["precision"] = p
        metrics["recall"] = r
        metrics["f1"] = f1
        metrics["map50"] = m50
        metrics["map50_95"] = m50_95

    return metrics
