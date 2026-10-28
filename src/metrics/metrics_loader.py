"""
Tomato Disease AI - Metrics Loader
==============================================================================
Loads calculated metrics from reports/metrics.json using project-relative paths.
Ensures seamless portability across local environments and cloud web deployments.
==============================================================================
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

# Resolve project root dynamically using pathlib (Section 16 - No hardcoded OS paths)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_default_model_metrics() -> Dict[str, float]:
    """Fallback baseline model metrics if evaluation has not yet been executed."""
    return {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "mean_iou": 0.0,
        "map50": 0.0,
        "map50_95": 0.0,
        "iou_50_rate": 0.0,
        "iou_75_rate": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 0
    }


def load_metrics_report(metrics_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Locates and reads the latest REAL model evaluation metrics from metrics.json.
    Searches standard project-relative locations.
    """
    candidate_paths = [
        metrics_path,
        PROJECT_ROOT / "results" / "reports" / "metrics.json",
        PROJECT_ROOT / "results" / "metrics.json",
        PROJECT_ROOT / "results" / "metrics" / "metrics.json",
        PROJECT_ROOT / "metrics" / "metrics.json"
    ]

    for p in candidate_paths:
        if p is not None and Path(p).exists() and Path(p).is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Ensure top-level metric fields exist
                overall = data.get("overall_performance", {})
                precision = data.get("precision", overall.get("precision", 0.0))
                recall = data.get("recall", overall.get("recall", 0.0))
                f1 = data.get("f1", overall.get("f1", 0.0))
                mean_iou = data.get("mean_iou", overall.get("mean_iou", 0.0))
                map50 = data.get("map50", overall.get("map50", 0.0))
                map50_95 = data.get("map50_95", overall.get("map50_95", 0.0))
                tp = data.get("tp", overall.get("tp", 0))
                fp = data.get("fp", overall.get("fp", 0))
                fn = data.get("fn", overall.get("fn", 0))
                iou_50_rate = data.get("iou_50_rate", overall.get("iou_gte_50_pct", 0.0) / 100.0 if "iou_gte_50_pct" in overall else 0.0)
                iou_75_rate = data.get("iou_75_rate", overall.get("iou_gte_75_pct", 0.0) / 100.0 if "iou_gte_75_pct" in overall else 0.0)

                data["precision"] = float(precision)
                data["recall"] = float(recall)
                data["f1"] = float(f1)
                data["mean_iou"] = float(mean_iou)
                data["map50"] = float(map50)
                data["map50_95"] = float(map50_95)
                data["tp"] = int(tp)
                data["fp"] = int(fp)
                data["fn"] = int(fn)
                data["iou_50_rate"] = float(iou_50_rate)
                data["iou_75_rate"] = float(iou_75_rate)

                return data
            except Exception as e:
                print(f"[!] Warning: Failed to read {p}: {e}")

    # Fallback if no report found yet
    defaults = get_default_model_metrics()
    return {
        **defaults,
        "overall_performance": defaults,
        "per_class_performance": [],
        "model_status": "NOT READY"
    }
