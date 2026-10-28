"""
Tomato Disease AI - Per-Class Metrics Calculator
Computes Precision, Recall, F1-Score, AP50, and AP50-95 per tomato disease.
"""

from typing import Dict, Any, List
import numpy as np


def compute_per_class_metrics(
    val_results: Any,
    class_names: Dict[int, str]
) -> List[Dict[str, Any]]:
    """
    Extracts per-class metrics from Ultralytics validation results object.
    """
    per_class_data = []

    # Check if results object has box metrics
    if hasattr(val_results, "box"):
        box = val_results.box
        # Ultralytics maps
        ap50 = box.ap50 if hasattr(box, "ap50") and box.ap50 is not None else []
        ap = box.ap if hasattr(box, "ap") and box.ap is not None else []
        p = box.p if hasattr(box, "p") and box.p is not None else []
        r = box.r if hasattr(box, "r") and box.r is not None else []
        f1 = box.f1 if hasattr(box, "f1") and box.f1 is not None else []

        for cid, cname in class_names.items():
            class_p = float(p[cid]) if cid < len(p) else 0.0
            class_r = float(r[cid]) if cid < len(r) else 0.0
            class_f1 = float(f1[cid]) if cid < len(f1) else (
                2 * (class_p * class_r) / (class_p + class_r) if (class_p + class_r) > 0 else 0.0
            )
            class_ap50 = float(ap50[cid]) if cid < len(ap50) else 0.0
            class_ap = float(ap[cid]) if cid < len(ap) else 0.0

            per_class_data.append({
                "class_id": cid,
                "disease_name": cname,
                "precision": class_p,
                "recall": class_r,
                "f1": class_f1,
                "ap50": class_ap50,
                "ap50_95": class_ap
            })
    else:
        # Fallback if box is not available
        for cid, cname in class_names.items():
            per_class_data.append({
                "class_id": cid,
                "disease_name": cname,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "ap50": 0.0,
                "ap50_95": 0.0
            })

    return per_class_data
