"""
Tomato Disease AI - IoU (Intersection over Union) Evaluation Metric Suite
==============================================================================
Provides robust, class-aware IoU calculation, bounding box matching,
per-class and threshold-based evaluation, and result export pipelines
for the YOLO26 Tomato Leaf Disease Detection system.
==============================================================================
"""

import os
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Sequence, Union
import numpy as np
import cv2

# Project root helper
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def xywh_to_xyxy(box: Sequence[float]) -> List[float]:
    """
    Converts a bounding box from YOLO format [x_center, y_center, width, height]
    to corner format [x1, y1, x2, y2]. Works for both normalized and pixel coordinates.
    """
    xc, yc, w, h = box[0], box[1], box[2], box[3]
    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0
    return [float(x1), float(y1), float(x2), float(y2)]


def xyxy_to_xywh(box: Sequence[float]) -> List[float]:
    """
    Converts a bounding box from corner format [x1, y1, x2, y2]
    to YOLO format [x_center, y_center, width, height].
    """
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    xc = x1 + w / 2.0
    yc = y1 + h / 2.0
    return [float(xc), float(yc), float(w), float(h)]


def calculate_iou(
    box1: Sequence[float],
    box2: Sequence[float],
    box_format: str = "xyxy"
) -> float:
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.

    Formula:
        IoU = Area of Intersection / Area of Union
        Value is strictly bounded between [0.0, 1.0].

    Parameters:
        box1: First bounding box coordinates (e.g. ground truth).
        box2: Second bounding box coordinates (e.g. prediction).
        box_format: 'xyxy' for [x1, y1, x2, y2] or 'xywh' for [xc, yc, w, h].

    Returns:
        float: IoU value in range [0.0, 1.0].
    """
    if box1 is None or box2 is None or len(box1) < 4 or len(box2) < 4:
        return 0.0

    # Convert from xywh to xyxy if required
    fmt = box_format.lower().strip()
    if fmt == "xywh":
        b1 = xywh_to_xyxy(box1)
        b2 = xywh_to_xyxy(box2)
    elif fmt == "xyxy":
        b1 = [float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3])]
        b2 = [float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3])]
    else:
        raise ValueError(f"Unsupported box_format: {box_format}. Choose 'xyxy' or 'xywh'.")

    # Ensure coordinates are ordered (x1 <= x2, y1 <= y2)
    b1_x1, b1_x2 = min(b1[0], b1[2]), max(b1[0], b1[2])
    b1_y1, b1_y2 = min(b1[1], b1[3]), max(b1[1], b1[3])
    b2_x1, b2_x2 = min(b2[0], b2[2]), max(b2[0], b2[2])
    b2_y1, b2_y2 = min(b2[1], b2[3]), max(b2[1], b2[3])

    # Calculate intersection coordinates
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    # Calculate intersection dimensions
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # If there is no overlap, return 0.0 immediately
    if inter_area <= 0.0:
        return 0.0

    # Calculate areas of individual boxes
    area1 = max(0.0, b1_x2 - b1_x1) * max(0.0, b1_y2 - b1_y1)
    area2 = max(0.0, b2_x2 - b2_x1) * max(0.0, b2_y2 - b2_y1)

    # Calculate union area
    union_area = area1 + area2 - inter_area

    # Avoid division-by-zero errors
    if union_area <= 0.0:
        return 0.0

    iou = inter_area / union_area

    # Clip to valid mathematical range [0.0, 1.0] to guard against floating-point precision artifacts
    return float(np.clip(iou, 0.0, 1.0))


def calculate_iou_matrix(
    boxes1: Sequence[Sequence[float]],
    boxes2: Sequence[Sequence[float]],
    box_format: str = "xyxy"
) -> np.ndarray:
    """
    Computes pairwise IoU between N boxes in boxes1 and M boxes in boxes2.

    Returns:
        np.ndarray of shape (N, M) containing IoU values in [0.0, 1.0].
    """
    n = len(boxes1)
    m = len(boxes2)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)

    iou_mat = np.zeros((n, m), dtype=np.float32)
    for i, b1 in enumerate(boxes1):
        for j, b2 in enumerate(boxes2):
            iou_mat[i, j] = calculate_iou(b1, b2, box_format=box_format)
    return iou_mat


def calculate_precision(tp: Union[int, float], fp: Union[int, float]) -> float:
    """
    Calculates Precision: TP / (TP + FP).
    Handles division by zero safely.
    """
    total = float(tp) + float(fp)
    return float(tp / total) if total > 0.0 else 0.0


def calculate_recall(tp: Union[int, float], fn: Union[int, float]) -> float:
    """
    Calculates Recall: TP / (TP + FN).
    Handles division by zero safely.
    """
    total = float(tp) + float(fn)
    return float(tp / total) if total > 0.0 else 0.0


def calculate_f1(precision: float, recall: float) -> float:
    """
    Calculates F1 Score: 2 * Precision * Recall / (Precision + Recall).
    Handles division by zero safely.
    """
    total = float(precision) + float(recall)
    if total <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / total)


def match_predictions_to_ground_truth(
    arg1: Optional[List[Dict[str, Any]]] = None,
    arg2: Optional[List[Dict[str, Any]]] = None,
    min_iou: float = 0.0,
    class_names: Optional[Dict[int, str]] = None,
    box_format: str = "xyxy",
    *,
    ground_truths: Optional[List[Dict[str, Any]]] = None,
    predictions: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Matches predicted bounding boxes with ground-truth boxes using:
      1. Class-aware matching: Only boxes of identical disease class are eligible.
      2. Highest IoU matching: Candidate pairings are prioritized by descending IoU.
      3. One-to-one matching: Each ground-truth box is paired with at most one prediction,
         and each prediction is paired with at most one ground-truth box.

    Supports both parameter orderings:
      - match_predictions_to_ground_truth(predictions, ground_truths)
      - match_predictions_to_ground_truth(ground_truths, predictions)
      - keyword arguments: predictions=..., ground_truths=...
    """
    # Resolve ground_truths and predictions
    if ground_truths is None or predictions is None:
        list1 = arg1 if arg1 is not None else []
        list2 = arg2 if arg2 is not None else []
        
        # Check if list1 contains predictions (e.g. has 'confidence') while list2 does not
        list1_has_conf = any("confidence" in item for item in list1 if isinstance(item, dict))
        list2_has_conf = any("confidence" in item for item in list2 if isinstance(item, dict))
        
        if list1_has_conf and not list2_has_conf:
            predictions = list1
            ground_truths = list2
        elif list2_has_conf and not list1_has_conf:
            ground_truths = list1
            predictions = list2
        else:
            # Default to legacy positional order: arg1=ground_truths, arg2=predictions
            # unless kwargs specifies
            ground_truths = kwargs.get("ground_truths", list1)
            predictions = kwargs.get("predictions", list2)

    ground_truths = ground_truths or []
    predictions = predictions or []
    class_names = class_names or {}
    matched_pairs: List[Dict[str, Any]] = []
    unmatched_predictions: List[Dict[str, Any]] = []
    unmatched_ground_truths: List[Dict[str, Any]] = []

    # Helper to get box coords from 'bbox' or 'box'
    def _extract_box(item: Dict[str, Any]) -> List[float]:
        if "bbox" in item and item["bbox"] is not None:
            return item["bbox"]
        return item.get("box", [])

    # Group ground truths and predictions by class_id
    gt_by_class: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
    for original_idx, gt in enumerate(ground_truths):
        cid = int(gt.get("class_id", -1))
        gt_by_class.setdefault(cid, []).append((original_idx, gt))

    pred_by_class: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
    for original_idx, pred in enumerate(predictions):
        cid = int(pred.get("class_id", -1))
        pred_by_class.setdefault(cid, []).append((original_idx, pred))

    all_matched_gt_indices = set()
    all_matched_pred_indices = set()

    all_classes = set(gt_by_class.keys()).union(set(pred_by_class.keys()))

    for cid in all_classes:
        c_gts = gt_by_class.get(cid, [])
        c_preds = pred_by_class.get(cid, [])

        if not c_gts or not c_preds:
            continue

        cname = class_names.get(cid, f"Class_{cid}")

        # Compute candidate pairs and pairwise IoU
        candidates = []
        for p_idx, (orig_p_idx, p) in enumerate(c_preds):
            p_box = _extract_box(p)
            p_conf = float(p.get("confidence", 1.0))
            for g_idx, (orig_g_idx, g) in enumerate(c_gts):
                g_box = _extract_box(g)
                iou_val = calculate_iou(g_box, p_box, box_format=box_format)
                if iou_val > 0.0 and iou_val >= min_iou:
                    # Sort primarily by IoU descending, secondarily by prediction confidence
                    candidates.append((iou_val, p_conf, p_idx, g_idx, orig_p_idx, orig_g_idx))

        # Sort candidate pairs by IoU descending
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

        matched_p_set = set()
        matched_g_set = set()

        for iou_val, p_conf, p_idx, g_idx, orig_p_idx, orig_g_idx in candidates:
            if p_idx not in matched_p_set and g_idx not in matched_g_set:
                matched_p_set.add(p_idx)
                matched_g_set.add(g_idx)
                all_matched_pred_indices.add(orig_p_idx)
                all_matched_gt_indices.add(orig_g_idx)

                matched_pairs.append({
                    "class_id": cid,
                    "class_name": cname,
                    "ground_truth_box": [round(float(coord), 4) for coord in _extract_box(c_gts[g_idx][1])],
                    "predicted_box": [round(float(coord), 4) for coord in _extract_box(c_preds[p_idx][1])],
                    "confidence": float(p_conf),
                    "iou": float(iou_val)
                })

    # Collect unmatched predictions
    for orig_idx, pred in enumerate(predictions):
        if orig_idx not in all_matched_pred_indices:
            cid = int(pred.get("class_id", -1))
            unmatched_predictions.append({
                "class_id": cid,
                "class_name": class_names.get(cid, f"Class_{cid}"),
                "predicted_box": [round(float(coord), 4) for coord in _extract_box(pred)],
                "confidence": float(pred.get("confidence", 0.0))
            })

    # Collect unmatched ground truths
    for orig_idx, gt in enumerate(ground_truths):
        if orig_idx not in all_matched_gt_indices:
            cid = int(gt.get("class_id", -1))
            unmatched_ground_truths.append({
                "class_id": cid,
                "class_name": class_names.get(cid, f"Class_{cid}"),
                "ground_truth_box": [round(float(coord), 4) for coord in _extract_box(gt)]
            })

    return {
        "matched_pairs": matched_pairs,
        "unmatched_predictions": unmatched_predictions,
        "unmatched_ground_truths": unmatched_ground_truths
    }


def calculate_mean_iou(matches: List[Dict[str, Any]]) -> float:
    """Calculates Mean IoU across all matched detections."""
    if not matches:
        return 0.0
    ious = [float(m["iou"]) for m in matches if "iou" in m]
    return float(np.mean(ious)) if ious else 0.0


def calculate_iou_thresholds(
    matches: List[Dict[str, Any]],
    thresholds: Optional[List[float]] = None
) -> Dict[float, Dict[str, Any]]:
    """
    Computes count and percentage of detections that satisfy given IoU thresholds (e.g. >= 0.50, >= 0.75).
    """
    thresholds = thresholds or [0.50, 0.75]
    total = len(matches)
    results = {}

    for t in thresholds:
        count = sum(1 for m in matches if float(m.get("iou", 0.0)) >= t)
        pct = (count / total * 100.0) if total > 0 else 0.0
        results[t] = {
            "count": int(count),
            "percentage": float(pct),
            "total": int(total)
        }
    return results


def calculate_per_class_iou(
    matches: List[Dict[str, Any]],
    class_names: Dict[int, str]
) -> List[Dict[str, Any]]:
    """
    Calculates Mean IoU, Min IoU, Max IoU, and threshold fulfillment separately
    for each tomato disease class.
    """
    per_class_data: List[Dict[str, Any]] = []

    # Group matches by class_id
    matches_by_class: Dict[int, List[float]] = {cid: [] for cid in class_names.keys()}
    for m in matches:
        cid = int(m.get("class_id", -1))
        if cid in matches_by_class:
            matches_by_class[cid].append(float(m.get("iou", 0.0)))
        else:
            matches_by_class.setdefault(cid, []).append(float(m.get("iou", 0.0)))

    for cid, cname in sorted(class_names.items(), key=lambda x: x[0]):
        ious = matches_by_class.get(cid, [])
        count = len(ious)
        if count > 0:
            mean_iou = float(np.mean(ious))
            min_iou = float(np.min(ious))
            max_iou = float(np.max(ious))
            c_gte_50 = sum(1 for v in ious if v >= 0.50)
            c_gte_75 = sum(1 for v in ious if v >= 0.75)
            pct_50 = (c_gte_50 / count) * 100.0
            pct_75 = (c_gte_75 / count) * 100.0
        else:
            mean_iou = 0.0
            min_iou = 0.0
            max_iou = 0.0
            c_gte_50 = 0
            c_gte_75 = 0
            pct_50 = 0.0
            pct_75 = 0.0

        per_class_data.append({
            "class_id": int(cid),
            "disease_name": cname,
            "evaluated_detections": int(count),
            "mean_iou": float(mean_iou),
            "min_iou": float(min_iou),
            "max_iou": float(max_iou),
            "iou_gte_50_count": int(c_gte_50),
            "iou_gte_50_pct": float(pct_50),
            "iou_gte_75_count": int(c_gte_75),
            "iou_gte_75_pct": float(pct_75)
        })

    return per_class_data


def calculate_iou_summary(
    matches: List[Dict[str, Any]],
    class_names: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """
    Computes a comprehensive IoU summary dictionary containing overall and per-class metrics.
    """
    class_names = class_names or {}
    total = len(matches)
    if total > 0:
        ious = [float(m["iou"]) for m in matches]
        mean_iou = float(np.mean(ious))
        min_iou = float(np.min(ious))
        max_iou = float(np.max(ious))
        count_50 = sum(1 for v in ious if v >= 0.50)
        pct_50 = (count_50 / total) * 100.0
        count_75 = sum(1 for v in ious if v >= 0.75)
        pct_75 = (count_75 / total) * 100.0
    else:
        mean_iou = 0.0
        min_iou = 0.0
        max_iou = 0.0
        count_50 = 0
        pct_50 = 0.0
        count_75 = 0
        pct_75 = 0.0

    per_class = calculate_per_class_iou(matches, class_names) if class_names else []

    return {
        "mean_iou": mean_iou,
        "mean_iou_pct": mean_iou * 100.0,
        "min_iou": min_iou,
        "max_iou": max_iou,
        "evaluated_detections": total,
        "iou_gte_50_count": count_50,
        "iou_gte_50_pct": pct_50,
        "iou_gte_75_count": count_75,
        "iou_gte_75_pct": pct_75,
        "per_class_iou": per_class
    }


def load_ground_truths_from_label_file(
    label_path: Path,
    image_width: int,
    image_height: int
) -> List[Dict[str, Any]]:
    """
    Reads ground-truth annotations from a YOLO format .txt file and converts them
    to pixel coordinates [x1, y1, x2, y2].
    """
    if not label_path.exists():
        return []

    gts = []
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = max(0.0, (xc - bw / 2.0) * image_width)
                    y1 = max(0.0, (yc - bh / 2.0) * image_height)
                    x2 = min(float(image_width), (xc + bw / 2.0) * image_width)
                    y2 = min(float(image_height), (yc + bh / 2.0) * image_height)
                    gts.append({
                        "class_id": cid,
                        "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                        "box_normalized": [round(xc, 4), round(yc, 4), round(bw, 4), round(bh, 4)]
                    })
    except Exception as e:
        print(f"[!] Error parsing ground-truth file {label_path}: {e}")
    return gts


def evaluate_iou(
    model: Any,
    dataset_dir: Path,
    split: str = "val",
    conf: float = 0.001,
    iou_nms: float = 0.50,
    class_names: Optional[Dict[int, str]] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Executes a complete, dataset-wide IoU evaluation pipeline on the specified split (val or test).
    Measures overlap between actual ground-truth boxes and YOLO26 predictions.

    Parameters:
        model: Ultralytics YOLO model instance.
        dataset_dir: Root dataset directory containing images/{split} and labels/{split}.
        split: 'val' or 'test'.
        conf: Prediction confidence threshold (default 0.001 to evaluate all model detections).
        iou_nms: Non-maximum suppression threshold.
        class_names: Mapping of class_id -> name.
        device: '0' for CUDA or 'cpu'.

    Returns:
        Dict with overall summary, per-class metrics, and full match list.
    """
    class_names = class_names or {}
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split

    if not images_dir.exists():
        return {
            "error": f"Image directory does not exist: {images_dir}",
            "evaluated_images": 0,
            "mean_iou": 0.0,
            "evaluated_detections": 0,
            "matches": []
        }

    image_paths = sorted(
        [p for p in images_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
    )

    all_matches: List[Dict[str, Any]] = []
    all_unmatched_preds: List[Dict[str, Any]] = []
    all_unmatched_gts: List[Dict[str, Any]] = []
    evaluated_images = 0

    for img_p in image_paths:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]
        evaluated_images += 1

        # 1. Load Ground Truth annotations
        lbl_p = labels_dir / f"{img_p.stem}.txt"
        gts = load_ground_truths_from_label_file(lbl_p, image_width=w, image_height=h)

        # 2. Run Model Inference
        preds = []
        try:
            results = model.predict(
                source=img,
                conf=conf,
                iou=iou_nms,
                device=device,
                verbose=False
            )
            for r in results:
                for b in r.boxes:
                    cls_id = int(b.cls[0])
                    b_conf = float(b.conf[0])
                    xyxy = b.xyxy[0].cpu().numpy()
                    preds.append({
                        "class_id": cls_id,
                        "confidence": b_conf,
                        "box": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]
                    })
        except Exception as ex:
            print(f"[!] Inference failed on {img_p.name}: {ex}")

        # 3. Match predictions to ground truth
        match_res = match_predictions_to_ground_truth(
            ground_truths=gts,
            predictions=preds,
            min_iou=0.0,
            class_names=class_names,
            box_format="xyxy"
        )

        for m in match_res["matched_pairs"]:
            m["image_name"] = img_p.name
            all_matches.append(m)

        for up in match_res["unmatched_predictions"]:
            up["image_name"] = img_p.name
            all_unmatched_preds.append(up)

        for ug in match_res["unmatched_ground_truths"]:
            ug["image_name"] = img_p.name
            all_unmatched_gts.append(ug)

    # 4. Compute overall and per-class summary metrics
    summary = calculate_iou_summary(all_matches, class_names=class_names)
    summary["evaluated_images"] = evaluated_images
    summary["unmatched_predictions_count"] = len(all_unmatched_preds)
    summary["unmatched_ground_truths_count"] = len(all_unmatched_gts)
    summary["matches"] = all_matches

    return summary


def export_iou_csv(
    matches: List[Dict[str, Any]],
    output_path: Path
) -> Path:
    """
    Exports detailed matched IoU evaluation records to CSV.
    Columns: image_name, class_name, ground_truth_box, predicted_box, confidence, iou
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_name",
        "class_name",
        "ground_truth_box",
        "predicted_box",
        "confidence",
        "iou"
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in matches:
            writer.writerow({
                "image_name": m.get("image_name", ""),
                "class_name": m.get("class_name", ""),
                "ground_truth_box": str(m.get("ground_truth_box", [])),
                "predicted_box": str(m.get("predicted_box", [])),
                "confidence": f"{float(m.get('confidence', 0.0)):.4f}",
                "iou": f"{float(m.get('iou', 0.0)):.4f}"
            })
    return output_path


def generate_iou_bar_chart(
    per_class_iou: List[Dict[str, Any]],
    output_path: Path
) -> Optional[Path]:
    """
    Renders and exports a high-aesthetic bar chart of Per-Class Mean IoU.
    """
    if not per_class_iou:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_path.parent.mkdir(parents=True, exist_ok=True)

        names = [
            c["disease_name"].replace("tomato_", "").replace("_", " ").title()
            for c in per_class_iou
        ]
        ious = [c["mean_iou"] for c in per_class_iou]

        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        y_pos = np.arange(len(names))

        # Color palette
        colors = [
            "#22c55e" if v >= 0.85 else "#38bdf8" if v >= 0.75 else "#f59e0b" if v >= 0.60 else "#ef4444"
            for v in ious
        ]

        bars = ax.barh(y_pos, ious, color=colors, height=0.65, edgecolor="none", alpha=0.9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=10, fontweight="medium")
        ax.invert_yaxis()  # Labels read top-to-bottom

        ax.set_xlabel("Mean Intersection over Union (IoU)", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_title("Tomato Leaf Disease Detection - Per-Class Mean IoU Benchmark", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlim(0.0, 1.05)

        # Reference threshold lines
        ax.axvline(0.50, color="#f59e0b", linestyle="--", linewidth=1.2, alpha=0.7, label="IoU = 0.50 Threshold")
        ax.axvline(0.75, color="#22c55e", linestyle="--", linewidth=1.2, alpha=0.7, label="IoU = 0.75 Threshold")

        # Value annotations on bars
        for bar, val in zip(bars, ious):
            width = bar.get_width()
            ax.text(
                width + 0.015,
                bar.get_y() + bar.get_height() / 2.0,
                f"{val:.3f} ({val*100:.1f}%)",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#1e293b"
            )

        ax.grid(axis="x", linestyle=":", alpha=0.5)
        ax.legend(loc="lower right", frameon=True)
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path
    except Exception as e:
        print(f"[!] Warning: Failed to render IoU chart: {e}")
        return None
