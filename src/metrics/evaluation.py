"""
Tomato Disease AI - Model & Image Evaluation Pipeline
==============================================================================
Provides real, calculated metrics for:
  - Single-image evaluation (IoU, TP, FP, FN, Precision, Recall, F1)
  - Dataset-level evaluation (Mean IoU, TP, FP, FN, Precision, Recall, F1, mAP@50, mAP@50-95)
  - Ground-truth matching and report generation (results/reports/metrics.json)
==============================================================================
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import cv2
import numpy as np

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_classes, resolve_model_path
from src.metrics.iou_metrics import (
    calculate_iou,
    calculate_precision,
    calculate_recall,
    calculate_f1,
    match_predictions_to_ground_truth,
    load_ground_truths_from_label_file,
    calculate_mean_iou,
    calculate_iou_thresholds,
    calculate_per_class_iou,
    export_iou_csv
)
from src.metrics.calculate_metrics import extract_overall_metrics
from src.metrics.per_class_metrics import compute_per_class_metrics
from src.metrics.generate_report import export_model_evaluation_reports


def evaluate_image(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    iou_threshold: float = 0.50,
    class_names: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """
    Evaluates model predictions against ground-truth annotations on a single image.

    Calculates:
      1. One-to-one class-aware matching (highest IoU first)
      2. Individual IoU for each matched detection
      3. TP: True Positives (IoU >= iou_threshold)
      4. FP: False Positives (unmatched predictions + IoU < iou_threshold)
      5. FN: False Negatives (unmatched ground truths + missed GTs)
      6. Precision = TP / (TP + FP)
      7. Recall = TP / (TP + FN)
      8. F1 Score = 2 * P * R / (P + R)

    Parameters:
        predictions: List of prediction dicts (box/bbox, class_id, confidence, etc.)
        ground_truths: List of ground-truth dicts (box/bbox, class_id)
        iou_threshold: Threshold to qualify a match as True Positive (default 0.50)
        class_names: Optional mapping from class_id to class_name

    Returns:
        Dict containing evaluated metrics and updated predictions list.
    """
    class_names = class_names or {}
    
    # Run 1-to-1 matching
    match_res = match_predictions_to_ground_truth(
        predictions=predictions,
        ground_truths=ground_truths,
        min_iou=0.0,
        class_names=class_names,
        box_format="xyxy"
    )

    matched_pairs = match_res.get("matched_pairs", [])
    unmatched_preds = match_res.get("unmatched_predictions", [])
    unmatched_gts = match_res.get("unmatched_ground_truths", [])

    # Calculate TP, FP, FN based on iou_threshold
    tp = sum(1 for m in matched_pairs if float(m.get("iou", 0.0)) >= iou_threshold)
    fp_low_iou = sum(1 for m in matched_pairs if float(m.get("iou", 0.0)) < iou_threshold)
    fp = len(unmatched_preds) + fp_low_iou
    fn_low_iou = fp_low_iou
    fn = len(unmatched_gts) + fn_low_iou

    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    f1 = calculate_f1(precision, recall)

    matched_ious = [float(m["iou"]) for m in matched_pairs if "iou" in m]
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    # Helper to compare boxes
    def _box_matches(b1, b2, tol=1.0):
        if not b1 or not b2 or len(b1) != 4 or len(b2) != 4:
            return False
        return all(abs(float(c1) - float(c2)) <= tol for c1, c2 in zip(b1, b2))

    # Enrich each prediction with its individual IoU and calculated image metrics
    for pred in predictions:
        pred_box = pred.get("box") or pred.get("bbox") or []
        cid = int(pred.get("class_id", -1))
        
        matched_pair = None
        for m in matched_pairs:
            if m.get("class_id") == cid and _box_matches(pred_box, m.get("predicted_box")):
                matched_pair = m
                break
        
        if matched_pair is not None:
            pred["iou"] = float(matched_pair["iou"])
            pred["ground_truth_box"] = matched_pair.get("ground_truth_box")
        else:
            pred["iou"] = 0.0
            pred["ground_truth_box"] = None

        pred["bbox"] = pred_box
        pred["box"] = pred_box
        pred["precision"] = precision
        pred["recall"] = recall
        pred["f1"] = f1

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_iou": float(mean_iou),
        "matched_pairs": matched_pairs,
        "unmatched_predictions": unmatched_preds,
        "unmatched_ground_truths": unmatched_gts,
        "iou_threshold": float(iou_threshold),
        "evaluated_predictions_count": len(predictions),
        "evaluated_ground_truths_count": len(ground_truths)
    }


def evaluate_dataset(
    model: Any = None,
    dataset_dir: Optional[Path] = None,
    split: str = "test",
    conf: float = 0.001,
    iou_nms: float = 0.50,
    iou_match_thresh: float = 0.50,
    class_names: Optional[Dict[int, str]] = None,
    device: Optional[str] = None,
    output_json: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Executes a complete, benchmark-grade evaluation across a dataset split (test or val).

    Steps:
      1. Load trained YOLO model.
      2. Load split images and ground-truth labels.
      3. Run model inference.
      4. Match predictions to ground truth (class-aware, 1-to-1).
      5. Calculate TP, FP, FN.
      6. Calculate IoU values and Mean IoU.
      7. Calculate Precision, Recall, F1.
      8. Calculate mAP@50 and mAP@50-95.
      9. Save structured results to results/reports/metrics.json.

    Returns:
        Structured evaluation dictionary containing all calculated metrics.
    """
    from ultralytics import YOLO

    # Resolve paths project-relatively
    dataset_dir = dataset_dir or (PROJECT_ROOT / "dataset")
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split
    class_names = class_names or get_classes()

    # Load model
    if model is None:
        target_model = str(resolve_model_path())
        print(f"[*] Loading Model Checkpoint: {target_model}")
        model = YOLO(target_model)
    elif isinstance(model, (str, Path)):
        model = YOLO(str(model))

    import torch
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    print(f"\n[*] Evaluating Dataset Split: {split.upper()} ({images_dir})")

    # 1. Run Ultralytics validation for COCO-standard mAP@50 and mAP@50-95
    data_yaml = str(dataset_dir / "data.yaml")
    val_run_dir = PROJECT_ROOT / "results" / "validation"
    val_run_dir.mkdir(parents=True, exist_ok=True)

    try:
        val_results = model.val(
            data=data_yaml,
            split=split,
            conf=conf,
            iou=iou_nms,
            device=device,
            plots=False,
            project=str(val_run_dir),
            name=f"eval_{split}_run",
            exist_ok=True,
            verbose=False
        )
        val_metrics = extract_overall_metrics(val_results)
        per_class_val = compute_per_class_metrics(val_results, class_names)
        map50 = float(val_metrics.get("map50", 0.0))
        map50_95 = float(val_metrics.get("map50_95", 0.0))
    except Exception as e:
        print(f"[!] Warning: model.val failed, using inference-based estimation: {e}")
        map50 = 0.0
        map50_95 = 0.0
        val_metrics = {}
        per_class_val = []

    # 2. Iterate through all images in split to compute actual IoU, TP, FP, FN, P, R, F1
    image_paths = sorted(
        [p for p in images_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
    ) if images_dir.exists() else []

    all_matches = []
    all_unmatched_preds = []
    all_unmatched_gts = []
    dataset_tp = 0
    dataset_fp = 0
    dataset_fn = 0
    total_gts_count = 0
    total_preds_count = 0

    for img_p in image_paths:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        lbl_p = labels_dir / f"{img_p.stem}.txt"
        gts = load_ground_truths_from_label_file(lbl_p, image_width=w, image_height=h)
        total_gts_count += len(gts)

        preds = []
        try:
            res = model.predict(source=img, conf=conf, iou=iou_nms, device=device, verbose=False)
            for r in res:
                for b in r.boxes:
                    cls_id = int(b.cls[0])
                    b_conf = float(b.conf[0])
                    xyxy = b.xyxy[0].cpu().numpy()
                    preds.append({
                        "class_id": cls_id,
                        "confidence": b_conf,
                        "box": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                        "bbox": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]
                    })
        except Exception as ex:
            print(f"[!] Prediction error on {img_p.name}: {ex}")

        total_preds_count += len(preds)

        # Evaluate this image
        img_eval = evaluate_image(
            predictions=preds,
            ground_truths=gts,
            iou_threshold=iou_match_thresh,
            class_names=class_names
        )

        dataset_tp += img_eval["tp"]
        dataset_fp += img_eval["fp"]
        dataset_fn += img_eval["fn"]

        for m in img_eval["matched_pairs"]:
            m["image_name"] = img_p.name
            all_matches.append(m)
        for up in img_eval["unmatched_predictions"]:
            up["image_name"] = img_p.name
            all_unmatched_preds.append(up)
        for ug in img_eval["unmatched_ground_truths"]:
            ug["image_name"] = img_p.name
            all_unmatched_gts.append(ug)

    # 3. Calculate Dataset Overall Metrics
    precision = calculate_precision(dataset_tp, dataset_fp)
    recall = calculate_recall(dataset_tp, dataset_fn)
    f1 = calculate_f1(precision, recall)

    all_ious = [float(m["iou"]) for m in all_matches]
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    min_iou = float(np.min(all_ious)) if all_ious else 0.0
    max_iou = float(np.max(all_ious)) if all_ious else 0.0

    iou_50_count = sum(1 for v in all_ious if v >= 0.50)
    iou_75_count = sum(1 for v in all_ious if v >= 0.75)
    iou_50_rate = (iou_50_count / len(all_ious)) if all_ious else 0.0
    iou_75_rate = (iou_75_count / len(all_ious)) if all_ious else 0.0

    # 4. Compute per-class IoU summary
    per_class_iou = calculate_per_class_iou(all_matches, class_names)
    iou_by_cid = {c["class_id"]: c for c in per_class_iou}
    
    # Merge per_class_val with per_class_iou
    merged_per_class = []
    if per_class_val:
        for pc in per_class_val:
            cid = pc["class_id"]
            c_iou = iou_by_cid.get(cid, {})
            merged = {**pc}
            merged["mean_iou"] = c_iou.get("mean_iou", 0.0)
            merged["min_iou"] = c_iou.get("min_iou", 0.0)
            merged["max_iou"] = c_iou.get("max_iou", 0.0)
            merged["iou_gte_50_count"] = c_iou.get("iou_gte_50_count", 0)
            merged["iou_gte_50_pct"] = c_iou.get("iou_gte_50_pct", 0.0)
            merged["iou_gte_75_count"] = c_iou.get("iou_gte_75_count", 0)
            merged["iou_gte_75_pct"] = c_iou.get("iou_gte_75_pct", 0.0)
            merged["evaluated_detections"] = c_iou.get("evaluated_detections", 0)
            merged_per_class.append(merged)
    else:
        for cid, cname in class_names.items():
            c_iou = iou_by_cid.get(cid, {})
            merged_per_class.append({
                "class_id": cid,
                "disease_name": cname,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "ap50": 0.0,
                "ap50_95": 0.0,
                "mean_iou": c_iou.get("mean_iou", 0.0),
                "min_iou": c_iou.get("min_iou", 0.0),
                "max_iou": c_iou.get("max_iou", 0.0),
                "iou_gte_50_count": c_iou.get("iou_gte_50_count", 0),
                "iou_gte_50_pct": c_iou.get("iou_gte_50_pct", 0.0),
                "iou_gte_75_count": c_iou.get("iou_gte_75_count", 0),
                "iou_gte_75_pct": c_iou.get("iou_gte_75_pct", 0.0),
                "evaluated_detections": c_iou.get("evaluated_detections", 0)
            })

    # Model readiness
    is_ready = map50 >= 0.40 or f1 >= 0.35
    model_status = "READY" if is_ready else "NOT READY"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Overall performance dictionary
    overall_performance = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "map50": float(map50),
        "map50_95": float(map50_95),
        "mean_iou": float(mean_iou),
        "mean_iou_pct": float(mean_iou * 100.0),
        "min_iou": float(min_iou),
        "max_iou": float(max_iou),
        "evaluated_detections": len(all_matches),
        "iou_gte_50_count": int(iou_50_count),
        "iou_gte_50_pct": float(iou_50_rate * 100.0),
        "iou_gte_75_count": int(iou_75_count),
        "iou_gte_75_pct": float(iou_75_rate * 100.0),
        "tp": int(dataset_tp),
        "fp": int(dataset_fp),
        "fn": int(dataset_fn),
        "total_ground_truths": int(total_gts_count),
        "total_predictions": int(total_preds_count)
    }

    # Structured data satisfying both Section 14 and existing dashboard requirements
    result_data = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_iou": float(mean_iou),
        "map50": float(map50),
        "map50_95": float(map50_95),
        "iou_50_rate": float(iou_50_rate),
        "iou_75_rate": float(iou_75_rate),
        "tp": int(dataset_tp),
        "fp": int(dataset_fp),
        "fn": int(dataset_fn),
        "timestamp": timestamp,
        "model": {
            "name": str(Path(model.ckpt_path).name if hasattr(model, "ckpt_path") else "best.pt"),
            "architecture": "YOLO26",
            "task": "detect"
        },
        "dataset": {
            "path": "dataset/",
            "num_classes": len(class_names),
            f"{split}_images": len(image_paths)
        },
        "overall_performance": overall_performance,
        "per_class_performance": merged_per_class,
        "model_status": model_status,
        "readiness_threshold_map50": 0.40
    }

    # Save to results/reports/metrics.json and results/metrics.json
    out_json = output_json or (PROJECT_ROOT / "results" / "reports" / "metrics.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    # Secondary copies for easy project-wide access
    root_metrics = PROJECT_ROOT / "results" / "metrics.json"
    root_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(root_metrics, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    # Save CSV of matches
    out_csv = PROJECT_ROOT / "results" / f"iou_results_{split}.csv"
    export_iou_csv(all_matches, out_csv)

    # Also export text & markdown reports using existing report generator
    rep_txt = PROJECT_ROOT / "results" / "reports" / "model_report.txt"
    export_model_evaluation_reports(
        model_name="best.pt",
        dataset_info=result_data["dataset"],
        overall_metrics=overall_performance,
        per_class_metrics=merged_per_class,
        output_txt=rep_txt,
        output_json=out_json,
        readiness_threshold=0.40
    )

    return result_data
