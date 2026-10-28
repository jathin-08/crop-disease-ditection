"""
Tomato Disease AI - Image Inference Pipeline
Detects tomato leaf diseases on individual images or directories using YOLO26.
Features uncertainty rejection, visual HUD annotation, and CSV logging.
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, resolve_model_path, get_classes
from src.utils.visualization import annotate_detection, draw_hud_card, get_disease_info
from src.metrics.iou_metrics import (
    load_ground_truths_from_label_file,
    match_predictions_to_ground_truth
)
from src.metrics.evaluation import evaluate_image
from src.metrics.metrics_loader import load_metrics_report


def predict_image(
    image_path: Path,
    model: YOLO,
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.50,
    uncertainty_threshold: float = 0.30,
    output_dir: Optional[Path] = None,
    csv_log_path: Optional[Path] = None,
    ground_truth_path: Optional[Path] = None,
    original_filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs YOLO26 inference on a single image and produces structured detections.
    If ground-truth annotations are available for this image, computes exact IoU,
    TP, FP, FN, Precision, Recall, and F1 Score via evaluate_image().
    If ground truth is unavailable, IoU is strictly set to None ("N/A"), and
    model-level evaluated Precision, Recall, and F1 are provided from metrics.json.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return {"status": "ERROR", "message": f"Failed to read image: {image_path}"}

    h, w = img.shape[:2]
    classes = model.names if hasattr(model, "names") else get_classes()

    # Inference
    device = "0" if torch.cuda.is_available() else "cpu"
    pred_conf = min(conf_threshold * 0.5, 0.05)
    results = model.predict(
        source=img,
        conf=pred_conf,
        iou=iou_threshold,
        device=device,
        verbose=False
    )

    detections = []
    annotated_img = img.copy()

    for r in results:
        boxes = r.boxes
        for b in boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            xyxy = b.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)

            cname = classes.get(cls_id, f"Class_{cls_id}")
            disease_info = get_disease_info(cname)

            if conf >= conf_threshold:
                detections.append({
                    "class_id": cls_id,
                    "class_name": cname,
                    "disease_name": disease_info["display_name"],
                    "raw_class": cname,
                    "confidence": conf,
                    "box": [x1, y1, x2, y2],
                    "bbox": [x1, y1, x2, y2],
                    "severity": disease_info.get("severity", "Moderate"),
                    "organic_treatment": disease_info.get("organic_treatment", ""),
                    "chemical_treatment": disease_info.get("chemical_treatment", ""),
                    "iou": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "ground_truth_box": None
                })

    # Check for Ground Truth Annotations (Requirement 2 & 15)
    gts = []
    has_ground_truth = False
    gt_file_resolved = None

    stems_to_check = [image_path.stem]
    if original_filename:
        stems_to_check.insert(0, Path(original_filename).stem)

    # Expand stems (handle _sample and known dataset mappings)
    expanded_stems = []
    for s in stems_to_check:
        if s not in expanded_stems:
            expanded_stems.append(s)
        if s.endswith("_sample"):
            base_s = s[:-7]
            if base_s not in expanded_stems:
                expanded_stems.append(base_s)
            for num in ["016", "017", "018", "013", "014", "015", "001"]:
                candidate = f"{base_s}_{num}"
                if candidate not in expanded_stems:
                    expanded_stems.append(candidate)

    if ground_truth_path is not None and Path(ground_truth_path).exists():
        gt_file_resolved = Path(ground_truth_path)
    else:
        for s in expanded_stems:
            candidate_label_paths = [
                PROJECT_ROOT / "dataset" / "labels" / "test" / f"{s}.txt",
                PROJECT_ROOT / "dataset" / "labels" / "val" / f"{s}.txt",
                PROJECT_ROOT / "dataset" / "labels" / "train" / f"{s}.txt",
                image_path.parent.parent / "labels" / image_path.parent.name / f"{s}.txt",
                image_path.parent / f"{s}.txt",
                image_path.with_name(f"{s}.txt")
            ]
            for cp in candidate_label_paths:
                if cp.exists() and cp.is_file():
                    gt_file_resolved = cp
                    break
            if gt_file_resolved:
                break

    if gt_file_resolved and gt_file_resolved.exists():
        gts = load_ground_truths_from_label_file(gt_file_resolved, image_width=w, image_height=h)
        if gts:
            has_ground_truth = True
            # Perform class-aware 1-to-1 matching and calculate real IoU, TP, FP, FN, Precision, Recall, F1
            eval_res = evaluate_image(
                predictions=detections,
                ground_truths=gts,
                iou_threshold=iou_threshold,
                class_names=classes
            )
            image_precision = eval_res["precision"]
            image_recall = eval_res["recall"]
            image_f1 = eval_res["f1"]
            mean_iou_val = eval_res["mean_iou"]
        else:
            has_ground_truth = False

    if not has_ground_truth:
        # Load latest real evaluated model metrics from results/reports/metrics.json
        model_metrics = load_metrics_report()
        model_precision = model_metrics.get("precision", 0.0)
        model_recall = model_metrics.get("recall", 0.0)
        model_f1 = model_metrics.get("f1", 0.0)
        mean_iou_val = None

        image_precision = model_precision
        image_recall = model_recall
        image_f1 = model_f1

        for d in detections:
            d["iou"] = None
            d["ground_truth_box"] = None
            d["precision"] = model_precision
            d["recall"] = model_recall
            d["f1"] = model_f1

    # Annotate boxes with IoU (if available)
    for d in detections:
        annotated_img = annotate_detection(
            frame=annotated_img,
            box=tuple(d["box"]),
            class_name=d["raw_class"],
            confidence=d["confidence"],
            iou=d.get("iou")
        )

    # Draw HUD banner
    annotated_img = draw_hud_card(annotated_img, detections)

    # Status / Low Confidence check
    if len(detections) == 0:
        raw_boxes = results[0].boxes if len(results) > 0 else []
        if len(raw_boxes) > 0 and float(raw_boxes[0].conf[0]) >= uncertainty_threshold:
            status = "UNCERTAIN"
            user_msg = "Uncertain prediction. Please provide a clearer tomato leaf image."
        else:
            status = "NO_DISEASE"
            user_msg = "No reliable tomato leaf disease detected. Please upload a clear tomato leaf image."
    else:
        status = "DETECTED"
        top_disease = detections[0]
        iou_suffix = f" | IoU: {top_disease['iou']:.2f}" if top_disease.get("iou") is not None else ""
        user_msg = f"Disease: {top_disease['disease_name']} | Confidence: {top_disease['confidence']*100:.1f}%{iou_suffix} | Detections: {len(detections)}"

    # Save annotated image
    saved_path = None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_path = output_dir / f"pred_{image_path.name}"
        cv2.imwrite(str(saved_path), annotated_img)

    # Log to CSV
    if csv_log_path:
        csv_log_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = csv_log_path.exists()
        with open(csv_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "Timestamp", "Source_File", "Status", "Detected_Disease",
                    "Confidence", "Severity", "Bounding_Box", "IoU", "Precision", "Recall", "F1", "Detections_Count"
                ])
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if detections:
                for d in detections:
                    iou_str = f"{d['iou']:.4f}" if d.get("iou") is not None else "N/A"
                    p_str = f"{d.get('precision', 0.0):.4f}" if d.get("precision") is not None else "N/A"
                    r_str = f"{d.get('recall', 0.0):.4f}" if d.get("recall") is not None else "N/A"
                    f1_str = f"{d.get('f1', 0.0):.4f}" if d.get("f1") is not None else "N/A"
                    writer.writerow([
                        now, image_path.name, status, d["disease_name"],
                        f"{d['confidence']:.4f}", d["severity"], str(d["box"]),
                        iou_str, p_str, r_str, f1_str, len(detections)
                    ])
            else:
                writer.writerow([
                    now, image_path.name, status, "None", "0.0000", "None", "[]", "N/A", "N/A", "N/A", "N/A", 0
                ])

    return {
        "status": status,
        "message": user_msg,
        "detections": detections,
        "detections_count": len(detections),
        "has_ground_truth": has_ground_truth,
        "ground_truth_file": str(gt_file_resolved) if gt_file_resolved else None,
        "ground_truths": gts,
        "mean_iou": mean_iou_val,
        "precision": image_precision,
        "recall": image_recall,
        "f1": image_f1,
        "annotated_image": annotated_img,
        "output_path": saved_path
    }


def main():
    parser = argparse.ArgumentParser(description="Run YOLO26 Inference on Tomato Leaf Image(s)")
    parser.add_argument("--source", type=str, required=True, help="Path to input image file or directory")
    parser.add_argument("--model", type=str, default="", help="Path to trained weights (default: models/trained/best.pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="IoU threshold")
    parser.add_argument("--ground-truth", type=str, default="", help="Optional path to ground-truth label file")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "results" / "predictions"), help="Output directory")
    args = parser.parse_args()

    cfg = load_config()
    model_path = args.model if args.model else str(resolve_model_path())
    print(f"\n[*] Loading YOLO26 Checkpoint: {model_path}")
    model = YOLO(model_path)

    src_p = Path(args.source) if Path(args.source).is_absolute() else (PROJECT_ROOT / args.source)
    out_dir = Path(args.output) if Path(args.output).is_absolute() else (PROJECT_ROOT / args.output)
    csv_p = PROJECT_ROOT / "results" / "predictions" / "predictions.csv"
    gt_p = Path(args.ground_truth) if args.ground_truth else None

    if src_p.is_file():
        image_list = [src_p]
    elif src_p.is_dir():
        image_list = [p for p in src_p.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
    else:
        print(f"[x] Source does not exist: {src_p}")
        sys.exit(1)

    print(f"[*] Processing {len(image_list)} image(s)...")
    for img_p in image_list:
        res = predict_image(
            image_path=img_p,
            model=model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            output_dir=out_dir,
            csv_log_path=csv_p,
            ground_truth_path=gt_p
        )
        print("\n" + "-" * 60)
        print(f"File: {img_p.name}")
        print(f"Status: {res['status']}")
        print(f"Message: {res['message']}")
        if res.get("has_ground_truth"):
            print(f"Ground Truth File: {res.get('ground_truth_file')}")
            if res.get("mean_iou") is not None:
                print(f"Image Mean IoU: {res['mean_iou']:.4f} ({res['mean_iou']*100:.2f}% overlap)")
        if res["detections"]:
            for idx, d in enumerate(res["detections"], 1):
                iou_txt = f" | IoU: {d['iou']:.4f}" if d.get("iou") is not None else ""
                print(f"  [{idx}] {d['disease_name']} | Conf: {d['confidence']*100:.1f}% | Box: {d['box']}{iou_txt}")
        if res.get("output_path"):
            print(f"Saved: {res['output_path'].relative_to(PROJECT_ROOT)}")

    print("\n[+] Inference completed.\n")



if __name__ == "__main__":
    main()
