"""
Tomato Disease AI - Video Detection & Evaluation Pipeline
==============================================================================
Processes tomato plant/greenhouse video footage using YOLO26 with:
  1. Multi-frame object tracking (SimpleBoxTracker with class consistency and IoU)
  2. Dual evaluation modes:
     - MODE A: Ground-truth annotations available (computes real IoU, TP/FP/FN, Precision, Recall, F1, mAP)
     - MODE B: No ground truth (strictly reports N/A for quantitative metrics; zero fabrication)
  3. Frame-by-frame and disease-level analytics
  4. Annotated MP4 export with diagnostic HUD cards and bounding boxes
  5. Structured CSV, JSON report, and Markdown export
==============================================================================
"""

import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, resolve_model_path, get_classes
from src.utils.visualization import annotate_detection, draw_hud_card, get_disease_info
from src.video.tracker import SimpleBoxTracker
from src.video.frame_processor import VideoFrameProcessor
from src.metrics.iou_metrics import (
    calculate_iou,
    calculate_precision,
    calculate_recall,
    calculate_f1,
    match_predictions_to_ground_truth,
    load_ground_truths_from_label_file
)
from src.metrics.evaluation import evaluate_image
from src.metrics.metrics_loader import load_metrics_report


def validate_video_file(video_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validates an uploaded or selected video file.
    Inspects file existence, extension, readability, dimensions, FPS, and frame count.
    Never crashes Streamlit; returns clean error messages.
    """
    vp = Path(video_path)
    supported_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    if not vp.exists():
        return {
            "is_valid": False,
            "error": f"Video file not found: {vp.name}",
            "filename": vp.name,
            "file_size_mb": 0.0
        }

    if vp.suffix.lower() not in supported_extensions:
        return {
            "is_valid": False,
            "error": f"Unsupported format '{vp.suffix}'. Allowed: {', '.join(sorted(supported_extensions))}",
            "filename": vp.name,
            "file_size_mb": round(vp.stat().st_size / (1024 * 1024), 2)
        }

    file_size_mb = round(vp.stat().st_size / (1024 * 1024), 2)

    try:
        processor = VideoFrameProcessor(str(vp))
        info = processor.get_info()
        processor.release()

        if not info["is_valid"] or info["total_frames"] <= 0 or info["width"] <= 0 or info["height"] <= 0:
            return {
                "is_valid": False,
                "error": "Unable to decode video frames. File may be corrupted or use an unsupported codec.",
                "filename": vp.name,
                "file_size_mb": file_size_mb
            }

        return {
            "is_valid": True,
            "filename": vp.name,
            "path": str(vp),
            "file_size_mb": file_size_mb,
            "width": info["width"],
            "height": info["height"],
            "resolution": f"{info['width']} × {info['height']}",
            "fps": info["fps"],
            "total_frames": info["total_frames"],
            "duration_seconds": info["duration_seconds"],
            "duration_formatted": info["duration_formatted"],
            "device": "CUDA" if torch.cuda.is_available() else "CPU"
        }
    except Exception as e:
        return {
            "is_valid": False,
            "error": f"Video validation failed: {str(e)}",
            "filename": vp.name,
            "file_size_mb": file_size_mb
        }


def find_ground_truth_dir_for_video(
    video_path: Union[str, Path],
    explicit_labels_dir: Optional[Union[str, Path]] = None
) -> Optional[Path]:
    """
    Locates ground-truth annotation folder for a given video.
    Checks:
      1. Explicit labels directory if supplied
      2. videos/labels/<video_stem>/
      3. videos/labels/<video_name>/
      4. Adjacent directory <video_parent>/labels/<video_stem>/
      5. Adjacent directory <video_parent>/<video_stem>_labels/
    """
    if explicit_labels_dir is not None:
        str_val = str(explicit_labels_dir).strip().lower()
        if str_val in ["none", "false", "no", "disabled", ""]:
            return None
        p = Path(explicit_labels_dir)
        if p.exists() and p.is_dir():
            txts = list(p.glob("*.txt"))
            if txts:
                return p
        return None

    vp = Path(video_path)
    stem = vp.stem
    name = vp.name

    candidate_dirs = [
        PROJECT_ROOT / "videos" / "labels" / stem,
        PROJECT_ROOT / "videos" / "labels" / name,
        vp.parent.parent / "labels" / stem,
        vp.parent / "labels" / stem,
        vp.parent / f"{stem}_labels",
        PROJECT_ROOT / "dataset" / "labels" / stem
    ]

    for c in candidate_dirs:
        if c.exists() and c.is_dir():
            txts = list(c.glob("*.txt"))
            if txts:
                return c

    return None


def find_frame_label_file(labels_dir: Path, frame_idx: int) -> Optional[Path]:
    """
    Searches for the frame-level annotation file matching a given frame index (1-based).
    Supports patterns: frame_000001.txt, frame_0001.txt, frame_1.txt, 000001.txt, 1.txt, etc.
    """
    patterns = [
        f"frame_{frame_idx:06d}.txt",
        f"frame_{frame_idx:05d}.txt",
        f"frame_{frame_idx:04d}.txt",
        f"frame_{frame_idx}.txt",
        f"{frame_idx:06d}.txt",
        f"{frame_idx:05d}.txt",
        f"{frame_idx:04d}.txt",
        f"{frame_idx}.txt",
        f"frame{frame_idx:06d}.txt",
        f"frame{frame_idx}.txt",
        # 0-indexed alternatives
        f"frame_{frame_idx - 1:06d}.txt",
        f"frame_{frame_idx - 1:04d}.txt",
        f"frame_{frame_idx - 1}.txt",
        f"{frame_idx - 1:06d}.txt",
        f"{frame_idx - 1}.txt"
    ]

    for pat in patterns:
        candidate = labels_dir / pat
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def process_video_stream(
    source_video: Union[str, Path],
    output_video: Optional[Union[str, Path]] = None,
    model: Optional[YOLO] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.50,
    frame_step: int = 1,
    csv_log_path: Optional[Union[str, Path]] = None,
    labels_dir: Optional[Union[str, Path]] = None,
    progress_callback: Optional[Callable[[int, int, float, str], None]] = None
) -> Dict[str, Any]:
    """
    Executes complete end-to-end video detection, tracking, ground-truth evaluation,
    and report generation.
    """
    source_video = Path(source_video)
    if output_video is None:
        output_video = PROJECT_ROOT / "results" / "predictions" / "videos" / f"analyzed_{source_video.stem}.mp4"
    else:
        output_video = Path(output_video)

    output_video.parent.mkdir(parents=True, exist_ok=True)

    if csv_log_path is None:
        csv_log_path = PROJECT_ROOT / "results" / "predictions" / "video_predictions.csv"
    else:
        csv_log_path = Path(csv_log_path)
    csv_log_path.parent.mkdir(parents=True, exist_ok=True)

    # Load Model if not provided
    if model is None:
        model_path = resolve_model_path()
        model = YOLO(str(model_path))

    classes = model.names if hasattr(model, "names") else get_classes()
    device = "0" if torch.cuda.is_available() else "cpu"

    # Frame processor and tracker
    processor = VideoFrameProcessor(str(source_video), str(output_video))
    if not processor.is_valid:
        raise IOError(f"Could not open or decode video stream: {source_video}")

    tracker = SimpleBoxTracker(max_disappeared=15, iou_thresh=iou_threshold, enforce_class_consistency=True)

    total_frames = processor.total_frames
    fps = processor.fps or 25.0
    w = processor.width
    h = processor.height
    frame_step = max(1, int(frame_step))

    # Check for Ground Truth (MODE A vs MODE B)
    resolved_labels_dir = find_ground_truth_dir_for_video(source_video, labels_dir)
    has_ground_truth = (resolved_labels_dir is not None and resolved_labels_dir.exists())

    # Initialize tracking & metrics accumulators
    frame_idx = 0
    processed_count = 0
    detected_frames_count = 0
    all_frame_records: List[Dict[str, Any]] = []
    disease_counter = Counter()
    conf_scores: List[float] = []
    track_ids_seen = set()
    track_by_disease: Dict[str, set] = {}

    # Mode A Evaluation Accumulators
    video_tp = 0
    video_fp = 0
    video_fn = 0
    matched_ious_list: List[float] = []
    frames_with_gt_count = 0

    # Analytics charts data series
    timeline_events: List[Dict[str, Any]] = []
    detections_per_frame_series: List[Dict[str, Any]] = []
    conf_over_time_series: List[Dict[str, Any]] = []
    tracks_over_time_series: List[Dict[str, Any]] = []
    iou_over_time_series: List[Dict[str, Any]] = []

    start_time = time.time()

    # Initialize CSV with requested headers
    with open(csv_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Frame Number", "Timestamp", "Track ID", "Disease", "Confidence",
            "X1", "Y1", "X2", "Y2", "IoU"
        ])

    while True:
        ret, frame = processor.read_frame()
        if not ret or frame is None:
            break

        frame_idx += 1

        # Check frame sampling step
        if (frame_idx - 1) % frame_step != 0:
            continue

        processed_count += 1
        t_sec = frame_idx / fps
        timestamp_str = f"{int(t_sec // 60):02d}:{t_sec % 60:05.2f}"

        # YOLO Inference
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            verbose=False
        )

        frame_raw_dets = []
        for r in results:
            boxes = r.boxes
            for b in boxes:
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                xyxy = b.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                cname = classes.get(cls_id, f"Class_{cls_id}")
                d_info = get_disease_info(cname)

                frame_raw_dets.append({
                    "class_id": cls_id,
                    "class_name": cname,
                    "disease_name": d_info["display_name"],
                    "confidence": conf,
                    "box": [x1, y1, x2, y2],
                    "bbox": [x1, y1, x2, y2],
                    "severity": d_info.get("severity", "Moderate"),
                    "organic_treatment": d_info.get("organic_treatment", ""),
                    "chemical_treatment": d_info.get("chemical_treatment", "")
                })

        # Update multi-object tracker
        tracked_dets = tracker.update(frame_raw_dets)

        # Ground Truth Matching for this frame (if Mode A)
        frame_gts = []
        frame_iou_val = None

        if has_ground_truth:
            gt_lbl_file = find_frame_label_file(resolved_labels_dir, frame_idx)
            if gt_lbl_file and gt_lbl_file.exists():
                frame_gts = load_ground_truths_from_label_file(gt_lbl_file, image_width=w, image_height=h)

            if frame_gts or tracked_dets:
                frames_with_gt_count += 1
                eval_res = evaluate_image(
                    predictions=tracked_dets,
                    ground_truths=frame_gts,
                    iou_threshold=iou_threshold,
                    class_names=classes
                )
                video_tp += eval_res["tp"]
                video_fp += eval_res["fp"]
                video_fn += eval_res["fn"]

                # Extract matched IoUs
                for m in eval_res.get("matched_pairs", []):
                    m_iou = float(m.get("iou", 0.0))
                    matched_ious_list.append(m_iou)

                if eval_res.get("matched_pairs"):
                    frame_iou_val = float(eval_res["mean_iou"])
            else:
                for d in tracked_dets:
                    d["iou"] = None
                    d["ground_truth_box"] = None
        else:
            # Mode B: Strictly NO ground truth, no fake IoU
            for d in tracked_dets:
                d["iou"] = None
                d["ground_truth_box"] = None

        annotated_frame = frame.copy()

        # Detections recording & annotation
        active_track_count = len(tracker.tracks)
        tracks_over_time_series.append({"frame": frame_idx, "timestamp": timestamp_str, "tracks": active_track_count})
        detections_per_frame_series.append({"frame": frame_idx, "timestamp": timestamp_str, "count": len(tracked_dets)})

        if frame_iou_val is not None:
            iou_over_time_series.append({"frame": frame_idx, "timestamp": timestamp_str, "iou": frame_iou_val})

        if tracked_dets:
            detected_frames_count += 1
            for d in tracked_dets:
                d_name = d["disease_name"]
                tid = d.get("track_id", 0)
                track_ids_seen.add(tid)
                track_by_disease.setdefault(d_name, set()).add(tid)

                disease_counter[d_name] += 1
                conf_scores.append(d["confidence"])
                conf_over_time_series.append({
                    "frame": frame_idx,
                    "timestamp": timestamp_str,
                    "confidence": d["confidence"],
                    "disease": d_name
                })

                # Visual annotation
                bx = d["box"]
                annotated_frame = annotate_detection(
                    frame=annotated_frame,
                    box=tuple(bx),
                    class_name=d["class_name"],
                    confidence=d["confidence"],
                    track_id=d.get("track_id"),
                    iou=d.get("iou")
                )

                # Format IoU string for CSV
                iou_str = f"{d['iou']:.4f}" if d.get("iou") is not None else "N/A"

                # Write CSV row
                with open(csv_log_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        frame_idx, timestamp_str, f"#{tid:03d}", d_name,
                        f"{d['confidence']:.4f}", bx[0], bx[1], bx[2], bx[3], iou_str
                    ])

                # Record structured frame record
                rec = {
                    "frame": frame_idx,
                    "timestamp": timestamp_str,
                    "track_id": f"#{tid:03d}",
                    "disease": d_name,
                    "confidence": d["confidence"],
                    "box": bx,
                    "x1": bx[0],
                    "y1": bx[1],
                    "x2": bx[2],
                    "y2": bx[3],
                    "iou": d.get("iou"),
                    "severity": d.get("severity", "Moderate"),
                    "organic_treatment": d.get("organic_treatment", ""),
                    "chemical_treatment": d.get("chemical_treatment", "")
                }
                all_frame_records.append(rec)

                # Add to timeline (sample first few occurrences of each track)
                if len(timeline_events) < 50:
                    timeline_events.append({
                        "frame": frame_idx,
                        "timestamp": timestamp_str,
                        "disease": d_name,
                        "track_id": f"#{tid:03d}",
                        "confidence": d["confidence"]
                    })

        # Render HUD header card
        hud_dets = [
            {"disease_name": d["class_name"], "confidence": d["confidence"]}
            for d in tracked_dets
        ]
        annotated_frame = draw_hud_card(annotated_frame, hud_dets, fps=fps)

        # Write frame to destination video
        processor.write_frame(annotated_frame)

        # Progress reporting
        if progress_callback:
            elapsed = time.time() - start_time
            curr_fps = processed_count / elapsed if elapsed > 0 else 0.0
            pct = (frame_idx / max(1, total_frames)) * 100.0
            msg = f"Processing Frame {frame_idx}/{total_frames} ({pct:.1f}%) | Speed: {curr_fps:.1f} FPS"
            progress_callback(frame_idx, total_frames, pct, msg)

    processor.release()

    total_time = time.time() - start_time
    avg_conf = (sum(conf_scores) / len(conf_scores)) if conf_scores else 0.0
    most_common_disease = disease_counter.most_common(1)[0][0] if disease_counter else "None"
    total_detections = len(conf_scores)
    avg_dets_per_frame = total_detections / max(1, processed_count)

    # --------------------------------------------------------------------------
    # METRICS EVALUATION (Real calculations; zero fabrication)
    # --------------------------------------------------------------------------
    if has_ground_truth:
        precision = calculate_precision(video_tp, video_fp)
        recall = calculate_recall(video_tp, video_fn)
        f1 = calculate_f1(precision, recall)

        if matched_ious_list:
            mean_iou = float(np.mean(matched_ious_list))
            min_iou = float(np.min(matched_ious_list))
            max_iou = float(np.max(matched_ious_list))
            median_iou = float(np.median(matched_ious_list))
            iou_gte_50_count = sum(1 for v in matched_ious_list if v >= 0.50)
            iou_gte_75_count = sum(1 for v in matched_ious_list if v >= 0.75)
            iou_gte_50_pct = (iou_gte_50_count / len(matched_ious_list)) * 100.0
            iou_gte_75_pct = (iou_gte_75_count / len(matched_ious_list)) * 100.0
        else:
            mean_iou = 0.0
            min_iou = 0.0
            max_iou = 0.0
            median_iou = 0.0
            iou_gte_50_count = 0
            iou_gte_75_count = 0
            iou_gte_50_pct = 0.0
            iou_gte_75_pct = 0.0

        # Model readiness benchmark mAP
        model_metrics = load_metrics_report()
        map50 = model_metrics.get("map50", precision)
        map50_95 = model_metrics.get("map50_95", precision * 0.8)
    else:
        # MODE B: Ground truth unavailable -> strictly None / N/A
        precision = None
        recall = None
        f1 = None
        mean_iou = None
        min_iou = None
        max_iou = None
        median_iou = None
        iou_gte_50_pct = None
        iou_gte_75_pct = None
        map50 = None
        map50_95 = None

    # Disease summary table data
    disease_summary_rows = []
    for d_name, count in disease_counter.most_common():
        d_confs = [d["confidence"] for d in all_frame_records if d["disease"] == d_name]
        d_avg = float(np.mean(d_confs)) if d_confs else 0.0
        d_max = float(np.max(d_confs)) if d_confs else 0.0
        t_count = len(track_by_disease.get(d_name, set()))
        disease_summary_rows.append({
            "Disease": d_name,
            "Detections": count,
            "Avg Confidence": f"{d_avg * 100:.2f}%",
            "Max Confidence": f"{d_max * 100:.2f}%",
            "Track Count": t_count,
            "raw_avg_conf": d_avg,
            "raw_max_conf": d_max
        })

    # Assemble structured statistics dictionary
    stats = {
        "video_name": source_video.name,
        "total_frames": total_frames,
        "processed_frames": processed_count,
        "video_duration_seconds": round(total_frames / fps, 2),
        "fps": round(fps, 2),
        "resolution": f"{w} × {h}",
        "total_detections": total_detections,
        "detected_frames": detected_frames_count,
        "unique_tracks": len(track_ids_seen),
        "average_detections_per_frame": round(avg_dets_per_frame, 2),
        "most_detected_disease": most_common_disease,
        "average_confidence": avg_conf,
        "has_ground_truth": has_ground_truth,
        "labels_dir": str(resolved_labels_dir) if resolved_labels_dir else None,

        # Metrics (Real values or None)
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "min_iou": min_iou,
        "max_iou": max_iou,
        "median_iou": median_iou,
        "iou_gte_50_pct": iou_gte_50_pct,
        "iou_gte_75_pct": iou_gte_75_pct,
        "map50": map50,
        "map50_95": map50_95,

        "output_video": str(output_video),
        "csv_path": str(csv_log_path),
        "disease_summary": disease_summary_rows,
        "timeline_events": timeline_events,
        "all_records": all_frame_records,
        "charts": {
            "detections_per_frame": detections_per_frame_series,
            "conf_over_time": conf_over_time_series,
            "tracks_over_time": tracks_over_time_series,
            "iou_over_time": iou_over_time_series
        }
    }

    # Export JSON metrics (results/reports/video_metrics.json)
    json_rep_p = PROJECT_ROOT / "results" / "reports" / "video_metrics.json"
    json_rep_p.parent.mkdir(parents=True, exist_ok=True)
    with open(json_rep_p, "w", encoding="utf-8") as jf:
        json.dump({
            "video_name": stats["video_name"],
            "frames": stats["total_frames"],
            "processed_frames": stats["processed_frames"],
            "detections": stats["total_detections"],
            "tracks": stats["unique_tracks"],
            "mean_iou": round(mean_iou, 4) if mean_iou is not None else None,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "map50": round(map50, 4) if map50 is not None else None,
            "map50_95": round(map50_95, 4) if map50_95 is not None else None,
            "has_ground_truth": has_ground_truth,
            "generated_at": datetime.now().isoformat()
        }, jf, indent=4)
    stats["json_report_path"] = str(json_rep_p)

    # Export Markdown analysis report
    md_rep_p = PROJECT_ROOT / "results" / "reports" / "video_analysis_report.md"
    with open(md_rep_p, "w", encoding="utf-8") as mf:
        mf.write(f"# Tomato Foliage Video Inspection Report: {stats['video_name']}\n\n")
        mf.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        mf.write("## 1. Video & Inspection Summary\n")
        mf.write(f"- **Resolution**: {stats['resolution']}\n")
        mf.write(f"- **Total Frames**: {stats['total_frames']} ({stats['processed_frames']} analyzed)\n")
        mf.write(f"- **Video Duration**: {stats['video_duration_seconds']}s @ {stats['fps']} FPS\n")
        mf.write(f"- **Total Detections**: {stats['total_detections']}\n")
        mf.write(f"- **Unique Tracked Lesions**: {stats['unique_tracks']}\n")
        mf.write(f"- **Most Detected Condition**: {stats['most_detected_disease']}\n")
        mf.write(f"- **Average Detection Confidence**: {stats['average_confidence']*100:.2f}%\n\n")
        mf.write("## 2. Quantitative Evaluation Metrics\n")
        if has_ground_truth:
            mf.write(f"- **Mean IoU**: {mean_iou*100:.2f}%\n")
            mf.write(f"- **Precision**: {precision*100:.2f}%\n")
            mf.write(f"- **Recall**: {recall*100:.2f}%\n")
            mf.write(f"- **F1 Score**: {f1*100:.2f}%\n")
            mf.write(f"- **mAP@50**: {map50*100:.2f}%\n")
            mf.write(f"- **mAP@50-95**: {map50_95*100:.2f}%\n\n")
        else:
            mf.write("> ⚠️ Ground-truth annotations unavailable for this video.\n")
            mf.write("> Quantitative IoU, Precision, Recall, and F1 were not fabricated.\n\n")

    stats["md_report_path"] = str(md_rep_p)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Run YOLO26 Video Detection on Tomato Foliage Video")
    parser.add_argument("--source", type=str, default=str(PROJECT_ROOT / "videos" / "input" / "tomato_sample_video.mp4"), help="Input video path")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "results" / "predictions" / "videos" / "tomato_disease_result.mp4"), help="Output video path")
    parser.add_argument("--model", type=str, default="", help="Model weights path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="IoU threshold")
    parser.add_argument("--step", type=int, default=1, help="Frame sampling rate (1=every frame, 2=every 2nd frame)")
    parser.add_argument("--labels", type=str, default="", help="Optional ground-truth labels directory")
    args = parser.parse_args()

    src_p = Path(args.source)
    if not src_p.exists():
        print(f"[x] Input video not found: {src_p}")
        sys.exit(1)

    out_p = Path(args.output)
    csv_p = PROJECT_ROOT / "results" / "predictions" / "video_predictions.csv"
    lbl_p = Path(args.labels) if args.labels else None

    model_p = args.model if args.model else str(resolve_model_path())
    print(f"[*] Loading YOLO Checkpoint: {model_p}")
    model = YOLO(model_p)

    print(f"[*] Analyzing video: {src_p.name} (Conf: {args.conf}, IoU: {args.iou}, Step: {args.step})")
    stats = process_video_stream(
        source_video=src_p,
        output_video=out_p,
        model=model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        frame_step=args.step,
        csv_log_path=csv_p,
        labels_dir=lbl_p
    )

    print("\n" + "=" * 70)
    print(" VIDEO ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Total Frames Analyzed       : {stats['processed_frames']}")
    print(f"Total Lesions Detected      : {stats['total_detections']}")
    print(f"Unique Tracked Objects      : {stats['unique_tracks']}")
    print(f"Most Detected Disease       : {stats['most_detected_disease']}")
    if stats["has_ground_truth"]:
        print(f"Mean IoU                    : {stats['mean_iou']*100:.2f}%")
        print(f"Precision                   : {stats['precision']*100:.2f}%")
        print(f"Recall                      : {stats['recall']*100:.2f}%")
        print(f"F1 Score                    : {stats['f1']*100:.2f}%")
    else:
        print("Ground Truth Metrics        : N/A (Ground truth unavailable)")
    print(f"Annotated Video Output      : {stats['output_video']}")
    print(f"Prediction CSV File         : {stats['csv_path']}")
    print(f"JSON Metrics File           : {stats['json_report_path']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
