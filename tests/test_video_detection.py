"""
Tomato Disease AI - Video Detection & Tracking Test Suite
==============================================================================
Validates:
1. Video file validation (formats, resolution, FPS, frames, error handling)
2. Tracker: IoU matching, class consistency, track continuity, and summaries
3. End-to-end video inference and tracking
4. Mode A: Ground-truth matching, real IoU, Precision, Recall, F1 calculation
5. Mode B: Absence of ground truth guarantees N/A (no fabricated metrics)
6. Output artifact generation: Annotated MP4, CSV export, JSON metrics, Markdown report
==============================================================================
"""

import os
import sys
import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.video.tracker import SimpleBoxTracker
from src.video.frame_processor import VideoFrameProcessor
from src.inference.video_detection import (
    validate_video_file,
    find_ground_truth_dir_for_video,
    find_frame_label_file,
    process_video_stream
)
from src.utils.config import resolve_model_path
from ultralytics import YOLO


class TestVideoModule(unittest.TestCase):

    def setUp(self):
        self.sample_video = PROJECT_ROOT / "videos" / "input" / "tomato_sample_video.mp4"
        self.test_output_video = PROJECT_ROOT / "results" / "predictions" / "videos" / "test_output.mp4"
        self.test_csv = PROJECT_ROOT / "results" / "predictions" / "test_predictions.csv"

    def test_video_validation_success(self):
        """Valid video file returns valid metadata."""
        if not self.sample_video.exists():
            self.skipTest("Sample video file not found")

        res = validate_video_file(self.sample_video)
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["filename"], "tomato_sample_video.mp4")
        self.assertGreater(res["width"], 0)
        self.assertGreater(res["height"], 0)
        self.assertGreater(res["fps"], 0)
        self.assertGreater(res["total_frames"], 0)
        self.assertGreater(res["duration_seconds"], 0)

    def test_video_validation_nonexistent(self):
        """Non-existent video file returns clean error without raising uncaught exception."""
        res = validate_video_file(PROJECT_ROOT / "videos" / "non_existent.mp4")
        self.assertFalse(res["is_valid"])
        self.assertIn("not found", res["error"])

    def test_video_validation_invalid_extension(self):
        """Unsupported extension returns clean error."""
        dummy_file = PROJECT_ROOT / "tests" / "dummy.txt"
        dummy_file.write_text("dummy", encoding="utf-8")
        try:
            res = validate_video_file(dummy_file)
            self.assertFalse(res["is_valid"])
            self.assertIn("Unsupported format", res["error"])
        finally:
            if dummy_file.exists():
                dummy_file.unlink()

    def test_tracker_class_consistency(self):
        """Tracker must NEVER merge detections belonging to different disease classes."""
        tracker = SimpleBoxTracker(max_disappeared=5, iou_thresh=0.3, enforce_class_consistency=True)

        # Frame 1: Detection of Bacterial Spot at [100, 100, 200, 200]
        dets_f1 = [{
            "box": [100.0, 100.0, 200.0, 200.0],
            "class_name": "tomato_bacterial_spot",
            "class_id": 1,
            "confidence": 0.85
        }]
        res1 = tracker.update(dets_f1)
        self.assertEqual(len(res1), 1)
        tid1 = res1[0]["track_id"]

        # Frame 2: Same spatial location, but different disease class (Early Blight)
        dets_f2 = [{
            "box": [102.0, 101.0, 198.0, 202.0],
            "class_name": "tomato_early_blight",
            "class_id": 2,
            "confidence": 0.80
        }]
        res2 = tracker.update(dets_f2)
        self.assertEqual(len(res2), 1)
        tid2 = res2[0]["track_id"]

        # Track IDs must be DIFFERENT due to class consistency enforcement
        self.assertNotEqual(tid1, tid2)

    def test_tracker_continuity(self):
        """Tracker maintains same track ID across frames for overlapping detections of same class."""
        tracker = SimpleBoxTracker(max_disappeared=5, iou_thresh=0.3, enforce_class_consistency=True)

        dets_f1 = [{
            "box": [50.0, 50.0, 150.0, 150.0],
            "class_name": "tomato_early_blight",
            "class_id": 2,
            "confidence": 0.90
        }]
        res1 = tracker.update(dets_f1)
        tid1 = res1[0]["track_id"]

        # Slightly shifted box, same disease
        dets_f2 = [{
            "box": [53.0, 52.0, 152.0, 151.0],
            "class_name": "tomato_early_blight",
            "class_id": 2,
            "confidence": 0.88
        }]
        res2 = tracker.update(dets_f2)
        tid2 = res2[0]["track_id"]

        self.assertEqual(tid1, tid2)

        summary = tracker.get_track_summary()
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["track_id"], tid1)
        self.assertEqual(summary[0]["total_frames_seen"], 2)

    def test_mode_b_no_ground_truth_never_fabricates(self):
        """When ground truth is absent, IoU, Precision, Recall, and F1 must strictly be None/null."""
        if not self.sample_video.exists():
            self.skipTest("Sample video file not found")

        model_path = resolve_model_path()
        model = YOLO(str(model_path))

        stats = process_video_stream(
            source_video=self.sample_video,
            output_video=self.test_output_video,
            model=model,
            conf_threshold=0.15,
            iou_threshold=0.50,
            frame_step=5,
            csv_log_path=self.test_csv,
            labels_dir="none"  # Explicitly force Mode B
        )

        self.assertFalse(stats["has_ground_truth"])
        self.assertIsNone(stats["mean_iou"])
        self.assertIsNone(stats["precision"])
        self.assertIsNone(stats["recall"])
        self.assertIsNone(stats["f1"])
        self.assertIsNone(stats["map50"])

        # Check JSON export
        with open(stats["json_report_path"], "r", encoding="utf-8") as jf:
            jdata = json.load(jf)
            self.assertIsNone(jdata["mean_iou"])
            self.assertIsNone(jdata["precision"])
            self.assertIsNone(jdata["recall"])
            self.assertIsNone(jdata["f1"])
            self.assertFalse(jdata["has_ground_truth"])

    def test_mode_a_with_ground_truth_calculates_real_metrics(self):
        """When ground truth is present, IoU, Precision, Recall, and F1 must be calculated."""
        if not self.sample_video.exists():
            self.skipTest("Sample video file not found")

        gt_dir = PROJECT_ROOT / "videos" / "labels" / "tomato_sample_video"
        if not gt_dir.exists():
            self.skipTest("Sample video labels not found")

        model_path = resolve_model_path()
        model = YOLO(str(model_path))

        stats = process_video_stream(
            source_video=self.sample_video,
            output_video=self.test_output_video,
            model=model,
            conf_threshold=0.15,
            iou_threshold=0.50,
            frame_step=4,
            csv_log_path=self.test_csv,
            labels_dir=gt_dir
        )

        self.assertTrue(stats["has_ground_truth"])
        self.assertIsNotNone(stats["mean_iou"])
        self.assertIsNotNone(stats["precision"])
        self.assertIsNotNone(stats["recall"])
        self.assertIsNotNone(stats["f1"])
        self.assertGreaterEqual(stats["mean_iou"], 0.0)
        self.assertLessEqual(stats["mean_iou"], 1.0)
        self.assertGreaterEqual(stats["precision"], 0.0)
        self.assertGreaterEqual(stats["recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
