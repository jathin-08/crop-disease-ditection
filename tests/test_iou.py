"""
Tomato Disease AI - IoU Unit & Integration Test Suite
==============================================================================
Validates:
1. Identical boxes -> IoU = 1.0
2. Non-overlapping boxes -> IoU = 0.0
3. Partial overlap -> Exact mathematical IoU calculations
4. Touching / adjacent edges -> IoU = 0.0
5. Degenerate and zero-area boxes -> IoU = 0.0 without division by zero
6. Coordinate formats: xyxy and xywh (normalized and pixel)
7. Pairwise IoU matrix computation
8. Class-aware matching: Different disease classes are NEVER matched
9. Greedy one-to-one matching: Highest IoU pairing without duplicate assignment
10. Threshold evaluations (IoU >= 0.50, IoU >= 0.75)
11. Per-class IoU computation
12. Error handling: Empty lists, missing annotations, unpredicted images
==============================================================================
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.iou_metrics import (
    calculate_iou,
    calculate_iou_matrix,
    xywh_to_xyxy,
    xyxy_to_xywh,
    match_predictions_to_ground_truth,
    calculate_mean_iou,
    calculate_per_class_iou,
    calculate_iou_thresholds,
    calculate_iou_summary
)


class TestIoUCalculation(unittest.TestCase):

    def test_identical_boxes(self):
        """Identical boxes must return an IoU of exactly 1.0."""
        # Pixel coordinates
        b1 = [10.0, 20.0, 110.0, 120.0]
        b2 = [10.0, 20.0, 110.0, 120.0]
        self.assertAlmostEqual(calculate_iou(b1, b2), 1.0, places=6)

        # Normalized coordinates
        bn1 = [0.1, 0.2, 0.8, 0.9]
        bn2 = [0.1, 0.2, 0.8, 0.9]
        self.assertAlmostEqual(calculate_iou(bn1, bn2), 1.0, places=6)

    def test_non_overlapping_boxes(self):
        """Boxes with no spatial intersection must return IoU of 0.0."""
        b1 = [0.0, 0.0, 10.0, 10.0]
        b2 = [20.0, 20.0, 30.0, 30.0]
        self.assertEqual(calculate_iou(b1, b2), 0.0)

        # Disjoint horizontally
        b3 = [0.0, 0.0, 10.0, 10.0]
        b4 = [15.0, 0.0, 25.0, 10.0]
        self.assertEqual(calculate_iou(b3, b4), 0.0)

    def test_partially_overlapping_boxes(self):
        """Partially overlapping boxes must match precise hand-calculated values."""
        # Example 1: 50% horizontal shift
        # Box 1: [0, 0, 10, 10] -> Area = 100
        # Box 2: [5, 0, 15, 10] -> Area = 100
        # Intersection: [5, 0, 10, 10] -> Area = 5 * 10 = 50
        # Union: 100 + 100 - 50 = 150
        # Expected IoU: 50 / 150 = 1/3
        b1 = [0.0, 0.0, 10.0, 10.0]
        b2 = [5.0, 0.0, 15.0, 10.0]
        self.assertAlmostEqual(calculate_iou(b1, b2), 1.0 / 3.0, places=6)

        # Example 2: Box 2 inside Box 1
        # Box 1: [0, 0, 10, 10] -> Area = 100
        # Box 2: [2, 2, 8, 8] -> Area = 36
        # Intersection: [2, 2, 8, 8] -> Area = 36
        # Union: 100 + 36 - 36 = 100
        # Expected IoU: 36 / 100 = 0.36
        b3 = [0.0, 0.0, 10.0, 10.0]
        b4 = [2.0, 2.0, 8.0, 8.0]
        self.assertAlmostEqual(calculate_iou(b3, b4), 0.36, places=6)

        # Example 3: 2D shift
        # Box 1: [0, 0, 4, 4] -> Area = 16
        # Box 2: [2, 2, 6, 6] -> Area = 16
        # Intersection: [2, 2, 4, 4] -> Area = 4
        # Union: 16 + 16 - 4 = 28
        # Expected IoU: 4 / 28 = 1/7
        b5 = [0.0, 0.0, 4.0, 4.0]
        b6 = [2.0, 2.0, 6.0, 6.0]
        self.assertAlmostEqual(calculate_iou(b5, b6), 4.0 / 28.0, places=6)

    def test_touching_edges(self):
        """Boxes that share an edge or corner but have zero intersection area must return 0.0."""
        b1 = [0.0, 0.0, 10.0, 10.0]
        b2 = [10.0, 0.0, 20.0, 10.0]  # Touching vertical edge
        self.assertEqual(calculate_iou(b1, b2), 0.0)

        b3 = [0.0, 0.0, 10.0, 10.0]
        b4 = [10.0, 10.0, 20.0, 20.0]  # Touching corner
        self.assertEqual(calculate_iou(b3, b4), 0.0)

    def test_zero_area_and_degenerate_boxes(self):
        """Zero-area or degenerate boxes must be handled safely without division by zero."""
        b_empty = [0.0, 0.0, 0.0, 0.0]
        b_normal = [0.0, 0.0, 10.0, 10.0]
        self.assertEqual(calculate_iou(b_empty, b_normal), 0.0)
        self.assertEqual(calculate_iou(b_empty, b_empty), 0.0)
        self.assertEqual(calculate_iou([], b_normal), 0.0)
        self.assertEqual(calculate_iou(None, b_normal), 0.0)

    def test_format_conversion_xywh(self):
        """Tests that 'xywh' format yields identical IoU results."""
        # Box 1 in xywh: center (5, 5), width 10, height 10 -> [0, 0, 10, 10]
        # Box 2 in xywh: center (10, 5), width 10, height 10 -> [5, 0, 15, 10]
        b1_xywh = [5.0, 5.0, 10.0, 10.0]
        b2_xywh = [10.0, 5.0, 10.0, 10.0]
        iou_xywh = calculate_iou(b1_xywh, b2_xywh, box_format="xywh")
        self.assertAlmostEqual(iou_xywh, 1.0 / 3.0, places=6)

        # Coordinate round-trip conversion test
        converted_xyxy = xywh_to_xyxy(b1_xywh)
        self.assertEqual(converted_xyxy, [0.0, 0.0, 10.0, 10.0])
        converted_back = xyxy_to_xywh(converted_xyxy)
        self.assertEqual(converted_back, b1_xywh)

    def test_iou_matrix(self):
        """Tests pairwise IoU matrix calculation for N x M boxes."""
        boxes1 = [
            [0.0, 0.0, 10.0, 10.0],
            [10.0, 10.0, 20.0, 20.0]
        ]
        boxes2 = [
            [0.0, 0.0, 10.0, 10.0],
            [5.0, 0.0, 15.0, 10.0],
            [100.0, 100.0, 110.0, 110.0]
        ]
        mat = calculate_iou_matrix(boxes1, boxes2)
        self.assertEqual(mat.shape, (2, 3))
        self.assertAlmostEqual(mat[0, 0], 1.0, places=6)
        self.assertAlmostEqual(mat[0, 1], 1.0 / 3.0, places=6)
        self.assertEqual(mat[0, 2], 0.0)
        self.assertEqual(mat[1, 2], 0.0)


class TestMatchingAndMetrics(unittest.TestCase):

    def setUp(self):
        self.class_names = {
            0: "tomato_healthy",
            1: "tomato_bacterial_spot",
            2: "tomato_early_blight",
            3: "tomato_late_blight"
        }

    def test_class_aware_matching(self):
        """
        Predictions must NEVER be matched to ground-truth boxes of a different class,
        even if the spatial overlap is 100%.
        """
        # GT is class 2 (Early Blight)
        gts = [
            {"class_id": 2, "box": [10.0, 10.0, 100.0, 100.0]}
        ]
        # Pred is class 1 (Bacterial Spot) at the exact same coordinates
        preds = [
            {"class_id": 1, "box": [10.0, 10.0, 100.0, 100.0], "confidence": 0.95}
        ]

        result = match_predictions_to_ground_truth(gts, preds, class_names=self.class_names)
        # Must NOT match because classes differ!
        self.assertEqual(len(result["matched_pairs"]), 0)
        self.assertEqual(len(result["unmatched_predictions"]), 1)
        self.assertEqual(len(result["unmatched_ground_truths"]), 1)

    def test_one_to_one_highest_iou_matching(self):
        """
        If two predictions overlap a single ground-truth box of the same class,
        the prediction with highest IoU must be paired, and the second remains unmatched.
        """
        gts = [
            {"class_id": 2, "box": [0.0, 0.0, 10.0, 10.0]}
        ]
        preds = [
            # Partial overlap (IoU = 1/3)
            {"class_id": 2, "box": [5.0, 0.0, 15.0, 10.0], "confidence": 0.70},
            # Perfect overlap (IoU = 1.0)
            {"class_id": 2, "box": [0.0, 0.0, 10.0, 10.0], "confidence": 0.90}
        ]

        result = match_predictions_to_ground_truth(gts, preds, class_names=self.class_names)
        self.assertEqual(len(result["matched_pairs"]), 1)
        matched = result["matched_pairs"][0]
        self.assertAlmostEqual(matched["iou"], 1.0, places=6)
        self.assertEqual(matched["confidence"], 0.90)

        # The other prediction is unmatched
        self.assertEqual(len(result["unmatched_predictions"]), 1)
        self.assertEqual(len(result["unmatched_ground_truths"]), 0)

    def test_mean_iou_calculation(self):
        """Tests overall Mean IoU computation."""
        matches = [
            {"iou": 1.0},
            {"iou": 0.5},
            {"iou": 0.9}
        ]
        mean_iou = calculate_mean_iou(matches)
        self.assertAlmostEqual(mean_iou, (1.0 + 0.5 + 0.9) / 3.0, places=6)

        # Empty matches
        self.assertEqual(calculate_mean_iou([]), 0.0)

    def test_iou_thresholds(self):
        """Tests counting and percentage calculation for IoU >= 0.50 and IoU >= 0.75."""
        matches = [
            {"iou": 0.85},  # >= 0.75, >= 0.50
            {"iou": 0.60},  # >= 0.50, < 0.75
            {"iou": 0.40},  # < 0.50
            {"iou": 0.95}   # >= 0.75, >= 0.50
        ]
        # Total = 4
        # >= 0.50: 3 out of 4 (75.0%)
        # >= 0.75: 2 out of 4 (50.0%)
        thresh = calculate_iou_thresholds(matches, thresholds=[0.50, 0.75])
        self.assertEqual(thresh[0.50]["count"], 3)
        self.assertAlmostEqual(thresh[0.50]["percentage"], 75.0, places=4)
        self.assertEqual(thresh[0.75]["count"], 2)
        self.assertAlmostEqual(thresh[0.75]["percentage"], 50.0, places=4)

    def test_per_class_iou(self):
        """Tests computing Mean IoU separately per disease class."""
        matches = [
            {"class_id": 0, "iou": 0.90},
            {"class_id": 0, "iou": 0.80},
            {"class_id": 1, "iou": 0.70}
        ]
        per_class = calculate_per_class_iou(matches, self.class_names)
        self.assertEqual(len(per_class), len(self.class_names))

        c0 = next(c for c in per_class if c["class_id"] == 0)
        self.assertEqual(c0["evaluated_detections"], 2)
        self.assertAlmostEqual(c0["mean_iou"], 0.85, places=5)
        self.assertAlmostEqual(c0["min_iou"], 0.80, places=5)
        self.assertAlmostEqual(c0["max_iou"], 0.90, places=5)

        c1 = next(c for c in per_class if c["class_id"] == 1)
        self.assertEqual(c1["evaluated_detections"], 1)
        self.assertAlmostEqual(c1["mean_iou"], 0.70, places=5)

        # Classes with no detections return 0.0 safely
        c2 = next(c for c in per_class if c["class_id"] == 2)
        self.assertEqual(c2["evaluated_detections"], 0)
        self.assertEqual(c2["mean_iou"], 0.0)

    def test_iou_summary_structure(self):
        """Tests the comprehensive summary output structure."""
        matches = [
            {"class_id": 0, "iou": 0.90},
            {"class_id": 1, "iou": 0.80}
        ]
        summary = calculate_iou_summary(matches, self.class_names)
        self.assertIn("mean_iou", summary)
        self.assertIn("mean_iou_pct", summary)
        self.assertIn("min_iou", summary)
        self.assertIn("max_iou", summary)
        self.assertIn("evaluated_detections", summary)
        self.assertIn("iou_gte_50_pct", summary)
        self.assertIn("iou_gte_75_pct", summary)
        self.assertIn("per_class_iou", summary)
        self.assertEqual(summary["evaluated_detections"], 2)


if __name__ == "__main__":
    unittest.main()
