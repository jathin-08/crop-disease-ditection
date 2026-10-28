"""
Unit Tests for Central Metric Engine and Evaluation Pipeline
==============================================================================
Validates:
1. Real IoU calculation (identical, partial overlap, zero overlap)
2. Real Precision calculation: TP / (TP + FP) and division by zero safe
3. Real Recall calculation: TP / (TP + FN) and division by zero safe
4. Real F1 Score calculation: 2 * P * R / (P + R) and division by zero safe
5. Class-aware, 1-to-1 matching: Predictions match only same-class GT with highest IoU
6. Single-image evaluation (evaluate_image): Correct TP, FP, FN, IoU, Precision, Recall, F1
7. Predictions enrichment with bbox, iou, precision, recall, f1
8. Metrics loader (load_metrics_report) with project-relative paths
==============================================================================
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.metrics_engine import (
    calculate_iou,
    calculate_precision,
    calculate_recall,
    calculate_f1,
    match_predictions_to_ground_truth,
    evaluate_image,
    load_metrics_report
)


class TestMetricEngine(unittest.TestCase):

    def test_calculate_iou(self):
        """Test exact IoU calculation."""
        # 1. Identical boxes -> 1.0
        box_a = [0, 0, 100, 100]
        box_b = [0, 0, 100, 100]
        self.assertAlmostEqual(calculate_iou(box_a, box_b), 1.0, places=5)

        # 2. No overlap -> 0.0
        box_c = [200, 200, 300, 300]
        self.assertAlmostEqual(calculate_iou(box_a, box_c), 0.0, places=5)

        # 3. 50% width overlap: [0, 0, 100, 100] and [50, 0, 150, 100]
        # Inter = 50 * 100 = 5000
        # Union = 10000 + 10000 - 5000 = 15000
        # IoU = 5000 / 15000 = 1/3
        box_d = [50, 0, 150, 100]
        self.assertAlmostEqual(calculate_iou(box_a, box_d), 1.0 / 3.0, places=5)

    def test_precision_calculation(self):
        """Precision = TP / (TP + FP)"""
        # Normal
        self.assertAlmostEqual(calculate_precision(8, 2), 0.8, places=5)
        self.assertAlmostEqual(calculate_precision(10, 0), 1.0, places=5)
        # Division by zero
        self.assertAlmostEqual(calculate_precision(0, 0), 0.0, places=5)

    def test_recall_calculation(self):
        """Recall = TP / (TP + FN)"""
        # Normal
        self.assertAlmostEqual(calculate_recall(7, 3), 0.7, places=5)
        self.assertAlmostEqual(calculate_recall(0, 5), 0.0, places=5)
        # Division by zero
        self.assertAlmostEqual(calculate_recall(0, 0), 0.0, places=5)

    def test_f1_calculation(self):
        """F1 = 2 * P * R / (P + R)"""
        # P = 0.8, R = 0.6 -> F1 = 2 * 0.48 / 1.4 = 0.96 / 1.4 ~ 0.685714
        expected_f1 = (2 * 0.8 * 0.6) / (0.8 + 0.6)
        self.assertAlmostEqual(calculate_f1(0.8, 0.6), expected_f1, places=5)
        # Division by zero
        self.assertAlmostEqual(calculate_f1(0.0, 0.0), 0.0, places=5)

    def test_class_aware_matching(self):
        """Ensure boxes of different classes never match."""
        gts = [
            {"class_id": 1, "box": [10, 10, 100, 100]},
            {"class_id": 2, "box": [150, 150, 250, 250]}
        ]
        # Prediction has same coordinates as class 1, but labeled as class 2
        preds = [
            {"class_id": 2, "box": [10, 10, 100, 100], "confidence": 0.95}
        ]
        # Should not match GT of class 1 because class_id differs
        res = match_predictions_to_ground_truth(preds, gts)
        self.assertEqual(len(res["matched_pairs"]), 0)
        self.assertEqual(len(res["unmatched_predictions"]), 1)

    def test_highest_iou_one_to_one_matching(self):
        """Two predictions for one GT: only the highest IoU matches."""
        gt = [{"class_id": 5, "box": [0, 0, 100, 100]}]
        preds = [
            {"class_id": 5, "box": [10, 10, 100, 100], "confidence": 0.90}, # Higher IoU
            {"class_id": 5, "box": [50, 50, 100, 100], "confidence": 0.80}  # Lower IoU
        ]
        res = match_predictions_to_ground_truth(preds, gt)
        self.assertEqual(len(res["matched_pairs"]), 1)
        self.assertEqual(len(res["unmatched_predictions"]), 1)
        # The matched prediction should be the one with higher IoU
        matched_box = res["matched_pairs"][0]["predicted_box"]
        self.assertEqual(matched_box, [10.0, 10.0, 100.0, 100.0])

    def test_evaluate_image(self):
        """Test evaluate_image computes TP, FP, FN, P, R, F1 and enriches predictions."""
        gts = [
            {"class_id": 3, "box": [0, 0, 100, 100]},
            {"class_id": 3, "box": [200, 200, 300, 300]}
        ]
        preds = [
            {"class_id": 3, "box": [0, 0, 100, 100], "confidence": 0.92}, # Perfect match -> TP (IoU=1.0)
            {"class_id": 3, "box": [400, 400, 500, 500], "confidence": 0.85} # FP (no GT)
        ]
        # 1 TP, 1 FP (the second pred), 1 FN (the second GT was missed)
        eval_res = evaluate_image(preds, gts, iou_threshold=0.50)
        self.assertEqual(eval_res["tp"], 1)
        self.assertEqual(eval_res["fp"], 1)
        self.assertEqual(eval_res["fn"], 1)
        self.assertAlmostEqual(eval_res["precision"], 0.5, places=5)
        self.assertAlmostEqual(eval_res["recall"], 0.5, places=5)
        self.assertAlmostEqual(eval_res["f1"], 0.5, places=5)

        # Check that predictions are enriched with real calculated metrics
        self.assertAlmostEqual(preds[0]["iou"], 1.0, places=5)
        self.assertAlmostEqual(preds[0]["precision"], 0.5, places=5)
        self.assertAlmostEqual(preds[0]["recall"], 0.5, places=5)
        self.assertAlmostEqual(preds[0]["f1"], 0.5, places=5)
        self.assertIn("bbox", preds[0])

        self.assertAlmostEqual(preds[1]["iou"], 0.0, places=5)

    def test_load_metrics_report(self):
        """Test load_metrics_report successfully retrieves metrics.json or defaults."""
        data = load_metrics_report()
        self.assertIsInstance(data, dict)
        self.assertIn("precision", data)
        self.assertIn("recall", data)
        self.assertIn("f1", data)
        self.assertIn("mean_iou", data)


if __name__ == "__main__":
    unittest.main()
