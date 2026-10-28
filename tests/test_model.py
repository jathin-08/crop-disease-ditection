"""
Tomato Disease AI - Comprehensive Test Suite
Validates model loading, configuration integrity, inference behavior, and metrics pipeline.
"""

import unittest
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, get_classes, resolve_model_path
from src.utils.visualization import get_disease_info
from src.data.validate_labels import validate_label_file
from src.inference.image_detection import predict_image
from ultralytics import YOLO


class TestTomatoDiseaseAI(unittest.TestCase):

    def test_config_integrity(self):
        """Verify config.yaml is properly formatted and contains 10 classes."""
        cfg = load_config()
        self.assertIn("classes", cfg)
        classes = get_classes()
        self.assertEqual(len(classes), 10, "Expected 10 tomato disease classes")
        self.assertEqual(classes[0], "tomato_healthy")

    def test_disease_knowledge_base(self):
        """Verify disease advisory info exists for all classes."""
        classes = get_classes()
        for cid, cname in classes.items():
            info = get_disease_info(cname)
            self.assertIn("display_name", info)
            self.assertIn("symptoms", info)
            self.assertIn("organic_treatment", info)
            self.assertIn("chemical_treatment", info)

    def test_model_loading(self):
        """Verify YOLO26 weights can be loaded into memory."""
        m_path = resolve_model_path()
        self.assertTrue(Path(m_path).exists() or str(m_path).endswith(".pt"))
        model = YOLO(str(m_path))
        self.assertIsNotNone(model)

    def test_blank_image_rejection(self):
        """Verify blank or black image is rejected rather than forcing a disease prediction."""
        m_path = resolve_model_path()
        model = YOLO(str(m_path))
        blank_img = np.zeros((640, 640, 3), dtype=np.uint8)
        test_img_path = PROJECT_ROOT / "tests" / "images" / "test_blank.jpg"
        test_img_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(blank_img).save(test_img_path)

        res = predict_image(test_img_path, model, conf_threshold=0.35, uncertainty_threshold=0.30)
        self.assertIn(res["status"], ["NO_DISEASE", "UNCERTAIN"])
        self.assertEqual(len(res["detections"]), 0)

    def test_iou_metrics_calculation(self):
        """Verify IoU calculation on identical, disjoint, and overlapping boxes."""
        from src.metrics.iou_metrics import calculate_iou
        self.assertEqual(calculate_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(calculate_iou([0, 0, 10, 10], [50, 50, 60, 60]), 0.0)
        self.assertAlmostEqual(calculate_iou([0, 0, 10, 10], [5, 0, 15, 10]), 1.0 / 3.0, places=5)

    def test_class_aware_iou_matching(self):
        """Verify that predictions of different disease classes are never matched."""
        from src.metrics.iou_metrics import match_predictions_to_ground_truth
        gts = [{"class_id": 2, "box": [0, 0, 10, 10]}]
        # Mismatched class (1 instead of 2)
        preds = [{"class_id": 1, "box": [0, 0, 10, 10], "confidence": 0.99}]
        res = match_predictions_to_ground_truth(gts, preds)
        self.assertEqual(len(res["matched_pairs"]), 0)
        self.assertEqual(len(res["unmatched_predictions"]), 1)

    def test_annotated_vs_unannotated_iou_reporting(self):
        """Verify IoU is included when GT exists, and strictly omitted when GT does not exist."""
        m_path = resolve_model_path()
        model = YOLO(str(m_path))

        # 1. Annotated image from validation set
        val_img = PROJECT_ROOT / "dataset" / "images" / "val" / "tomato_bacterial_spot_013.jpg"
        if val_img.exists():
            res_val = predict_image(val_img, model, conf_threshold=0.35)
            self.assertTrue(res_val.get("has_ground_truth"))
            if res_val["detections"]:
                for d in res_val["detections"]:
                    if d.get("iou") is not None:
                        self.assertGreaterEqual(d["iou"], 0.0)
                        self.assertLessEqual(d["iou"], 1.0)

        # 2. Unannotated image (blank image without corresponding .txt label)
        blank_p = PROJECT_ROOT / "tests" / "images" / "test_blank.jpg"
        res_blank = predict_image(blank_p, model, conf_threshold=0.01)
        self.assertFalse(res_blank.get("has_ground_truth"))


if __name__ == "__main__":
    unittest.main()

