"""
Tomato Disease AI - Python Diagnostic API
Provides programmatic access to tomato leaf disease diagnosis.
"""

from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.utils.config import resolve_model_path, get_classes
from src.utils.visualization import get_disease_info


class TomatoDiseaseDetector:
    """Production Python API wrapper for YOLO26 tomato disease diagnosis."""

    def __init__(self, model_path: Union[str, Path, None] = None, conf: float = 0.35, iou: float = 0.50):
        self.weights = str(resolve_model_path(str(model_path) if model_path else None))
        self.model = YOLO(self.weights)
        self.conf = conf
        self.iou = iou
        self.classes = self.model.names if hasattr(self.model, "names") else get_classes()

    def diagnose(self, image: Union[str, Path, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """Diagnoses an image and returns structured medical foliage card."""
        results = self.model.predict(source=image, conf=self.conf, iou=self.iou, verbose=False)
        detections = []

        for r in results:
            for b in r.boxes:
                cid = int(b.cls[0])
                conf = float(b.conf[0])
                box = [int(v) for v in b.xyxy[0].cpu().numpy()]
                cname = self.classes.get(cid, f"Class_{cid}")
                info = get_disease_info(cname)

                detections.append({
                    "class_id": cid,
                    "disease_name": info["display_name"],
                    "scientific_name": info.get("scientific_name", ""),
                    "confidence": conf,
                    "bounding_box_xyxy": box,
                    "severity": info.get("severity", "Moderate"),
                    "organic_treatment": info.get("organic_treatment", ""),
                    "chemical_treatment": info.get("chemical_treatment", ""),
                    "prevention": info.get("prevention", "")
                })

        return {
            "status": "DISEASE_DETECTED" if detections else "HEALTHY_OR_CLEAR",
            "total_detections": len(detections),
            "detections": detections,
            "model_evaluated": self.weights
        }
