"""
Tomato Disease AI - Annotation Visualizer
Overlays ground-truth YOLO labels on sample images and exports visual verification grid.
"""

from pathlib import Path
from typing import Dict, Optional, List
import cv2
import numpy as np
from PIL import Image

from src.utils.visualization import annotate_detection, get_disease_info

def visualize_dataset_annotations(
    dataset_dir: Path,
    class_names: Dict[int, str],
    output_path: Path,
    samples_per_split: int = 2
) -> None:
    """
    Renders ground-truth boxes onto sample images from train/val/test splits.
    Saves a montage grid to output_path.
    """
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    splits = ["train", "val", "test"]

    annotated_samples = []

    for split in splits:
        img_split = images_dir / split
        lbl_split = labels_dir / split
        if not img_split.exists() or not lbl_split.exists():
            continue

        images = sorted(list(img_split.glob("*.jpg")) + list(img_split.glob("*.png")))
        for img_p in images[:samples_per_split]:
            lbl_p = lbl_split / f"{img_p.stem}.txt"
            img = cv2.imread(str(img_p))
            if img is None:
                continue

            h, w = img.shape[:2]

            if lbl_p.exists():
                with open(lbl_p, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            xc, yc, bw, bh = map(float, parts[1:5])
                            x1 = int((xc - bw / 2.0) * w)
                            y1 = int((yc - bh / 2.0) * h)
                            x2 = int((xc + bw / 2.0) * w)
                            y2 = int((yc + bh / 2.0) * h)
                            
                            cname = class_names.get(cid, f"Class_{cid}")
                            img = annotate_detection(
                                frame=img,
                                box=(x1, y1, x2, y2),
                                class_name=cname,
                                confidence=1.0  # Ground truth
                            )

            # Add split watermark
            cv2.putText(
                img, f"SPLIT: {split.upper()} | {img_p.name}",
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )
            annotated_samples.append(cv2.resize(img, (400, 400)))

    if not annotated_samples:
        return

    # Create montage grid
    n = len(annotated_samples)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    # Fill empty slots if any
    grid_img = np.zeros((rows * 400, cols * 400, 3), dtype=np.uint8)
    for idx, sample in enumerate(annotated_samples):
        r = idx // cols
        c = idx % cols
        grid_img[r * 400:(r + 1) * 400, c * 400:(c + 1) * 400] = sample

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid_img)
