"""
Tomato Disease AI - YOLO Label Validation Module
Validates normalized YOLO bounding box syntax, coordinates, and class bounds.
"""

from pathlib import Path
from typing import Dict, List, Any, Tuple


def validate_label_file(
    label_path: Path,
    image_path: Path,
    num_classes: int = 10,
    min_box_size: float = 0.01,
    max_box_size: float = 1.05
) -> Tuple[bool, List[str], List[Dict[str, Any]]]:
    """
    Validates a single YOLO label file.
    Returns: (is_valid, error_list, parsed_boxes)
    """
    errors = []
    boxes = []

    if not label_path.exists():
        errors.append(f"Missing label file: {label_path.name}")
        return False, errors, boxes

    if label_path.stat().st_size == 0:
        errors.append(f"Empty label file: {label_path.name}")
        return False, errors, boxes

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        errors.append(f"Empty label content in: {label_path.name}")
        return False, errors, boxes

    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"Line {idx+1} in {label_path.name} has {len(parts)} tokens, expected 5.")
            continue

        try:
            cls_id = int(parts[0])
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            errors.append(f"Line {idx+1} in {label_path.name} has non-numeric coordinates.")
            continue

        # Class ID range
        if cls_id < 0 or cls_id >= num_classes:
            errors.append(f"Line {idx+1} class {cls_id} out of bounds [0, {num_classes-1}].")

        # Range checks
        if not (0.0 <= xc <= 1.0) or not (0.0 <= yc <= 1.0):
            errors.append(f"Line {idx+1} center ({xc:.3f}, {yc:.3f}) outside [0, 1].")

        if w <= 0.0 or h <= 0.0:
            errors.append(f"Line {idx+1} zero or negative dimensions (w={w}, h={h}).")

        if w > max_box_size or h > max_box_size:
            errors.append(f"Line {idx+1} dimension exceeds limit (w={w}, h={h}).")

        if w < min_box_size or h < min_box_size:
            errors.append(f"Line {idx+1} warning: extremely small box (w={w}, h={h}).")

        boxes.append({
            "class_id": cls_id,
            "x_center": xc,
            "y_center": yc,
            "width": w,
            "height": h
        })

    is_valid = len(errors) == 0
    return is_valid, errors, boxes
