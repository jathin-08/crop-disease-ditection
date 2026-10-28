"""
Tomato Disease AI - Master Dataset Validation Pipeline
Conducts comprehensive 25-point validation check on images, labels, splits, data.yaml,
class distribution, bounding box ranges, and duplicate image leakage.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, get_classes
from src.utils.logger import setup_logger
from src.data.validate_labels import validate_label_file
from src.data.check_duplicates import check_dataset_duplicates
from src.data.analyze_classes import analyze_class_distribution
from src.data.visualize_annotations import visualize_dataset_annotations

logger = setup_logger("validate_dataset")


def validate_master_dataset(dataset_dir: Path, data_yaml_path: Path) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Executes the 25-point verification suite.
    Returns: (passed, report_dict, error_messages)
    """
    cfg = load_config()
    configured_classes = get_classes()
    nc = len(configured_classes)

    errors = []
    warnings = []
    
    # 1. Verify data.yaml
    if not data_yaml_path.exists():
        errors.append(f"[Check 21/24] data.yaml not found at: {data_yaml_path}")
        return False, {}, errors

    import yaml
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        dy = yaml.safe_load(f)

    if dy.get("nc") != nc:
        errors.append(f"[Check 22] Class count in data.yaml (nc={dy.get('nc')}) does not match config.yaml (nc={nc}).")

    yaml_names = dy.get("names", {})
    if isinstance(yaml_names, list):
        yaml_names = {i: name for i, name in enumerate(yaml_names)}
    
    for cid, cname in configured_classes.items():
        if cid not in yaml_names or yaml_names[cid] != cname:
            errors.append(f"[Check 23] Class mismatch for ID {cid}: expected '{cname}', got '{yaml_names.get(cid)}'.")

    # 2. Check Directory Structure
    splits = ["train", "val", "test"]
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    for s in splits:
        if not (images_dir / s).exists():
            errors.append(f"[Check 24] Missing image directory for split: {images_dir / s}")
        if not (labels_dir / s).exists():
            errors.append(f"[Check 24] Missing label directory for split: {labels_dir / s}")

    # 3. Check Images and Labels Pairings
    all_images: Dict[str, List[Path]] = {}
    corrupted_images = []
    unsupported_formats = []
    missing_labels = []
    empty_labels = []
    invalid_boxes = []

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for s in splits:
        s_img_dir = images_dir / s
        s_lbl_dir = labels_dir / s
        if not s_img_dir.exists() or not s_lbl_dir.exists():
            continue

        images = [p for p in s_img_dir.glob("*") if not p.name.startswith(".")]
        all_images[s] = images

        for img_p in images:
            # Check format
            if img_p.suffix.lower() not in valid_extensions:
                unsupported_formats.append(str(img_p))
                continue

            # Check corruption
            try:
                with Image.open(img_p) as im:
                    im.verify()
            except Exception:
                corrupted_images.append(str(img_p))
                continue

            # Check matching label
            lbl_p = s_lbl_dir / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                missing_labels.append(f"{s}/{img_p.name} -> missing {lbl_p.name}")
            else:
                is_valid, b_errs, _ = validate_label_file(lbl_p, img_p, num_classes=nc)
                if not is_valid:
                    invalid_boxes.extend(b_errs)

        # Check orphan labels
        for lbl_p in s_lbl_dir.glob("*.txt"):
            matching_imgs = [s_img_dir / f"{lbl_p.stem}{ext}" for ext in valid_extensions]
            if not any(p.exists() for p in matching_imgs):
                errors.append(f"[Check 4] Orphan label file with no matching image: {s}/{lbl_p.name}")

    if corrupted_images:
        errors.append(f"[Check 12] Corrupted images found ({len(corrupted_images)}): {corrupted_images[:3]}")
    if unsupported_formats:
        errors.append(f"[Check 13] Unsupported image formats ({len(unsupported_formats)}): {unsupported_formats[:3]}")
    if missing_labels:
        errors.append(f"[Check 2] Missing label files for images ({len(missing_labels)}): {missing_labels[:3]}")
    if invalid_boxes:
        errors.append(f"[Check 5-11, 25] Bounding box / syntax errors ({len(invalid_boxes)}): {invalid_boxes[:3]}")

    # 4. Check Duplicates & Data Leakage
    dup_res = check_dataset_duplicates(dataset_dir)
    if dup_res["has_leakage"]:
        errors.append(f"[Check 15-17] Data Leakage detected: {len(dup_res['dataset_leakage'])} duplicate images across splits.")
    if dup_res["has_duplicates"]:
        warnings.append(f"[Check 14] Intra-split duplicate images found: {len(dup_res['intra_split_duplicates'])}")

    # 5. Class Distribution Analysis
    graphs_dir = PROJECT_ROOT / "results" / "graphs"
    dist_graph = graphs_dir / "class_distribution.png"
    class_stats = analyze_class_distribution(dataset_dir, configured_classes, output_graph_path=dist_graph)

    # Check minimum required samples per class
    min_samples = cfg.get("dataset", {}).get("min_samples_per_class", 5)
    for cinfo in class_stats["class_summary"]:
        if cinfo["total_images"] < min_samples:
            warnings.append(f"[Check 18-20] Class '{cinfo['class_name']}' has only {cinfo['total_images']} images (recommended >= {min_samples}).")
        if cinfo["train_annotations"] == 0:
            errors.append(f"[Check 18] Class '{cinfo['class_name']}' has ZERO training annotations!")

    # 6. Sample Annotations Montage
    sample_annot_path = graphs_dir / "sample_annotations.png"
    visualize_dataset_annotations(dataset_dir, configured_classes, sample_annot_path)

    # Totals
    train_count = len(all_images.get("train", []))
    val_count = len(all_images.get("val", []))
    test_count = len(all_images.get("test", []))

    if train_count == 0:
        errors.append("[Check 1] Training image set is empty!")
    if val_count == 0:
        errors.append("[Check 1] Validation image set is empty!")

    passed = len(errors) == 0

    report = {
        "status": "PASSED" if passed else "FAILED",
        "total_images": train_count + val_count + test_count,
        "split_counts": {
            "train": train_count,
            "val": val_count,
            "test": test_count
        },
        "num_classes": nc,
        "class_summary": class_stats["class_summary"],
        "unrepresented_classes": class_stats["unrepresented_classes"],
        "duplicates": {
            "intra_split": len(dup_res["intra_split_duplicates"]),
            "cross_split_leakage": len(dup_res["dataset_leakage"])
        },
        "errors": errors,
        "warnings": warnings
    }

    return passed, report, errors


def generate_reports(report: Dict[str, Any], output_txt: Path, output_json: Path):
    """Saves structured validation reports to disk."""
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = [
        "=" * 75,
        " TOMATO LEAF DISEASE AI - DATASET VALIDATION REPORT",
        "=" * 75,
        f"Validation Status    : {report['status']}",
        f"Total Verified Images: {report['total_images']} (Train: {report['split_counts']['train']}, Val: {report['split_counts']['val']}, Test: {report['split_counts']['test']})",
        f"Configured Classes   : {report['num_classes']}",
        "-" * 75,
        "CLASS-WISE SAMPLE AND ANNOTATION DISTRIBUTION:",
        f"  {'ID':<3} | {'Class Name':<32} | {'Train Img':<10} | {'Val Img':<8} | {'Test Img':<8} | {'Total Boxes'}",
        "  " + "-" * 72,
    ]
    for c in report.get("class_summary", []):
        lines.append(
            f"  {c['class_id']:<3} | {c['class_name']:<32} | {c['train_images']:<10} | {c['val_images']:<8} | {c['test_images']:<8} | {c['total_annotations']}"
        )
    lines.append("-" * 75)

    if report.get("errors"):
        lines.append("CRITICAL VALIDATION ERRORS:")
        for err in report["errors"]:
            lines.append(f"  [x] {err}")
        lines.append("-" * 75)

    if report.get("warnings"):
        lines.append("VALIDATION WARNINGS:")
        for w in report["warnings"]:
            lines.append(f"  [!] {w}")
        lines.append("-" * 75)

    lines.append("=" * 75 + "\n")

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Validate Tomato Leaf Disease YOLO Dataset")
    parser.add_argument("--dataset", type=str, default=str(PROJECT_ROOT / "dataset"), help="Path to dataset/")
    parser.add_argument("--data", type=str, default=str(PROJECT_ROOT / "dataset" / "data.yaml"), help="Path to data.yaml")
    args = parser.parse_args()

    ds_path = Path(args.dataset)
    dy_path = Path(args.data)

    logger.info("Executing 25-Point Comprehensive Dataset Pre-flight Validation...")
    passed, report, errors = validate_master_dataset(ds_path, dy_path)

    report_txt = PROJECT_ROOT / "results" / "reports" / "dataset_report.txt"
    report_json = PROJECT_ROOT / "dataset" / "dataset_report.json"
    generate_reports(report, report_txt, report_json)

    if not passed:
        print("\n" + "!" * 75)
        print("DATASET VALIDATION FAILED")
        print("!" * 75)
        print(f"Total Critical Errors: {len(errors)}")
        for e in errors:
            print(f"  [-] {e}")
        print("\nPlease fix the above dataset/annotation/split issues before training.")
        print(f"Detailed diagnostics saved to: {report_txt.relative_to(PROJECT_ROOT)}")
        print("!" * 75 + "\n")
        sys.exit(1)
    else:
        print("\n" + "=" * 75)
        print("DATASET VALIDATION PASSED - READY FOR YOLO26 TRAINING")
        print("=" * 75)
        print(f"Total Verified Images: {report['total_images']}")
        print(f"Classes Verified: {report['num_classes']}")
        print(f"Report saved: {report_txt.relative_to(PROJECT_ROOT)}")
        print("=" * 75 + "\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
