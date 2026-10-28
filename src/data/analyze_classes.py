"""
Tomato Disease AI - Class Distribution Analyzer
Analyzes class representation, annotations-per-class, and generates distribution graphs.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def analyze_class_distribution(
    dataset_dir: Path,
    class_names: Dict[int, str],
    output_graph_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Computes image counts and annotation counts per class across train, val, test.
    """
    labels_dir = dataset_dir / "labels"
    splits = ["train", "val", "test"]
    
    # Class ID -> split -> count
    annotation_counts: Dict[int, Dict[str, int]] = {
        cid: {s: 0 for s in splits} for cid in class_names.keys()
    }
    image_counts_per_class: Dict[int, Dict[str, int]] = {
        cid: {s: 0 for s in splits} for cid in class_names.keys()
    }
    split_totals: Dict[str, int] = {s: 0 for s in splits}

    for split in splits:
        split_lbl_dir = labels_dir / split
        if not split_lbl_dir.exists():
            continue
        
        lbl_files = list(split_lbl_dir.glob("*.txt"))
        split_totals[split] = len(lbl_files)

        for lfile in lbl_files:
            classes_in_file = set()
            try:
                with open(lfile, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            if cid in annotation_counts:
                                annotation_counts[cid][split] += 1
                                classes_in_file.add(cid)
            except Exception:
                pass
            
            for cid in classes_in_file:
                image_counts_per_class[cid][split] += 1

    # Format summary
    class_summary = []
    unrepresented_classes = []
    for cid, cname in class_names.items():
        total_annos = sum(annotation_counts[cid].values())
        total_imgs = sum(image_counts_per_class[cid].values())
        if total_imgs == 0:
            unrepresented_classes.append(cname)
        class_summary.append({
            "class_id": cid,
            "class_name": cname,
            "train_images": image_counts_per_class[cid]["train"],
            "val_images": image_counts_per_class[cid]["val"],
            "test_images": image_counts_per_class[cid]["test"],
            "total_images": total_imgs,
            "train_annotations": annotation_counts[cid]["train"],
            "val_annotations": annotation_counts[cid]["val"],
            "test_annotations": annotation_counts[cid]["test"],
            "total_annotations": total_annos
        })

    # Render Graph
    if output_graph_path:
        output_graph_path.parent.mkdir(parents=True, exist_ok=True)
        names = [c["class_name"].replace("tomato_", "").replace("_", " ").title() for c in class_summary]
        train_vals = [c["train_annotations"] for c in class_summary]
        val_vals = [c["val_annotations"] for c in class_summary]
        test_vals = [c["test_annotations"] for c in class_summary]

        x = np.arange(len(names))
        width = 0.25

        plt.figure(figsize=(14, 7), dpi=200)
        plt.bar(x - width, train_vals, width, label="Train Annotations", color="#2e7d32", alpha=0.85)
        plt.bar(x, val_vals, width, label="Val Annotations", color="#1976d2", alpha=0.85)
        plt.bar(x + width, test_vals, width, label="Test Annotations", color="#f57c00", alpha=0.85)

        plt.xlabel("Tomato Disease Class", fontsize=11, fontweight="bold")
        plt.ylabel("Total Object Annotations", fontsize=11, fontweight="bold")
        plt.title("Tomato Disease AI - Class Distribution Across Splits", fontsize=14, fontweight="bold", pad=15)
        plt.xticks(x, names, rotation=35, ha="right", fontsize=9)
        plt.legend(frameon=True, facecolor="#ffffff", framealpha=0.9)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_graph_path, bbox_inches="tight")
        plt.close()

    return {
        "class_summary": class_summary,
        "split_totals": split_totals,
        "unrepresented_classes": unrepresented_classes
    }
