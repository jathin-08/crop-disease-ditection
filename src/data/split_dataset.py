"""
Tomato Disease AI - Dataset Splitting Utility
Performs class-stratified train/val/test partitioning (70/20/10) with leakage prevention.
"""

import os
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def split_tomato_dataset(
    source_dir: Path,
    dest_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
    seed: int = 42
) -> Dict[str, int]:
    """
    Splits images and corresponding .txt labels from source_dir into dest_dir/images and dest_dir/labels.
    """
    random.seed(seed)
    
    # Destination directories
    for s in ["train", "val", "test"]:
        (dest_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (dest_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

    # Group by class from label files
    class_to_pairs = defaultdict(list)
    image_files = list((source_dir / "images").glob("*.jpg")) + list((source_dir / "images").glob("*.png"))

    for img_p in image_files:
        lbl_p = source_dir / "labels" / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue
        
        # Read primary class id
        try:
            with open(lbl_p, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                cid = int(first_line.split()[0]) if first_line else -1
        except Exception:
            cid = -1
            
        class_to_pairs[cid].append((img_p, lbl_p))

    counts = {"train": 0, "val": 0, "test": 0}

    for cid, pairs in class_to_pairs.items():
        random.shuffle(pairs)
        n = len(pairs)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        
        train_pairs = pairs[:n_train]
        val_pairs = pairs[n_train:n_train + n_val]
        test_pairs = pairs[n_train + n_val:]
        if not test_pairs and len(pairs) >= 3:
            test_pairs = [val_pairs.pop()]

        split_map = {
            "train": train_pairs,
            "val": val_pairs,
            "test": test_pairs
        }

        for split_name, assigned in split_map.items():
            for img, lbl in assigned:
                shutil.copy2(img, dest_dir / "images" / split_name / img.name)
                shutil.copy2(lbl, dest_dir / "labels" / split_name / lbl.name)
                counts[split_name] += 1

    return counts
