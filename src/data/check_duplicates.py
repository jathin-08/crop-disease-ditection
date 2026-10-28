"""
Tomato Disease AI - Image & Dataset Duplicate Checker
Detects exact file duplicates and cross-split dataset leakage via MD5 hashing.
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any
from PIL import Image

def compute_image_hash(image_path: Path) -> str:
    """Computes MD5 hash of image pixel buffer."""
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def check_dataset_duplicates(dataset_dir: Path) -> Dict[str, Any]:
    """
    Checks for intra-split and cross-split image duplicates.
    Returns summary dictionary.
    """
    images_dir = dataset_dir / "images"
    splits = ["train", "val", "test"]
    
    hash_to_files: Dict[str, List[Tuple[str, Path]]] = {}
    leakage_records = []
    intra_split_duplicates = []

    for split in splits:
        split_path = images_dir / split
        if not split_path.exists():
            continue
        
        for img_file in split_path.glob("*"):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                try:
                    h = compute_image_hash(img_file)
                    if h in hash_to_files:
                        prev_split, prev_path = hash_to_files[h][0]
                        if prev_split != split:
                            leakage_records.append({
                                "hash": h,
                                "file1": str(prev_path),
                                "split1": prev_split,
                                "file2": str(img_file),
                                "split2": split
                            })
                        else:
                            intra_split_duplicates.append({
                                "hash": h,
                                "file1": str(prev_path),
                                "file2": str(img_file),
                                "split": split
                            })
                        hash_to_files[h].append((split, img_file))
                    else:
                        hash_to_files[h] = [(split, img_file)]
                except Exception as e:
                    pass

    return {
        "total_unique_hashes": len(hash_to_files),
        "intra_split_duplicates": intra_split_duplicates,
        "dataset_leakage": leakage_records,
        "has_leakage": len(leakage_records) > 0,
        "has_duplicates": len(intra_split_duplicates) > 0
    }

if __name__ == "__main__":
    from src.utils.config import get_project_root
    res = check_dataset_duplicates(get_project_root() / "dataset")
    print(f"Dataset Leakage Count: {len(res['dataset_leakage'])}")
    print(f"Intra-split Duplicates: {len(res['intra_split_duplicates'])}")
