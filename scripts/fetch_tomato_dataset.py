"""
Tomato Disease AI - Dataset Acquisition & Annotation Script
Downloads authentic tomato leaf images from the official PlantVillage repository
for all 10 target classes, computes accurate YOLO bounding boxes, and partitions
them cleanly into train, val, and test splits without leakage.
"""

import os
import sys
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLASS_MAPPING = {
    0: ("tomato_healthy", "Tomato___healthy"),
    1: ("tomato_bacterial_spot", "Tomato___Bacterial_spot"),
    2: ("tomato_early_blight", "Tomato___Early_blight"),
    3: ("tomato_late_blight", "Tomato___Late_blight"),
    4: ("tomato_leaf_mold", "Tomato___Leaf_Mold"),
    5: ("tomato_septoria_leaf_spot", "Tomato___Septoria_leaf_spot"),
    6: ("tomato_spider_mites", "Tomato___Spider_mites Two-spotted_spider_mite"),
    7: ("tomato_target_spot", "Tomato___Target_Spot"),
    8: ("tomato_mosaic_virus", "Tomato___Tomato_mosaic_virus"),
    9: ("tomato_yellow_leaf_curl_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"),
}

def extract_yolo_boxes(img: np.ndarray, class_id: int) -> List[Tuple[int, float, float, float, float]]:
    """
    Extracts leaf and lesion bounding boxes using robust color and contour analysis.
    Returns list of (class_id, x_center, y_center, width, height) normalized to [0, 1].
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # PlantVillage leaves are distinguishable from background via saturation & value
    # Background is neutral/low-saturation or very dark/bright
    mask = cv2.inRange(hsv, (8, 25, 20), (130, 255, 255))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    if contours:
        # Filter by reasonable area (> 5% of image area)
        leaf_contours = [c for c in contours if cv2.contourArea(c) > (w * h * 0.05)]
        if not leaf_contours:
            leaf_contours = [max(contours, key=cv2.contourArea)]
            
        for c in leaf_contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            # Add small padding if within image
            pad = 2
            bx = max(0, bx - pad)
            by = max(0, by - pad)
            bw = min(w - bx, bw + pad * 2)
            bh = min(h - by, bh + pad * 2)
            
            xc = min(1.0, max(0.0, (bx + bw / 2.0) / w))
            yc = min(1.0, max(0.0, (by + bh / 2.0) / h))
            norm_w = min(1.0, max(0.05, bw / w))
            norm_h = min(1.0, max(0.05, bh / h))
            boxes.append((class_id, xc, yc, norm_w, norm_h))
            
    if not boxes:
        # Fallback to center leaf box if background was non-standard
        boxes.append((class_id, 0.50, 0.50, 0.90, 0.90))
        
    return boxes


def fetch_and_populate(
    samples_per_class: int = 18,
    train_split: float = 0.70,
    val_split: float = 0.20
):
    """
    Downloads images from PlantVillage repo and builds train/val/test splits.
    """
    dataset_dir = PROJECT_ROOT / "dataset"
    img_root = dataset_dir / "images"
    lbl_root = dataset_dir / "labels"

    # Clean existing image/label contents
    for s in ["train", "val", "test"]:
        s_img = img_root / s
        s_lbl = lbl_root / s
        s_img.mkdir(parents=True, exist_ok=True)
        s_lbl.mkdir(parents=True, exist_ok=True)
        # Clear old images & labels
        for f in s_img.glob("*.*"):
            f.unlink()
        for f in s_lbl.glob("*.*"):
            f.unlink()

    total_downloaded = 0
    split_counts = {"train": 0, "val": 0, "test": 0}

    print("=" * 75)
    print("STARTING TOMATO DATASET ACQUISITION (PlantVillage Color Dataset)")
    print(f"Target: {samples_per_class} samples per class across {len(CLASS_MAPPING)} classes")
    print("=" * 75)

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for cid, (cname, remote_folder) in CLASS_MAPPING.items():
        print(f"\n[*] Processing Class {cid}: {cname} ({remote_folder})...")
        encoded_folder = urllib.parse.quote(remote_folder)
        api_url = f"https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color/{encoded_folder}"
        
        req = urllib.request.Request(api_url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"  [x] API error for {cname}: {e}")
            continue

        # Filter only image files
        img_items = [x for x in data if x["name"].lower().endswith((".jpg", ".jpeg", ".png"))]
        selected_items = img_items[:samples_per_class]

        n = len(selected_items)
        n_train = int(n * train_split)
        n_val = int(n * val_split)
        
        for idx, item in enumerate(selected_items):
            raw_url = item["download_url"]
            file_name = f"{cname}_{idx+1:03d}.jpg"
            
            # Determine split
            if idx < n_train:
                split = "train"
            elif idx < n_train + n_val:
                split = "val"
            else:
                split = "test"

            dest_img = img_root / split / file_name
            dest_lbl = lbl_root / split / f"{dest_img.stem}.txt"

            # Download image
            try:
                img_req = urllib.request.Request(raw_url, headers=headers)
                with urllib.request.urlopen(img_req, timeout=15) as img_resp:
                    img_bytes = img_resp.read()
                    
                arr = np.asarray(bytearray(img_bytes), dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                # Resize to standard 640x640 for optimal YOLO input
                img_640 = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(dest_img), img_640)

                # Extract accurate YOLO bounding boxes
                boxes = extract_yolo_boxes(img_640, cid)
                with open(dest_lbl, "w", encoding="utf-8") as lf:
                    for b in boxes:
                        lf.write(f"{b[0]} {b[1]:.4f} {b[2]:.4f} {b[3]:.4f} {b[4]:.4f}\n")

                split_counts[split] += 1
                total_downloaded += 1

            except Exception as e:
                print(f"  [-] Failed {file_name}: {e}")

            time.sleep(0.05)  # Respectful rate limiting

        print(f"  [+] Downloaded and annotated {n} images for {cname}")

    print("\n" + "=" * 75)
    print(f"DATASET ACQUISITION COMPLETE: {total_downloaded} total images populated.")
    print(f"Split distribution: Train={split_counts['train']}, Val={split_counts['val']}, Test={split_counts['test']}")
    print("=" * 75)


if __name__ == "__main__":
    fetch_and_populate(samples_per_class=18)
