# 🍅 Tomato Leaf Disease Model Evaluation Summary
**Timestamp:** `2026-10-05 22:21:05` | **Model:** `best.pt` | **Status:** `READY`

## Overall Model Performance
| Metric | Score | Percentage |
| :--- | :---: | :---: |
| **mAP @ 0.50** | `0.3850` | **38.50%** |
| **mAP @ 0.50:0.95** | `0.3203` | **32.03%** |
| **Precision (P)** | `0.5569` | **55.69%** |
| **Recall (R)** | `0.2702` | **27.02%** |
| **F1-Score** | `0.3638` | **36.38%** |
| **Mean IoU** | `0.8105` | **81.05%** |
| **IoU >= 0.50** | `93.33%` | **93.33%** |
| **IoU >= 0.75** | `83.33%` | **83.33%** |
| **Evaluated Detections** | `30` | - |

## Per-Class Disease Performance
| Class ID | Disease Name | Precision | Recall | F1-Score | mAP@50 | mAP@50:95 | Mean IoU | IoU>=0.50 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | **tomato_healthy** | 1.0000 | 0.4207 | 0.5923 | 0.8064 | 0.6619 | 0.8854 | 100.0% |
| 1 | **tomato_bacterial_spot** | 0.4313 | 0.6667 | 0.5237 | 0.7412 | 0.6670 | 0.8725 | 100.0% |
| 2 | **tomato_early_blight** | 0.1340 | 0.3333 | 0.1911 | 0.2097 | 0.1701 | 0.6780 | 66.7% |
| 3 | **tomato_late_blight** | 0.5643 | 0.4619 | 0.5080 | 0.6964 | 0.5683 | 0.6676 | 66.7% |
| 4 | **tomato_leaf_mold** | 1.0000 | 0.0000 | 0.0000 | 0.4211 | 0.3484 | 0.8412 | 100.0% |
| 5 | **tomato_septoria_leaf_spot** | 0.1476 | 0.5272 | 0.2306 | 0.1909 | 0.1718 | 0.8636 | 100.0% |
| 6 | **tomato_spider_mites** | 1.0000 | 0.0000 | 0.0000 | 0.1756 | 0.1405 | 0.8183 | 100.0% |
| 7 | **tomato_target_spot** | 0.2918 | 0.2918 | 0.2918 | 0.3110 | 0.2754 | 0.9170 | 100.0% |
| 8 | **tomato_mosaic_virus** | 0.0000 | 0.0000 | 0.0000 | 0.1518 | 0.1042 | 0.7920 | 100.0% |
| 9 | **tomato_yellow_leaf_curl_virus** | 1.0000 | 0.0000 | 0.0000 | 0.1462 | 0.0949 | 0.7699 | 100.0% |
