# YOLO Weights Directory

This directory stores deep learning checkpoints for the Plant Disease YOLO system:

- `yolo26n.pt` / `yolo11n.pt` / `yolov8n.pt`: Base pretrained weights used for transfer learning.
- `best.pt`: The optimal checkpoint automatically exported after training (`src/train.py`).

### How to obtain weights:
1. **Pretrained Base Model**: Ultralytics automatically downloads base weights (such as `yolo11n.pt` or `yolov8n.pt`) upon the first training run.
2. **Custom Trained Model**: When you run `python src/train.py`, your best-performing weights from `runs/train/plant_disease_yolo/weights/best.pt` are automatically copied to `models/best.pt`.
