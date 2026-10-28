"""
Tomato Disease AI - Real-Time Webcam Stream Detection
Executes real-time YOLO26 inference on camera feed with bounding boxes, disease names,
confidence metrics, diagnostic HUD, and graceful keyboard termination.
"""

import sys
import time
import argparse
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, resolve_model_path, get_classes
from src.utils.visualization import annotate_detection, draw_hud_card, get_disease_info


def run_webcam(
    camera_index: int = 0,
    model_path: str = "",
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.50
):
    """Starts live camera inference loop."""
    target_weights = model_path if model_path else str(resolve_model_path())
    print(f"\n[*] Loading YOLO26 Weights: {target_weights}")
    model = YOLO(target_weights)
    classes = model.names if hasattr(model, "names") else get_classes()

    print(f"[*] Opening Camera Stream (Device index: {camera_index})...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[!] ERROR: Could not open camera {camera_index}. Please check connection or permissions.")
        return

    device = "0" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 75)
    print(" LIVE WEBCAM INFERENCE ACTIVE")
    print(" Controls: Press 'q' or 'ESC' on the video window to safely exit.")
    print("=" * 75 + "\n")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[!] Empty frame received or camera disconnected.")
            break

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Predict
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            verbose=False
        )

        detections = []
        for r in results:
            boxes = r.boxes
            for b in boxes:
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                xyxy = b.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                cname = classes.get(cls_id, f"Class_{cls_id}")

                detections.append({
                    "class_name": cname,
                    "confidence": conf
                })

                frame = annotate_detection(
                    frame=frame,
                    box=(x1, y1, x2, y2),
                    class_name=cname,
                    confidence=conf
                )

        hud_dets = [{"disease_name": d["class_name"], "confidence": d["confidence"]} for d in detections]
        frame = draw_hud_card(frame, hud_dets, fps=fps)

        cv2.imshow("Tomato Leaf AI - Live Monitor (Press 'q' to Exit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            print("[*] User initiated exit.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[+] Webcam stream terminated cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Run Live YOLO26 Webcam Inference")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--model", type=str, default="", help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="IoU threshold")
    args = parser.parse_args()

    run_webcam(
        camera_index=args.camera,
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )


if __name__ == "__main__":
    main()
