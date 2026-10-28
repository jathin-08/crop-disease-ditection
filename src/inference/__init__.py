"""Tomato Disease AI - Inference Services Package"""

from src.inference.image_detection import predict_image
from src.inference.video_detection import process_video_stream
from src.inference.webcam_detection import run_webcam

__all__ = ["predict_image", "process_video_stream", "run_webcam"]
