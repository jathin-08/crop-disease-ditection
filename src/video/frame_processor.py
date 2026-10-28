"""
Tomato Disease AI - Frame Processor
==============================================================================
Handles video stream capture, metadata validation, frame reading,
and multi-platform codec output management with fallback encoding.
==============================================================================
"""

from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import cv2
import numpy as np


class VideoFrameProcessor:
    """Manages OpenCV VideoCapture and VideoWriter pipelines with robust validation."""

    def __init__(
        self,
        source_path: str,
        output_path: Optional[str] = None,
        output_fps: Optional[float] = None
    ):
        self.source_path = str(source_path)
        self.output_path = str(output_path) if output_path else None
        self.cap = cv2.VideoCapture(self.source_path)
        self.is_valid = self.cap.isOpened()

        if not self.is_valid:
            self.fps = 25.0
            self.total_frames = 0
            self.width = 0
            self.height = 0
            self.duration = 0.0
            self.writer = None
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        if self.fps <= 0 or np.isnan(self.fps):
            self.fps = 25.0

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = (self.total_frames / self.fps) if self.fps > 0 else 0.0

        self.writer = None
        if self.output_path and self.width > 0 and self.height > 0:
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            target_fps = output_fps if output_fps is not None else self.fps
            
            # Try robust codecs (mp4v -> XVID)
            codecs_to_try = ["mp4v", "XVID"]
            for codec in codecs_to_try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                w = cv2.VideoWriter(self.output_path, fourcc, target_fps, (self.width, self.height))
                if w.isOpened():
                    self.writer = w
                    break
                else:
                    w.release()

            if self.writer is None or not self.writer.isOpened():
                # Final fallback
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.writer = cv2.VideoWriter(self.output_path, fourcc, target_fps, (self.width, self.height))

    def get_info(self) -> Dict[str, Any]:
        """Returns structured metadata about the video source."""
        return {
            "source_path": self.source_path,
            "filename": Path(self.source_path).name,
            "is_valid": self.is_valid,
            "width": self.width,
            "height": self.height,
            "resolution": f"{self.width} × {self.height}",
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames,
            "duration_seconds": round(self.duration, 2),
            "duration_formatted": f"{int(self.duration // 60):02d}:{self.duration % 60:04.1f}"
        }

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads next video frame."""
        if not self.is_valid:
            return False, None
        return self.cap.read()

    def write_frame(self, frame: np.ndarray) -> None:
        """Writes frame to destination video."""
        if self.writer is not None and self.writer.isOpened():
            self.writer.write(frame)

    def release(self) -> None:
        """Releases capture and writer resources."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        if self.writer is not None:
            self.writer.release()
