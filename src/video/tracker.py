"""
Tomato Disease AI - Video Detection Object Tracker
==============================================================================
Tracks detected diseased foliage lesions across consecutive video frames using:
  1. Class consistency: Enforces that tracks only associate with identical disease classes
  2. Bounding Box IoU: Overlap between consecutive frames
  3. Centroid Distance: Distance between bounding box centers to handle rapid movement
==============================================================================
"""

from typing import List, Dict, Any, Tuple, Optional
import math
import numpy as np


class SimpleBoxTracker:
    """
    Lightweight, robust multi-object tracker for tomato foliage video inspection.
    Reuses existing track associations while ensuring class consistency, IoU thresholding,
    and centroid proximity matching.
    """

    def __init__(
        self,
        max_disappeared: int = 15,
        iou_thresh: float = 0.50,
        enforce_class_consistency: bool = True,
        centroid_weight: float = 0.20
    ):
        """
        Parameters:
            max_disappeared: Max consecutive frames a track can be missing before deletion.
            iou_thresh: Minimum IoU overlap required to match an existing track.
            enforce_class_consistency: If True, only match detections sharing the same disease class.
            centroid_weight: Proportional weighting for centroid distance proximity.
        """
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.max_disappeared = max_disappeared
        self.iou_thresh = iou_thresh
        self.enforce_class_consistency = enforce_class_consistency
        self.centroid_weight = centroid_weight

        # Global registry of all observed tracks for reporting
        self.all_tracks_summary: Dict[int, Dict[str, Any]] = {}

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Updates active tracks with incoming frame detections.
        Each detection is a dict with:
          - 'box': [x1, y1, x2, y2]
          - 'class_name': str
          - 'class_id': int (optional)
          - 'confidence': float (optional)

        Returns list of detections enriched with 'track_id' (int) and 'track_label' (str).
        """
        if not detections:
            # Mark existing tracks as disappeared
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]
            return []

        # If no active tracks, initialize all detections as new tracks
        if not self.tracks:
            updated_dets = []
            for det in detections:
                tid = self.next_id
                self.next_id += 1
                cname = det.get("class_name", "Unknown")
                cid = det.get("class_id", -1)
                conf = float(det.get("confidence", 0.0))
                bx = list(det["box"])

                self.tracks[tid] = {
                    "track_id": tid,
                    "box": bx,
                    "class_name": cname,
                    "class_id": cid,
                    "disappeared": 0,
                    "total_frames": 1,
                    "confidences": [conf]
                }
                self.all_tracks_summary[tid] = {
                    "track_id": tid,
                    "class_name": cname,
                    "class_id": cid,
                    "first_box": bx,
                    "last_box": bx,
                    "total_frames": 1,
                    "confidences": [conf]
                }

                det_copy = dict(det)
                det_copy["track_id"] = tid
                det_copy["track_label"] = f"#{tid:03d}"
                updated_dets.append(det_copy)
            return updated_dets

        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid]["box"] for tid in track_ids]
        track_classes = [self.tracks[tid]["class_name"] for tid in track_ids]
        det_boxes = [d["box"] for d in detections]
        det_classes = [d.get("class_name", "Unknown") for d in detections]

        num_tracks = len(track_ids)
        num_dets = len(detections)

        # Compute cost / affinity matrix
        affinity_mat = np.zeros((num_tracks, num_dets), dtype=float)

        for i, (tb, tc) in enumerate(zip(track_boxes, track_classes)):
            tc_centroid = self._box_centroid(tb)
            tb_diag = math.hypot(max(1.0, tb[2] - tb[0]), max(1.0, tb[3] - tb[1]))

            for j, (db, dc) in enumerate(zip(det_boxes, det_classes)):
                # Class consistency check
                if self.enforce_class_consistency and tc != dc and tc != "Unknown" and dc != "Unknown":
                    affinity_mat[i, j] = -1.0
                    continue

                iou = self._box_iou(tb, db)
                if iou < self.iou_thresh:
                    # Check centroid distance fallback if close enough
                    dc_centroid = self._box_centroid(db)
                    dist = math.hypot(tc_centroid[0] - dc_centroid[0], tc_centroid[1] - dc_centroid[1])
                    norm_dist = dist / max(1.0, tb_diag)
                    if norm_dist < 0.6 and iou > 0.15:
                        affinity_mat[i, j] = iou - (self.centroid_weight * norm_dist)
                    else:
                        affinity_mat[i, j] = -1.0
                else:
                    affinity_mat[i, j] = iou

        matched_tracks = set()
        matched_dets = set()
        track_to_det = {}

        if affinity_mat.size > 0:
            while True:
                max_val = np.max(affinity_mat)
                if max_val < 0.0:
                    break
                i, j = np.unravel_index(np.argmax(affinity_mat), affinity_mat.shape)
                tid = track_ids[i]
                track_to_det[i] = j
                matched_tracks.add(i)
                matched_dets.add(j)

                # Update track state
                conf = float(detections[j].get("confidence", 0.0))
                bx = list(det_boxes[j])
                self.tracks[tid]["box"] = bx
                self.tracks[tid]["disappeared"] = 0
                self.tracks[tid]["total_frames"] += 1
                self.tracks[tid]["confidences"].append(conf)

                # Update summary
                if tid in self.all_tracks_summary:
                    self.all_tracks_summary[tid]["last_box"] = bx
                    self.all_tracks_summary[tid]["total_frames"] += 1
                    self.all_tracks_summary[tid]["confidences"].append(conf)

                affinity_mat[i, :] = -1.0
                affinity_mat[:, j] = -1.0

        # Unmatched existing tracks increment disappeared
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]

        # Build output detections
        updated_dets = []
        for j, det in enumerate(detections):
            det_copy = dict(det)
            if j in matched_dets:
                # Find track id that matched this det
                matched_tid = None
                for t_idx, d_idx in track_to_det.items():
                    if d_idx == j:
                        matched_tid = track_ids[t_idx]
                        break
                det_copy["track_id"] = matched_tid
                det_copy["track_label"] = f"#{matched_tid:03d}" if matched_tid else "#000"
            else:
                # New track initialization
                tid = self.next_id
                self.next_id += 1
                cname = det.get("class_name", "Unknown")
                cid = det.get("class_id", -1)
                conf = float(det.get("confidence", 0.0))
                bx = list(det["box"])

                self.tracks[tid] = {
                    "track_id": tid,
                    "box": bx,
                    "class_name": cname,
                    "class_id": cid,
                    "disappeared": 0,
                    "total_frames": 1,
                    "confidences": [conf]
                }
                self.all_tracks_summary[tid] = {
                    "track_id": tid,
                    "class_name": cname,
                    "class_id": cid,
                    "first_box": bx,
                    "last_box": bx,
                    "total_frames": 1,
                    "confidences": [conf]
                }
                det_copy["track_id"] = tid
                det_copy["track_label"] = f"#{tid:03d}"

            updated_dets.append(det_copy)

        return updated_dets

    @staticmethod
    def _box_centroid(box: List[float]) -> Tuple[float, float]:
        """Calculates center coordinates (cx, cy) of [x1, y1, x2, y2]."""
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    @staticmethod
    def _box_iou(boxA: List[float], boxB: List[float]) -> float:
        """Computes spatial Intersection over Union (IoU) of two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
        boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

        denom = float(boxAArea + boxBArea - interArea)
        return interArea / denom if denom > 0.0 else 0.0

    def get_track_summary(self) -> List[Dict[str, Any]]:
        """Returns structured summary of all tracked objects throughout video."""
        summary = []
        for tid, data in sorted(self.all_tracks_summary.items(), key=lambda x: x[0]):
            confs = data.get("confidences", [0.0])
            avg_c = float(np.mean(confs)) if confs else 0.0
            max_c = float(np.max(confs)) if confs else 0.0
            summary.append({
                "track_id": tid,
                "track_label": f"#{tid:03d}",
                "class_name": data.get("class_name", "Unknown"),
                "class_id": data.get("class_id", -1),
                "total_frames_seen": data.get("total_frames", 1),
                "avg_confidence": avg_c,
                "max_confidence": max_c
            })
        return summary
