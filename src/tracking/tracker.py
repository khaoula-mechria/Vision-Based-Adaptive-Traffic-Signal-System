"""Multi-object tracker (SORT: Kalman prediction + Hungarian IoU association).

Behaviour worth knowing
-----------------------
* IDs are monotonically increasing per tracker instance and are never reused or reset by
  ``reset_on_gap``-style events inside a stream, so a vehicle keeps one id while it is tracked.
* A track survives ``max_age`` frames without a detection (occlusion). With
  ``output_coasting > 0`` such tracks are still *returned* (flagged ``is_coasting``) for that
  many frames, so a briefly occluded stopped car does not vanish from the queue count.
* Invalid detections (NaN, non-positive size) are dropped and counted, not propagated.
* Dead tracks are removed from ``self.tracks``: memory is bounded by ``max_age``.
* A gap in timestamps larger than ``max_gap_s`` (camera interruption) drops all tracks
  (association across such a gap is unreliable); ids keep increasing, so a vehicle seen again
  after the outage gets a new id rather than colliding with an old one.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None

from ..perception.detection_result import DetectionResult
from .kalman_filter import KalmanBoxTracker, iou_batch
from .track import Track

logger = logging.getLogger(__name__)


class MultiObjectTracker:
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        output_coasting: int = 0,
        max_gap_s: Optional[float] = 2.0,
    ) -> None:
        if max_age < 1 or min_hits < 1 or not 0 < iou_threshold <= 1 or output_coasting < 0:
            raise ValueError("invalid tracker parameters")
        if linear_sum_assignment is None:
            raise ImportError("scipy is required for tracking: pip install scipy")
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.output_coasting = output_coasting
        self.max_gap_s = max_gap_s
        self.trackers: List[KalmanBoxTracker] = []
        self.tracks: Dict[int, Track] = {}
        self.frame_count = 0
        self.dropped_invalid = 0
        self._next_id = 1
        self._last_ts: Optional[float] = None

    # ------------------------------------------------------------------------------ update
    def update(self, detection_result: DetectionResult) -> List[Track]:
        """Advance one frame. Returns confirmed tracks (plus coasting ones if enabled)."""
        self.frame_count += 1
        ts = float(detection_result.timestamp)
        if self._last_ts is not None:
            if ts < self._last_ts:
                logger.warning("non-monotonic timestamp %.3f < %.3f", ts, self._last_ts)
                ts = self._last_ts
            elif self.max_gap_s is not None and ts - self._last_ts > self.max_gap_s and self.trackers:
                logger.warning("frame gap %.2fs > %.2fs: dropping %d tracks", ts - self._last_ts,
                               self.max_gap_s, len(self.trackers))
                self.trackers.clear()
                self.tracks.clear()
        self._last_ts = ts

        dets, det_objs = self._valid_detections(detection_result)

        # predict
        preds = np.zeros((len(self.trackers), 4))
        keep: List[int] = []
        for i, trk in enumerate(self.trackers):
            box = trk.predict()
            if np.all(np.isfinite(box)):
                preds[len(keep)] = box
                keep.append(i)
        self.trackers = [self.trackers[i] for i in keep]
        preds = preds[: len(keep)]

        matches, unmatched_dets, _ = self._associate(dets, preds)

        for d, t in matches:
            trk = self.trackers[t]
            trk.update(dets[d, :4])
            det = det_objs[d]
            det.track_id = trk.id
            track = self.tracks[trk.id]
            track.update_position(dets[d, :4].copy(), ts)
            track.confidence = float(dets[d, 4])
            track.class_name, track.class_id = det.class_name, det.class_id
            track.is_vehicle, track.is_pedestrian = det.is_vehicle, det.is_pedestrian
            track.is_cyclist, track.is_emergency = det.is_cyclist, det.is_emergency

        for d in unmatched_dets:
            det = det_objs[d]
            trk = KalmanBoxTracker(dets[d, :4], track_id=self._next_id)
            self._next_id += 1
            self.trackers.append(trk)
            det.track_id = trk.id
            track = Track(
                track_id=trk.id, class_name=det.class_name, class_id=det.class_id, bbox=dets[d, :4].copy(),
                confidence=float(det.confidence), is_vehicle=det.is_vehicle, is_pedestrian=det.is_pedestrian,
                is_cyclist=det.is_cyclist, is_emergency=det.is_emergency, first_seen=ts, last_seen=ts,
            )
            track.update_position(dets[d, :4].copy(), ts)
            self.tracks[trk.id] = track

        # retire dead trackers and their metadata
        alive = []
        for trk in self.trackers:
            if trk.time_since_update > self.max_age:
                self.tracks.pop(trk.id, None)
            else:
                alive.append(trk)
        self.trackers = alive

        out: List[Track] = []
        for trk in self.trackers:
            confirmed = trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            visible = trk.time_since_update < 1
            coasting = 0 < trk.time_since_update <= self.output_coasting and trk.hits >= self.min_hits
            if (visible and confirmed) or coasting:
                track = self.tracks[trk.id]
                track.bbox = trk.get_state()
                track.hits, track.age = trk.hits, trk.age
                track.time_since_update = trk.time_since_update
                track.is_coasting = not visible
                out.append(track)
        return out

    def _valid_detections(self, result: DetectionResult) -> Tuple[np.ndarray, list]:
        rows, objs = [], []
        for d in result.detections:
            row = [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2, d.confidence]
            if not np.all(np.isfinite(row)) or d.bbox.x2 <= d.bbox.x1 or d.bbox.y2 <= d.bbox.y1:
                self.dropped_invalid += 1
                continue
            rows.append(row)
            objs.append(d)
        return (np.asarray(rows, float) if rows else np.empty((0, 5))), objs

    def _associate(self, dets: np.ndarray, preds: np.ndarray):
        n_d, n_t = len(dets), len(preds)
        if n_t == 0:
            return [], list(range(n_d)), []
        if n_d == 0:
            return [], [], list(range(n_t))
        iou = iou_batch(dets[:, :4], preds[:, :4])
        rows, cols = linear_sum_assignment(1.0 - iou)
        matches, used_d, used_t = [], set(), set()
        for r, c in zip(rows, cols):
            if iou[r, c] >= self.iou_threshold:
                matches.append((int(r), int(c)))
                used_d.add(int(r))
                used_t.add(int(c))
        return (matches, [d for d in range(n_d) if d not in used_d],
                [t for t in range(n_t) if t not in used_t])

    # ---------------------------------------------------------------------------- queries
    def get_active_tracks(self) -> List[Track]:
        return [self.tracks[t.id] for t in self.trackers if t.time_since_update < self.max_age and t.id in self.tracks]

    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        return self.tracks.get(track_id)

    def get_vehicle_tracks(self) -> List[Track]:
        return [t for t in self.get_active_tracks() if t.is_vehicle]

    def get_pedestrian_tracks(self) -> List[Track]:
        return [t for t in self.get_active_tracks() if t.is_pedestrian]

    def get_emergency_tracks(self) -> List[Track]:
        return [t for t in self.get_active_tracks() if t.is_emergency]

    def reset(self) -> None:
        """Clear tracks. The id counter is *not* reset, so ids stay unique per instance."""
        self.trackers.clear()
        self.tracks.clear()
        self.frame_count = 0
        self._last_ts = None

    def get_statistics(self) -> dict:
        active = self.get_active_tracks()
        return {
            "frame_count": self.frame_count,
            "total_tracks_created": self._next_id - 1,
            "active_tracks": len(active),
            "vehicle_tracks": sum(t.is_vehicle for t in active),
            "pedestrian_tracks": sum(t.is_pedestrian for t in active),
            "cyclist_tracks": sum(t.is_cyclist for t in active),
            "emergency_tracks": sum(t.is_emergency for t in active),
            "dropped_invalid_detections": self.dropped_invalid,
        }
