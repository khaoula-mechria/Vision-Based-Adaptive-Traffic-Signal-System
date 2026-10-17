"""Lane-level queue, flow and waiting-time estimation from tracked vehicles.

What is measured, and how
-------------------------
* ``vehicle_count``   tracked vehicles whose box centre lies in the lane polygon.
* ``queue_length``    vehicles that are *stopped* (moving less than ``stopped_speed``
  body-lengths per second for at least ``min_stop_s``) and have not crossed the stop line.
  Speed is normalised by box size, so the stopped/moving decision needs no camera calibration.
* ``arrival_rate``    new track ids entering the lane; ``departure_rate`` unique ids crossing
  the stop line (a vehicle is never counted twice, even if its track coasts through occlusion).
* waiting time        time a track has spent stopped, taken from the track's own timestamps
  (never wall-clock), so replayed video and simulation time behave identically.
* ``occupancy``       summed box area inside the lane / lane polygon area (pixel space).

Limitations: distances in metres are only produced when ``meters_per_pixel`` is configured and
are approximate (a single camera without a homography has perspective distortion); counts are
reliable only while vehicles are detected, so heavy occlusion under-counts queues.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..tracking.track import Track
from ..utils.geometry import rect_polygon
from .lane import CongestionLevel, Lane, LaneState

logger = logging.getLogger(__name__)


class QueueEstimator:
    def __init__(
        self,
        stopped_speed: float = 0.15,
        min_stop_s: float = 1.0,
        meters_per_pixel: Optional[float] = None,
        vehicle_length_meters: float = 4.5,
        congestion_thresholds: Optional[Mapping[str, int]] = None,
        rate_window_s: float = 60.0,
    ) -> None:
        if stopped_speed <= 0 or min_stop_s < 0 or rate_window_s <= 0:
            raise ValueError("stopped_speed and rate_window_s must be > 0, min_stop_s >= 0")
        self.stopped_speed = stopped_speed
        self.min_stop_s = min_stop_s
        self.meters_per_pixel = meters_per_pixel
        self.vehicle_length_meters = vehicle_length_meters
        th = congestion_thresholds or {"low": 5, "medium": 10, "high": 20}
        self.congestion_thresholds = (int(th["low"]), int(th["medium"]), int(th["high"]))
        self.rate_window_s = rate_window_s
        self.lanes: Dict[int, Lane] = {}
        self._track_lane: Dict[int, int] = {}
        self._prev_side: Dict[int, float] = {}
        self._stopped_since: Dict[int, float] = {}
        self._last_ts: Optional[float] = None

    # ---- configuration -----------------------------------------------------------------
    def add_lane(
        self,
        lane_id: int,
        direction: str,
        x1: float = 0.0,
        y1: float = 0.0,
        x2: float = 0.0,
        y2: float = 0.0,
        lane_type: str = "through",
        length_meters: float = 100.0,  # kept for API compatibility; unused without calibration
        polygon: Optional[Sequence[Sequence[float]]] = None,
        stop_line: Optional[Sequence[Sequence[float]]] = None,
        approach: Optional[str] = None,
    ) -> Lane:
        if lane_id in self.lanes:
            raise ValueError(f"duplicate lane id {lane_id}")
        poly = polygon if polygon is not None else rect_polygon(x1, y1, x2, y2)
        sl = (tuple(stop_line[0]), tuple(stop_line[1])) if stop_line is not None else None
        lane = Lane(
            lane_id=lane_id, direction=direction, polygon=np.asarray(poly, float), stop_line=sl,  # type: ignore[arg-type]
            lane_type=lane_type, approach=approach, meters_per_pixel=self.meters_per_pixel,
            vehicle_length_meters=self.vehicle_length_meters, congestion_thresholds=self.congestion_thresholds,
        )
        self.lanes[lane_id] = lane
        return lane

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "QueueEstimator":
        """Build from a dict like ``{"lanes": [{"id": 0, "approach": "N", "polygon": [...],
        "stop_line": [[x, y], [x, y]]}, ...], "stopped_speed": 0.15, ...}``."""
        est = cls(**{k: v for k, v in cfg.items() if k in (
            "stopped_speed", "min_stop_s", "meters_per_pixel", "vehicle_length_meters",
            "congestion_thresholds", "rate_window_s")})
        for ln in cfg.get("lanes", []):
            est.add_lane(
                lane_id=int(ln["id"]), direction=str(ln.get("direction", ln.get("approach", "?"))),
                polygon=ln["polygon"], stop_line=ln.get("stop_line"), approach=ln.get("approach"),
                lane_type=ln.get("type", "through"),
            )
        return est

    def setup_default_intersection(self, frame_width: int, frame_height: int, num_lanes_per_direction: int = 2) -> None:
        """Demo layout (rectangles around the frame centre, no stop lines). Real deployments
        should supply polygons and stop lines via ``from_config`` / ``add_lane``."""
        cx, cy, lw, gap = frame_width / 2, frame_height / 2, 50, 50
        n, lane_id = num_lanes_per_direction, 0
        for d, rect in (
            ("north", lambda i: (cx - (n - i) * lw, 0, cx - (n - i - 1) * lw, cy - gap)),
            ("south", lambda i: (cx + i * lw, cy + gap, cx + (i + 1) * lw, frame_height)),
            ("east", lambda i: (cx + gap, cy - (n - i) * lw, frame_width, cy - (n - i - 1) * lw)),
            ("west", lambda i: (0, cy + i * lw, cx - gap, cy + (i + 1) * lw)),
        ):
            for i in range(n):
                self.add_lane(lane_id, d, *rect(i), lane_type="through")  # type: ignore[misc]
                lane_id += 1

    # ---- update ------------------------------------------------------------------------
    def _assign(self, cx: float, cy: float) -> Optional[Lane]:
        for lane in self.lanes.values():
            if lane.contains_point(cx, cy):
                return lane
        return None

    def update(self, tracks: Sequence[Track], timestamp: Optional[float] = None) -> Dict[int, LaneState]:
        """Update all lanes from this frame's tracks. ``timestamp`` must be the frame time
        (the same clock the tracks use); wall-clock is only a fallback."""
        if timestamp is None:
            timestamp = time.time()
        if self._last_ts is not None and timestamp < self._last_ts:
            logger.warning("non-monotonic frame time %.3f < %.3f", timestamp, self._last_ts)
            timestamp = self._last_ts
        self._last_ts = timestamp

        per_lane: Dict[int, List[Track]] = {lid: [] for lid in self.lanes}
        for t in tracks:
            if not t.is_vehicle:
                continue
            cx, cy = t.get_center()
            lane = self._assign(cx, cy)
            prev_lane_id = self._track_lane.get(t.track_id)
            # stop-line crossing (checked against the lane the track was in, even if now beyond it)
            check = self.lanes.get(prev_lane_id) if prev_lane_id is not None else lane
            if check is not None and check.stop_line is not None:
                side = check.side_of_stop_line((cx, cy))
                prev = self._prev_side.get(t.track_id)
                if side is not None and prev is not None and prev > 0 >= side:
                    check.record_departure(t.track_id, timestamp, self.rate_window_s)
                if side is not None:
                    self._prev_side[t.track_id] = side
            if lane is None:
                continue
            if check is lane and lane.has_crossed(t.track_id):
                continue  # already past the stop line: not part of the queue
            if lane.stop_line is not None and (lane.side_of_stop_line((cx, cy)) or 1.0) <= 0:
                continue  # beyond the stop line
            lane.record_arrival(t.track_id, timestamp, self.rate_window_s)
            lane.touch(t.track_id, timestamp)
            self._track_lane[t.track_id] = lane.lane_id
            per_lane[lane.lane_id].append(t)

        states: Dict[int, LaneState] = {}
        for lid, lane in self.lanes.items():
            vehicles = per_lane[lid]
            waits: List[float] = []
            coverage = 0.0
            farthest_px: Optional[float] = None
            queue = 0
            for t in vehicles:
                x1, y1, x2, y2 = (float(v) for v in t.bbox)
                coverage += max(0.0, x2 - x1) * max(0.0, y2 - y1)
                if self._is_stopped(t, timestamp):
                    queue += 1
                    waits.append(t.get_current_waiting_time(timestamp))
                    cx, cy = t.get_center()
                    if lane.stop_line is not None:
                        d = abs(lane.side_of_stop_line((cx, cy)) or 0.0) / (
                            np.hypot(*(np.subtract(*lane.stop_line[::-1])) ) + 1e-9)
                        farthest_px = d if farthest_px is None else max(farthest_px, d)
            states[lid] = lane.update_state(len(vehicles), queue, waits, coverage, farthest_px,
                                            timestamp, self.rate_window_s)
        self._prune(timestamp, {t.track_id for t in tracks})
        return states

    def _is_stopped(self, t: Track, now: float) -> bool:
        """Stopped = slower than ``stopped_speed`` body-lengths/s for >= ``min_stop_s``."""
        x1, y1, x2, y2 = (float(v) for v in t.bbox)
        scale = max(x2 - x1, y2 - y1, 1.0)
        slow = (t.get_speed() / scale) < self.stopped_speed and len(t.position_history) >= 2
        if slow:
            since = self._stopped_since.setdefault(t.track_id, now)
            if now - since >= self.min_stop_s:
                t.start_waiting(since)
                return True
            return False
        self._stopped_since.pop(t.track_id, None)
        t.stop_waiting(now)
        return False

    def _prune(self, now: float, live_ids: set) -> None:
        for d in (self._track_lane, self._prev_side, self._stopped_since):
            for k in [k for k in d if k not in live_ids and len(d) > 2000]:
                d.pop(k, None)

    # ---- queries -----------------------------------------------------------------------
    def get_lane_state(self, lane_id: int) -> Optional[LaneState]:
        lane = self.lanes.get(lane_id)
        return lane.state if lane else None

    def get_all_lane_states(self) -> Dict[int, LaneState]:
        return {lid: lane.state for lid, lane in self.lanes.items()}

    def approach_states(self) -> Dict[str, Dict[str, float]]:
        """Aggregate lanes per signal approach: total queue, longest wait, mean wait."""
        out: Dict[str, Dict[str, float]] = {}
        for lane in self.lanes.values():
            a = out.setdefault(lane.approach or "?", {"queue": 0.0, "max_wait_s": 0.0, "_wsum": 0.0, "_n": 0.0})
            s = lane.state
            a["queue"] += s.queue_length
            a["max_wait_s"] = max(a["max_wait_s"], s.max_waiting_time)
            a["_wsum"] += s.average_waiting_time * s.queue_length
            a["_n"] += s.queue_length
        for a in out.values():
            a["mean_wait_s"] = a["_wsum"] / a["_n"] if a["_n"] else 0.0
            del a["_wsum"], a["_n"]
        return out

    def get_congestion_map(self) -> Dict[int, CongestionLevel]:
        return {lid: lane.state.congestion_level for lid, lane in self.lanes.items()}

    def identify_imbalanced_lanes(self, threshold: float = 2.0) -> List[int]:
        return [lid for lid, lane in self.lanes.items() if abs(lane.get_flow_imbalance()) > threshold]

    def get_total_queue_length(self) -> int:
        return sum(lane.state.queue_length for lane in self.lanes.values())

    def get_max_waiting_time(self) -> float:
        return max((lane.state.max_waiting_time for lane in self.lanes.values()), default=0.0)

    def get_summary(self) -> dict:
        congestion = self.get_congestion_map()
        return {
            "total_lanes": len(self.lanes),
            "total_queue": self.get_total_queue_length(),
            "max_waiting_time": self.get_max_waiting_time(),
            "congested_lanes": sum(c in (CongestionLevel.HIGH, CongestionLevel.CRITICAL) for c in congestion.values()),
            "lanes": {lid: lane.to_dict() for lid, lane in self.lanes.items()},
        }

    def reset(self) -> None:
        for lane in self.lanes.values():
            lane.reset()
        self._track_lane.clear()
        self._prev_side.clear()
        self._stopped_since.clear()
        self._last_ts = None
