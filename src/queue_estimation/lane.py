"""Lane data structure and per-lane state.

A lane is a polygon in image space with an optional stop line. Queue length is a *vehicle
count* (reliable from a single camera); metre values are only filled in when a calibration
(``meters_per_pixel``) is supplied and are approximate: a single un-calibrated camera has
perspective distortion, so pixel distances are not proportional to road distance.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from ..utils.geometry import (
    Point,
    as_polygon,
    point_in_polygon,
    polygon_area,
    polygon_centroid,
    rect_polygon,
    side_of_line,
)


class CongestionLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LaneState:
    queue_length: int = 0  # stopped vehicles waiting in the lane (not yet past the stop line)
    vehicle_count: int = 0  # all tracked vehicles in the lane (moving + stopped)
    queue_length_meters: Optional[float] = None  # only with calibration; approximate
    occupancy: float = 0.0  # fraction of lane polygon area covered by vehicle boxes (0-1)
    average_waiting_time: float = 0.0  # seconds, over stopped vehicles
    max_waiting_time: float = 0.0
    arrival_rate: float = 0.0  # vehicles per minute (new tracks entering the lane)
    departure_rate: float = 0.0  # vehicles per minute (crossing the stop line)
    congestion_level: CongestionLevel = CongestionLevel.LOW
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "queue_length": self.queue_length,
            "vehicle_count": self.vehicle_count,
            "queue_length_meters": self.queue_length_meters,
            "occupancy": self.occupancy,
            "average_waiting_time": self.average_waiting_time,
            "max_waiting_time": self.max_waiting_time,
            "arrival_rate": self.arrival_rate,
            "departure_rate": self.departure_rate,
            "congestion_level": self.congestion_level.value,
            "timestamp": self.timestamp,
        }


@dataclass
class Lane:
    lane_id: int
    direction: str  # compass label of the approach, e.g. "north"
    polygon: np.ndarray = field(default_factory=lambda: rect_polygon(0, 0, 1, 1))
    stop_line: Optional[Tuple[Point, Point]] = None  # directed segment; vehicles cross it leaving the lane
    lane_type: str = "through"
    approach: Optional[str] = None  # signal approach name used by the controller (default: initial of direction)
    meters_per_pixel: Optional[float] = None  # None => no metre estimates
    vehicle_length_meters: float = 4.5
    vehicle_gap_meters: float = 2.5
    congestion_thresholds: Tuple[int, int, int] = (5, 10, 20)  # medium, high, critical queue lengths

    state: LaneState = field(default_factory=LaneState)
    _arrivals: Deque[float] = field(default_factory=deque)
    _departures: Deque[float] = field(default_factory=deque)
    _crossed: Dict[int, float] = field(default_factory=dict)  # track id -> time it crossed the stop line
    _seen: Dict[int, float] = field(default_factory=dict)  # track id -> last time seen (arrival dedup)
    _approach_sign: float = 0.0

    def __post_init__(self) -> None:
        self.polygon = as_polygon(self.polygon)
        self._area = polygon_area(self.polygon)
        if self._area <= 0:
            raise ValueError(f"lane {self.lane_id}: degenerate polygon")
        if self.approach is None:
            self.approach = self.direction[:1].upper()
        if self.stop_line is not None:
            a, b = self.stop_line
            self._approach_sign = float(np.sign(side_of_line(polygon_centroid(self.polygon), a, b))) or 1.0

    # legacy rectangle accessors
    @classmethod
    def from_rect(cls, lane_id: int, direction: str, x1: float, y1: float, x2: float, y2: float, **kw) -> "Lane":
        return cls(lane_id=lane_id, direction=direction, polygon=rect_polygon(x1, y1, x2, y2), **kw)

    def contains_point(self, x: float, y: float) -> bool:
        return point_in_polygon(x, y, self.polygon)

    def side_of_stop_line(self, p: Point) -> Optional[float]:
        """>0 on the approaching side, <0 beyond the stop line; None without a stop line."""
        if self.stop_line is None:
            return None
        return side_of_line(p, *self.stop_line) * self._approach_sign

    # ---- history ---------------------------------------------------------------------
    def record_arrival(self, track_id: int, timestamp: float, window_s: float = 120.0) -> bool:
        """Count a track once per lane visit. Returns True if it is a new arrival."""
        new = track_id not in self._seen
        self._seen[track_id] = timestamp
        if new:
            self._arrivals.append(timestamp)
        self._trim(timestamp, window_s)
        return new

    def touch(self, track_id: int, timestamp: float) -> None:
        if track_id in self._seen:
            self._seen[track_id] = timestamp

    def record_departure(self, track_id: int, timestamp: float, window_s: float = 120.0) -> bool:
        if track_id in self._crossed:
            return False
        self._crossed[track_id] = timestamp
        self._departures.append(timestamp)
        self._trim(timestamp, window_s)
        return True

    def has_crossed(self, track_id: int) -> bool:
        return track_id in self._crossed

    def _trim(self, now: float, window_s: float, forget_s: float = 60.0) -> None:
        for dq in (self._arrivals, self._departures):
            while dq and dq[0] < now - window_s:
                dq.popleft()
        self._seen = {k: t for k, t in self._seen.items() if now - t <= forget_s}
        self._crossed = {k: t for k, t in self._crossed.items() if now - t <= 2 * forget_s}

    def rate_per_min(self, events: Deque[float], now: float, window_s: float) -> float:
        n = sum(1 for t in events if t >= now - window_s)
        return n * 60.0 / window_s

    # ---- state -----------------------------------------------------------------------
    def update_state(
        self,
        vehicle_count: int,
        queue_length: int,
        waiting_times: List[float],
        coverage: float,
        farthest_queue_px: Optional[float],
        timestamp: float,
        rate_window_s: float = 60.0,
    ) -> LaneState:
        s = self.state
        s.vehicle_count = vehicle_count
        s.queue_length = queue_length
        s.occupancy = float(min(1.0, coverage / self._area))
        s.average_waiting_time = float(np.mean(waiting_times)) if waiting_times else 0.0
        s.max_waiting_time = float(max(waiting_times)) if waiting_times else 0.0
        if self.meters_per_pixel is not None and farthest_queue_px is not None:
            s.queue_length_meters = float(farthest_queue_px * self.meters_per_pixel)
        elif self.meters_per_pixel is not None:
            s.queue_length_meters = 0.0
        else:
            s.queue_length_meters = None
        s.arrival_rate = self.rate_per_min(self._arrivals, timestamp, rate_window_s)
        s.departure_rate = self.rate_per_min(self._departures, timestamp, rate_window_s)
        med, high, crit = self.congestion_thresholds
        s.congestion_level = (
            CongestionLevel.CRITICAL if queue_length >= crit else
            CongestionLevel.HIGH if queue_length >= high else
            CongestionLevel.MEDIUM if queue_length >= med else CongestionLevel.LOW
        )
        s.timestamp = timestamp
        return s

    def get_flow_imbalance(self) -> float:
        """Arrival minus departure rate (veh/min); positive means the queue is building."""
        return self.state.arrival_rate - self.state.departure_rate

    def reset(self) -> None:
        self.state = LaneState()
        self._arrivals.clear()
        self._departures.clear()
        self._crossed.clear()
        self._seen.clear()

    def to_dict(self) -> dict:
        return {
            "lane_id": self.lane_id,
            "direction": self.direction,
            "approach": self.approach,
            "lane_type": self.lane_type,
            "polygon": self.polygon.tolist(),
            "stop_line": [list(p) for p in self.stop_line] if self.stop_line else None,
            "state": self.state.to_dict(),
            "flow_imbalance": self.get_flow_imbalance(),
        }
