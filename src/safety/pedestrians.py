"""Pedestrian observation from tracks: who is in / waiting at each crosswalk.

This is a *sensing* component. It never decides anything; the ``SignalSafetyManager`` consumes
its output and is designed so that safety does not depend on it being complete (an exclusive
pedestrian phase, full flashing-clearance and all-red intervals protect pedestrians that the
camera missed). Uncertainty is reported via ``uncertain`` so the manager adds margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from ..state import CrosswalkObservation
from ..tracking.track import Track
from ..utils.geometry import as_polygon, point_in_polygon, polygon_centroid


@dataclass
class CrosswalkZone:
    name: str
    polygon: np.ndarray
    waiting_margin_px: float = 60.0  # pedestrians within this distance of the zone count as waiting


class CrosswalkMonitor:
    def __init__(
        self,
        zones: Mapping[str, Sequence[Sequence[float]]],
        stopped_speed: float = 0.15,
        low_confidence: float = 0.6,
        waiting_margin_px: float = 60.0,
    ) -> None:
        self.zones = {n: CrosswalkZone(n, as_polygon(p), waiting_margin_px) for n, p in zones.items()}
        self.stopped_speed = stopped_speed
        self.low_confidence = low_confidence
        self._wait_start: Dict[int, float] = {}

    @classmethod
    def from_config(cls, items: Sequence[Mapping]) -> "CrosswalkMonitor":
        return cls({str(i["id"]): i["polygon"] for i in items})

    def update(self, tracks: Sequence[Track], timestamp: float) -> Tuple[Dict[str, CrosswalkObservation], bool]:
        """Return per-crosswalk observations and whether pedestrian sensing looks unreliable."""
        occupied = {n: 0 for n in self.zones}
        waiting = {n: 0 for n in self.zones}
        max_wait = {n: 0.0 for n in self.zones}
        uncertain = False
        live = set()
        for t in tracks:
            if not (t.is_pedestrian or t.is_cyclist):
                continue
            cx, cy = t.get_center()
            for name, z in self.zones.items():
                inside = point_in_polygon(cx, cy, z.polygon)
                gx, gy = polygon_centroid(z.polygon)
                near = inside or np.hypot(cx - gx, cy - gy) < z.waiting_margin_px + np.sqrt(
                    max(1.0, float(np.ptp(z.polygon[:, 0]) * np.ptp(z.polygon[:, 1]))))
                if not near:
                    continue
                if t.is_coasting or t.confidence < self.low_confidence:
                    uncertain = True  # a pedestrian near a crosswalk that we only half see
                if inside:
                    occupied[name] += 1
                    continue
                x1, y1, x2, y2 = (float(v) for v in t.bbox)
                slow = t.get_speed() / max(x2 - x1, y2 - y1, 1.0) < self.stopped_speed
                if slow:
                    live.add(t.track_id)
                    start = self._wait_start.setdefault(t.track_id, timestamp)
                    waiting[name] += 1
                    max_wait[name] = max(max_wait[name], timestamp - start)
        for tid in [k for k in self._wait_start if k not in live]:
            del self._wait_start[tid]
        obs = {n: CrosswalkObservation(float(waiting[n]), max_wait[n], float(occupied[n])) for n in self.zones}
        return obs, uncertain

    def reset(self) -> None:
        self._wait_start.clear()
