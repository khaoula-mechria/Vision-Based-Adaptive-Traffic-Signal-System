"""Emergency vehicle confirmation.

A single-frame detection must never trigger preemption (a false positive stops cross traffic),
so an emergency track is *confirmed* only after ``min_detections`` consecutive-ish sightings
with confidence >= ``detection_confidence``. A confirmed approach stays active for
``hold_s`` after the vehicle was last seen (occlusion, missed frames). The handler only reports
*which approaches* have a confirmed emergency vehicle; the ``SignalSafetyManager`` decides
whether and how the signal can serve it without violating clearance / pedestrian constraints.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, FrozenSet, Optional, Sequence

from ..tracking.track import Track

logger = logging.getLogger(__name__)


class EmergencyVehicleHandler:
    def __init__(
        self,
        detection_confidence: float = 0.7,
        min_detections: int = 3,
        hold_s: float = 5.0,
        approach_of: Optional[Callable[[Track], Optional[str]]] = None,
    ) -> None:
        if not 0 <= detection_confidence <= 1 or min_detections < 1 or hold_s < 0:
            raise ValueError("invalid emergency handler parameters")
        self.detection_confidence = detection_confidence
        self.min_detections = min_detections
        self.hold_s = hold_s
        self.approach_of = approach_of
        self._hits: Dict[int, int] = {}
        self._last_seen: Dict[int, float] = {}
        self._approach: Dict[int, str] = {}
        self.history: list = []

    def update(self, tracks: Sequence[Track], timestamp: float) -> FrozenSet[str]:
        """Return approaches with a confirmed emergency vehicle at ``timestamp``."""
        for t in tracks:
            if not t.is_emergency or t.confidence < self.detection_confidence or t.is_coasting:
                continue
            first = t.track_id not in self._hits
            self._hits[t.track_id] = self._hits.get(t.track_id, 0) + 1
            self._last_seen[t.track_id] = timestamp
            ap = self.approach_of(t) if self.approach_of else None
            if ap is not None:
                self._approach[t.track_id] = ap
            if first:
                logger.info("emergency track %d first seen", t.track_id)
        active = set()
        for tid in list(self._last_seen):
            if timestamp - self._last_seen[tid] > self.hold_s:
                for d in (self._hits, self._last_seen, self._approach):
                    d.pop(tid, None)
                self.history.append({"event": "cleared", "track": tid, "t": timestamp})
            elif self._hits[tid] >= self.min_detections and tid in self._approach:
                active.add(self._approach[tid])
        return frozenset(active)

    def reset(self) -> None:
        self._hits.clear()
        self._last_seen.clear()
        self._approach.clear()
        self.history.clear()
