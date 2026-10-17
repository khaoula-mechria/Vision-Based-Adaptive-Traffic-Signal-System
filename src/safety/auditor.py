"""Independent audit of displayed signal states.

``SignalAuditor`` watches the *lights* (from ``SignalCommand`` or read back from the
simulator) and checks invariants without using any of the safety manager's logic:

* ``conflicting_green``      two conflicting elements are non-red at the same time
* ``missing_clearance``      an element went green -> red without passing yellow / flashing
* ``short_clearance``        yellow / flashing-don't-walk shorter than required
* ``short_all_red``          a conflicting element turned green before ``all_red_s`` elapsed
                             after the previous one turned red
* ``short_green``            green shorter than the minimum (unless flagged as emergency override)

Use it as a runtime monitor, as the safety-violation counter in evaluations, and in tests.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

from .intersection import IntersectionModel
from .signal_safety import SignalTiming

NON_RED = {"G", "y", "W", "F"}


@dataclass(frozen=True)
class Violation:
    kind: str
    time_s: float
    detail: str


class SignalAuditor:
    def __init__(self, model: IntersectionModel, timing: SignalTiming, tolerance_s: float = 1e-6) -> None:
        self.model = model
        self.timing = timing
        self.tol = tolerance_s
        self._prev: Dict[str, str] = {}
        self._since: Dict[str, float] = {}  # time the element entered its current state
        self._red_since: Dict[str, float] = {}
        self._green_start: Dict[str, float] = {}
        self._last_t: Optional[float] = None
        self.violations: List[Violation] = []
        self.counts: Counter = Counter()

    def _flag(self, kind: str, t: float, detail: str) -> None:
        self.violations.append(Violation(kind, t, detail))
        self.counts[kind] += 1

    def observe(
        self,
        time_s: float,
        approach_lights: Mapping[str, str],
        crosswalk_lights: Mapping[str, str],
        emergency_override: bool = False,
    ) -> List[Violation]:
        """Record one snapshot of the lights. Returns violations found at this snapshot."""
        before = len(self.violations)
        lights: Dict[str, str] = {**approach_lights, **crosswalk_lights}
        t = self.timing

        # 1. simultaneous service of conflicting elements
        active = [e for e, s in lights.items() if s in NON_RED]
        for i, a in enumerate(active):
            for b in active[i + 1:]:
                if self.model.conflict(a, b):
                    self._flag("conflicting_green", time_s, f"{a}={lights[a]} conflicts with {b}={lights[b]}")

        for e, s in lights.items():
            prev = self._prev.get(e)
            if prev is None:
                self._since[e] = time_s
                if s in ("G", "W"):
                    self._green_start[e] = time_s
                if s not in NON_RED:
                    self._red_since[e] = time_s
                continue
            if s == prev:
                continue
            # state change for e
            started_green = s in ("G", "W")
            if prev in ("G", "W") and s in ("r", "D"):
                self._flag("missing_clearance", time_s, f"{e}: {prev}->{s} without yellow/flashing")
            if prev in ("G", "W"):
                held = time_s - self._green_start.get(e, time_s)
                lo = t.ped_walk_min_s if e in self.model.crosswalks else t.min_green_s
                if held + self.tol < lo and not emergency_override:
                    self._flag("short_green", time_s, f"{e} green for {held:.1f}s < {lo:.1f}s")
            if prev in ("y", "F") and s in ("r", "D"):
                dur = time_s - self._since[e]
                need = t.ped_clearance_s if e in self.model.crosswalks else t.yellow_s
                if dur + self.tol < need:
                    self._flag("short_clearance", time_s, f"{e} clearance {dur:.1f}s < {need:.1f}s")
                self._red_since[e] = time_s
            if started_green:
                self._green_start[e] = time_s
                for other in lights:
                    if other != e and self.model.conflict(e, other):
                        if lights[other] in NON_RED:
                            continue  # already reported as conflicting_green
                        gap = time_s - self._red_since.get(other, float("-inf"))
                        if gap + self.tol < t.all_red_s:
                            self._flag("short_all_red", time_s, f"{e} green only {gap:.1f}s after {other} red")
            self._since[e] = time_s
        for e, s in lights.items():
            self._prev[e] = s
        self._last_t = time_s
        return self.violations[before:]

    def observe_command(self, cmd) -> List[Violation]:
        return self.observe(cmd.time_s, cmd.approach_lights, cmd.crosswalk_lights, cmd.emergency_override)

    @property
    def total(self) -> int:
        return len(self.violations)
