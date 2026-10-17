"""Data structures exchanged between pipeline stages.

Pipeline:  perception -> tracking -> state estimation -> ``IntersectionObservation``
           -> policy (RL / actuated / fixed) -> ``PhaseRequest``
           -> ``SignalSafetyManager`` -> ``SignalCommand`` -> signal executor / simulator

Policies never produce lights directly; they only *request* a phase. The safety
manager is the only component that produces a ``SignalCommand``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Mapping, Optional, Tuple


class Stage(str, Enum):
    """Stage of the safety manager's signal state machine."""

    HOLD = "hold"  # a phase is green / walking
    CLEARANCE = "clearance"  # yellow (vehicles) or flashing don't-walk (pedestrians)
    ALL_RED = "all_red"  # everything red / don't walk


class RequestSource(str, Enum):
    RL = "rl"
    ACTUATED = "actuated"
    FIXED = "fixed"
    FALLBACK = "fallback"
    MANUAL = "manual"


@dataclass(frozen=True)
class PhaseRequest:
    """A *request* for a phase. Never executed directly."""

    phase: int
    source: RequestSource = RequestSource.FIXED


@dataclass(frozen=True)
class ApproachObservation:
    """Per-approach traffic state (one entry per approach, however many lanes)."""

    queue: float = 0.0  # stopped / halting vehicles
    max_wait_s: float = 0.0
    mean_wait_s: float = 0.0
    emergency: bool = False  # confirmed emergency vehicle on this approach


@dataclass(frozen=True)
class CrosswalkObservation:
    waiting: float = 0.0  # pedestrians waiting to cross
    max_wait_s: float = 0.0
    occupied: float = 0.0  # pedestrians currently inside the crosswalk


@dataclass(frozen=True)
class IntersectionObservation:
    """Everything the decision layer and safety layer know about the intersection.

    ``sensors_ok`` is False when perception is stale/failed; ``ped_uncertain`` is True
    when pedestrian detection cannot be trusted (e.g. night, occlusion, high miss rate).
    Both make the safety manager fall back to conservative behaviour.
    """

    time_s: float
    approaches: Mapping[str, ApproachObservation]
    crosswalks: Mapping[str, CrosswalkObservation]
    sensors_ok: bool = True
    ped_uncertain: bool = False
    box_vehicles: int = 0  # vehicles currently inside the junction box (conflict zone)

    def emergency_approaches(self) -> Tuple[str, ...]:
        return tuple(n for n, a in self.approaches.items() if a.emergency)

    def occupied_crosswalks(self) -> Dict[str, float]:
        return {n: c.occupied for n, c in self.crosswalks.items() if c.occupied > 0}


@dataclass(frozen=True)
class SafetyDecision:
    """Outcome of validating one request. ``accepted`` False => request was overridden."""

    accepted: bool
    requested_phase: Optional[int]
    effective_phase: int  # phase the intersection will be in / heading to
    reason: str = "ok"


@dataclass(frozen=True)
class SignalCommand:
    """What the signal heads must display *now*."""

    time_s: float
    stage: Stage
    phase: int  # current (HOLD) or outgoing (CLEARANCE) phase; target phase in ALL_RED
    target_phase: int
    approach_lights: Mapping[str, str]  # 'G' green, 'y' yellow, 'r' red
    crosswalk_lights: Mapping[str, str]  # 'W' walk, 'F' flashing don't walk, 'D' don't walk
    decision: Optional[SafetyDecision] = None
    emergency_override: bool = False
    phase_elapsed_s: float = 0.0
    events: Tuple[str, ...] = field(default_factory=tuple)
