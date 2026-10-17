"""Independent signal safety manager.

This module is the *only* place that turns decisions into signal lights. Policies (RL,
actuated, fixed-time) can only submit a ``PhaseRequest``; the manager validates it against
hard constraints and drives the green -> clearance -> all-red -> green state machine itself.
It has no dependency on the RL code, so an invalid RL action is rejected no matter what the
network outputs.

Hard constraints enforced here
------------------------------
* Phases come from an ``IntersectionModel`` whose constructor rejects conflicting greens.
* Every phase change passes through clearance (yellow / flashing don't-walk) and an
  all-red interval; a green is never replaced directly by another green.
* Minimum green / minimum walk and maximum green / maximum walk (max forces rotation so no
  movement can be starved by a policy that keeps requesting the same phase).
* A pedestrian walk is never cut short by vehicle demand: it runs at least ``ped_walk_min_s``,
  is extended (up to ``ped_walk_max_s``) while pedestrians are still detected in a crosswalk,
  and always ends with a full flashing-don't-walk clearance.
* A conflicting phase is not released while a crosswalk it conflicts with is detected
  occupied, or while vehicles are in the junction box (bounded hold; on timeout the
  pedestrians are *served* rather than the conflicting vehicles released).
* Emergency preemption goes through the same state machine: it may skip the remaining
  minimum green of a vehicle phase but never skips clearance, all-red, or the minimum walk.
* Degraded sensing (``sensors_ok`` False) -> the manager ignores policy requests and runs a
  fixed cycle that serves every phase, including pedestrians. Uncertain pedestrian detection
  adds extra all-red time. Visual detection is *never* assumed to be complete.

This is a research prototype and is not certified for use on public roads.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..errors import ConfigError
from ..state import (
    IntersectionObservation,
    PhaseRequest,
    SafetyDecision,
    SignalCommand,
    Stage,
)
from .intersection import IntersectionModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignalTiming:
    """Timing constraints in seconds. All values are validated."""

    yellow_s: float = 4.0
    all_red_s: float = 2.0
    min_green_s: float = 8.0
    max_green_s: float = 45.0
    ped_walk_min_s: float = 7.0
    ped_walk_max_s: float = 25.0
    ped_clearance_s: float = 8.0  # flashing don't-walk; should be >= crossing width / walk speed
    max_conflict_hold_s: float = 20.0
    max_pedestrian_wait_s: float = 90.0
    uncertain_extra_all_red_s: float = 2.0
    emergency_min_green_s: float = 4.0
    emergency_extension_s: float = 15.0
    fallback_green_s: float = 20.0
    startup_all_red_s: float = 6.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, (int, float)) or value < 0:
                raise ConfigError(f"signal timing {name} must be a non-negative number, got {value!r}")
        for name in ("yellow_s", "all_red_s", "min_green_s", "ped_walk_min_s", "ped_clearance_s"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"signal timing {name} must be > 0 (safety interval)")
        if self.min_green_s > self.max_green_s:
            raise ConfigError("min_green_s must be <= max_green_s")
        if self.ped_walk_min_s > self.ped_walk_max_s:
            raise ConfigError("ped_walk_min_s must be <= ped_walk_max_s")
        if self.emergency_min_green_s > self.max_green_s:
            raise ConfigError("emergency_min_green_s must be <= max_green_s")

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "SignalTiming":
        cfg = dict(cfg or {})
        unknown = set(cfg) - set(cls.__dataclass_fields__)
        if unknown:
            raise ConfigError(f"unknown signal timing keys: {sorted(unknown)}")
        return cls(**cfg)


@dataclass(frozen=True)
class PhaseContext:
    """Read-only view of the manager state handed to policies."""

    phase: int
    target_phase: int
    stage: Stage
    phase_elapsed_s: float
    min_hold_s: float
    max_hold_s: float

    @property
    def can_switch(self) -> bool:
        return self.stage is Stage.HOLD and self.phase_elapsed_s >= self.min_hold_s


class SignalSafetyManager:
    """Validates phase requests and executes safe signal transitions."""

    def __init__(
        self,
        model: IntersectionModel,
        timing: Optional[SignalTiming] = None,
        initial_phase: int = 0,
        start_time_s: float = 0.0,
    ) -> None:
        self.model = model
        self.timing = timing or SignalTiming()
        if not 0 <= initial_phase < model.num_phases:
            raise ConfigError(f"initial_phase {initial_phase} out of range")
        self._phase = initial_phase
        self._target = initial_phase
        self._stage = Stage.ALL_RED  # safe start: unknown previous state -> full all-red first
        self._stage_start = start_time_s
        self._phase_start = start_time_s
        self._last_time = start_time_s
        self._hold_start: Optional[float] = None  # start of an extended all-red hold
        self._startup = True
        self._emergency_transition = False
        self.stats: Counter = Counter()

    # ------------------------------------------------------------------ queries
    @property
    def stage(self) -> Stage:
        return self._stage

    @property
    def phase(self) -> int:
        return self._phase

    def _hold_limits(self, phase: int) -> Tuple[float, float]:
        t = self.timing
        if self.model.phases[phase].is_pedestrian:
            return t.ped_walk_min_s, t.ped_walk_max_s
        return t.min_green_s, t.max_green_s

    def context(self, now: float) -> PhaseContext:
        lo, hi = self._hold_limits(self._phase)
        return PhaseContext(
            phase=self._phase,
            target_phase=self._target,
            stage=self._stage,
            phase_elapsed_s=max(0.0, now - self._phase_start) if self._stage is Stage.HOLD else 0.0,
            min_hold_s=lo,
            max_hold_s=hi,
        )

    def _next_cyclic(self, phase: int) -> int:
        return (phase + 1) % self.model.num_phases

    # ------------------------------------------------------------- validation
    def evaluate(
        self,
        now: float,
        request: Optional[PhaseRequest],
        obs: Optional[IntersectionObservation],
    ) -> Tuple[SafetyDecision, int, bool]:
        """Decide what to do while in HOLD. Pure: does not change manager state.

        Returns ``(decision, desired_phase, emergency_override)``.
        """
        model, t = self.model, self.timing
        phase = self._phase
        elapsed = now - self._phase_start
        lo, hi = self._hold_limits(phase)
        requested: Optional[int] = None
        if request is not None and isinstance(request.phase, (int, np.integer)):
            requested = int(request.phase)
        valid_request = requested is not None and 0 <= requested < model.num_phases

        def decide(desired: int, reason: str, emergency: bool = False):
            accepted = valid_request and requested == desired
            return SafetyDecision(accepted, requested if valid_request else None, desired, reason), desired, emergency

        # -- degraded sensing: ignore policy, run a fixed cycle that serves everything
        if obs is None or not obs.sensors_ok:
            if elapsed >= (lo if model.phases[phase].is_pedestrian else max(lo, t.fallback_green_s)):
                return decide(self._next_cyclic(phase), "sensors_degraded_fallback_cycle")
            return decide(phase, "sensors_degraded_hold")

        # -- emergency preemption (through the same state machine)
        for approach in obs.emergency_approaches():
            ep = model.phase_for_approach(approach)
            if ep is None:
                continue
            if ep == phase:
                # keep the corridor open a bit beyond the normal maximum, then rotate
                if elapsed >= hi + t.emergency_extension_s:
                    break
                return decide(phase, "emergency_hold", True)
            floor = lo if model.phases[phase].is_pedestrian else t.emergency_min_green_s
            if elapsed >= floor:
                return decide(ep, "emergency_preempt", True)
            return decide(phase, "emergency_waiting_for_minimum", True)

        # -- liveness bounds (not safety): pedestrian wait cap and maximum green
        ped_phase = model.pedestrian_phase()
        if ped_phase is not None and phase != ped_phase and elapsed >= lo:
            worst = max((c.max_wait_s for c in obs.crosswalks.values()), default=0.0)
            if worst >= t.max_pedestrian_wait_s:
                return decide(ped_phase, "pedestrian_wait_cap")

        if elapsed >= hi:
            desired = requested if valid_request and requested is not None and requested != phase else self._next_cyclic(phase)
            return decide(desired, "max_phase_duration")

        # -- ordinary request validation
        if not valid_request:
            return decide(phase, "invalid_request")
        assert requested is not None
        if requested == phase:
            return decide(phase, "ok")
        if elapsed < lo:
            return decide(phase, "min_phase_duration")
        if model.phases[phase].is_pedestrian and obs.occupied_crosswalks():
            return decide(phase, "pedestrians_in_crosswalk")
        return decide(requested, "ok")

    # --------------------------------------------------------------- execution
    def step(
        self,
        now: float,
        request: Optional[PhaseRequest],
        obs: Optional[IntersectionObservation],
    ) -> SignalCommand:
        """Advance the state machine to time ``now`` and return the lights to display."""
        if now < self._last_time:
            logger.warning("non-monotonic time %.3f < %.3f; clamping", now, self._last_time)
            self.stats["non_monotonic_time"] += 1
            now = self._last_time
        self._last_time = now
        events = []
        decision: Optional[SafetyDecision] = None
        t = self.timing
        elapsed_stage = now - self._stage_start

        if self._stage is Stage.HOLD:
            decision, desired, emergency = self.evaluate(now, request, obs)
            if request is not None and not decision.accepted:
                self.stats[f"rejected:{decision.reason}"] += 1
            if desired != self._phase:
                self._begin_clearance(now, desired, emergency)
                events.append(f"phase_end:{self.model.phases[self._phase].name}")
                if emergency:
                    events.append("emergency_preempt")
        elif self._stage is Stage.CLEARANCE:
            dur = t.ped_clearance_s if self.model.phases[self._phase].is_pedestrian else t.yellow_s
            if elapsed_stage >= dur:
                self._stage, self._stage_start, self._hold_start = Stage.ALL_RED, now, None
        else:  # ALL_RED
            required = t.startup_all_red_s if self._startup else t.all_red_s
            if obs is None or not obs.sensors_ok or obs.ped_uncertain:
                required += t.uncertain_extra_all_red_s
            if elapsed_stage >= required:
                events.extend(self._try_release(now, obs))

        return self._command(now, decision, events)

    def _begin_clearance(self, now: float, target: int, emergency: bool) -> None:
        self._target = target
        self._stage = Stage.CLEARANCE
        self._stage_start = now
        self._emergency_transition = emergency
        self.stats["phase_changes"] += 1

    def _try_release(self, now: float, obs: Optional[IntersectionObservation]) -> list:
        model, t = self.model, self.timing
        events: list = []
        target = self._target
        sensing = obs is not None and obs.sensors_ok

        if sensing and not self._startup:  # a new emergency may retarget while everything is red
            for approach in obs.emergency_approaches():  # type: ignore[union-attr]
                ep = model.phase_for_approach(approach)
                if ep is not None and ep != target:
                    target = ep
                    self._emergency_transition = True
                    events.append("emergency_retarget")
                    break

        blocked_cw: list = []
        box = False
        if sensing:
            occupied = obs.occupied_crosswalks()  # type: ignore[union-attr]
            blocked_cw = [cw for cw in model.crosswalks_conflicting_with_phase(target) if cw in occupied]
            box = obs.box_vehicles > 0  # type: ignore[union-attr]

        if blocked_cw or box:
            if self._hold_start is None:
                self._hold_start = now
            names = blocked_cw + (["junction_box"] if box else [])
            if now - self._hold_start < t.max_conflict_hold_s:
                self.stats["conflict_hold_steps"] += 1
                self._target = target
                return events + [f"conflict_hold:{','.join(names)}"]
            self.stats["conflict_hold_timeouts"] += 1
            if blocked_cw:
                # Pedestrians still detected: serve them rather than release conflicting vehicles.
                ped_phase = model.pedestrian_phase()
                if ped_phase is None or target == ped_phase:
                    return events + ["conflict_hold_timeout_remain_red"]
                target = ped_phase
                events.append("conflict_hold_timeout_serve_pedestrians")
            else:
                events.append("junction_box_hold_timeout")  # a stuck vehicle must not deadlock the intersection

        self._phase, self._target = target, target
        self._stage, self._stage_start, self._phase_start = Stage.HOLD, now, now
        self._hold_start = None
        self._startup = False
        events.append(f"phase_start:{model.phases[target].name}")
        return events

    # ------------------------------------------------------------------ lights
    def lights_for(self, stage: Stage, phase: int) -> Tuple[Dict[str, str], Dict[str, str]]:
        ph = self.model.phases[phase]
        approach = {a: "r" for a in self.model.approaches}
        cross = {c: "D" for c in self.model.crosswalks}
        if stage is Stage.HOLD:
            for a in ph.green_approaches:
                approach[a] = "G"
            for c in ph.walk_crosswalks:
                cross[c] = "W"
        elif stage is Stage.CLEARANCE:
            for a in ph.green_approaches:
                approach[a] = "y"
            for c in ph.walk_crosswalks:
                cross[c] = "F"
        return approach, cross

    def _command(self, now: float, decision: Optional[SafetyDecision], events: list) -> SignalCommand:
        approach, cross = self.lights_for(self._stage, self._phase)
        return SignalCommand(
            time_s=now,
            stage=self._stage,
            phase=self._phase,
            target_phase=self._target,
            approach_lights=approach,
            crosswalk_lights=cross,
            decision=decision,
            emergency_override=self._emergency_transition,
            phase_elapsed_s=(now - self._phase_start) if self._stage is Stage.HOLD else 0.0,
            events=tuple(events),
        )
