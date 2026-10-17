"""Signal policies. A policy maps an observation to a *phase request*; it never drives lights.

* ``FixedTimePolicy``  - cyclic plan with fixed green times (baseline 1).
* ``ActuatedPolicy``   - demand-responsive gap-out / max-out logic (baseline 2). Deliberately
  simple and robust; this is the benchmark the RL policy has to beat.
* ``RLPolicy``         - greedy DQN action from an observation vector.
* ``GuardedPolicy``    - wraps any policy; on exception, invalid output or unhealthy sensors it
  substitutes the conservative fixed-time fallback.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol, Sequence

import numpy as np

from ..safety.intersection import IntersectionModel
from ..safety.signal_safety import PhaseContext
from ..state import IntersectionObservation, PhaseRequest, RequestSource, Stage

logger = logging.getLogger(__name__)


class SignalPolicy(Protocol):
    name: str

    def reset(self) -> None: ...

    def request(self, obs: IntersectionObservation, ctx: PhaseContext) -> PhaseRequest: ...


class FixedTimePolicy:
    """Serve phases cyclically for fixed durations (pedestrian phase included each cycle)."""

    name = "fixed_time"

    def __init__(
        self,
        model: IntersectionModel,
        green_times_s: Optional[Sequence[float]] = None,
        source: RequestSource = RequestSource.FIXED,
    ) -> None:
        self.model = model
        if green_times_s is None:
            green_times_s = [10.0 if p.is_pedestrian else 30.0 for p in model.phases]
        if len(green_times_s) != model.num_phases:
            raise ValueError("one green time per phase is required")
        self.green_times_s = list(green_times_s)
        self.source = source

    def reset(self) -> None:
        pass

    def request(self, obs: IntersectionObservation, ctx: PhaseContext) -> PhaseRequest:
        if ctx.stage is not Stage.HOLD:
            return PhaseRequest(ctx.target_phase, self.source)
        if ctx.phase_elapsed_s >= self.green_times_s[ctx.phase]:
            return PhaseRequest((ctx.phase + 1) % self.model.num_phases, self.source)
        return PhaseRequest(ctx.phase, self.source)


class ActuatedPolicy:
    """Gap-out / max-out actuated control based on stopped-vehicle demand.

    Stay in the current vehicle phase while it still has queued vehicles (the safety
    manager's max green forces a change eventually). When the current phase has been idle
    for ``gap_s`` and another phase has demand, switch to the phase with the highest demand
    score. A pedestrian phase is requested when pedestrians have waited longer than
    ``ped_trigger_s``, or when pedestrians are waiting and no vehicle phase has demand.
    """

    name = "actuated"

    def __init__(
        self,
        model: IntersectionModel,
        gap_s: float = 2.0,
        ped_trigger_s: float = 60.0,
        wait_weight: float = 0.1,
    ) -> None:
        self.model = model
        self.gap_s = gap_s
        self.ped_trigger_s = ped_trigger_s
        self.wait_weight = wait_weight
        self._idle_since: Optional[float] = None

    def reset(self) -> None:
        self._idle_since = None

    def _demand(self, phase: int, obs: IntersectionObservation) -> float:
        ph = self.model.phases[phase]
        score = 0.0
        for a in ph.green_approaches:
            ao = obs.approaches.get(a)
            if ao is not None and ao.queue > 0:
                score += ao.queue + self.wait_weight * ao.max_wait_s
        for c in ph.walk_crosswalks:
            co = obs.crosswalks.get(c)
            if co is not None and co.waiting > 0:
                score += co.waiting * 2.0 + self.wait_weight * co.max_wait_s
        return score

    def request(self, obs: IntersectionObservation, ctx: PhaseContext) -> PhaseRequest:
        src = RequestSource.ACTUATED
        if ctx.stage is not Stage.HOLD:
            return PhaseRequest(ctx.target_phase, src)
        cur = ctx.phase
        cur_demand = self._demand(cur, obs)
        if cur_demand <= 0:
            if self._idle_since is None:
                self._idle_since = obs.time_s
        else:
            self._idle_since = None
        idle = 0.0 if self._idle_since is None else obs.time_s - self._idle_since

        others = [p for p in range(self.model.num_phases) if p != cur]
        demands = {p: self._demand(p, obs) for p in others}
        ped = self.model.pedestrian_phase()
        veh_other = max((d for p, d in demands.items() if p != ped), default=0.0)

        if ped is not None and ped != cur and demands.get(ped, 0.0) > 0:
            worst = max((c.max_wait_s for c in obs.crosswalks.values()), default=0.0)
            if worst >= self.ped_trigger_s or (cur_demand <= 0 and veh_other <= 0):
                return PhaseRequest(ped, src)

        best = max(others, key=lambda p: demands[p], default=cur)
        if demands.get(best, 0.0) > 0:
            gapped_out = cur_demand <= 0 and idle >= self.gap_s
            maxed_out = ctx.phase_elapsed_s >= ctx.max_hold_s
            ped_done = self.model.phases[cur].is_pedestrian and cur_demand <= 0
            if gapped_out or maxed_out or ped_done:
                return PhaseRequest(best, src)
        return PhaseRequest(cur, src)


class RLPolicy:
    """Greedy policy from a trained agent. ``encoder`` turns observations into vectors."""

    name = "rl_dqn"

    def __init__(self, agent, encoder) -> None:
        self.agent = agent
        self.encoder = encoder

    def reset(self) -> None:
        pass

    def request(self, obs: IntersectionObservation, ctx: PhaseContext) -> PhaseRequest:
        vec = self.encoder.encode(obs, ctx)
        action = self.agent.select_action(vec, training=False)
        return PhaseRequest(int(action), RequestSource.RL)


class GuardedPolicy:
    """Substitute a conservative fixed-time cycle when the primary policy cannot be trusted."""

    def __init__(self, primary: SignalPolicy, fallback: SignalPolicy) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = getattr(primary, "name", "policy")
        self.fallback_count = 0

    def reset(self) -> None:
        self.primary.reset()
        self.fallback.reset()

    def request(self, obs: IntersectionObservation, ctx: PhaseContext) -> PhaseRequest:
        if obs is not None and obs.sensors_ok:
            try:
                req = self.primary.request(obs, ctx)
                if req is not None and isinstance(req.phase, (int, np.integer)):
                    return req
                logger.warning("policy %s returned an invalid request %r", self.name, req)
            except Exception:  # noqa: BLE001 - any policy failure must degrade safely
                logger.exception("policy %s raised; using fallback", self.name)
        self.fallback_count += 1
        safe = self.fallback.request(obs, ctx)  # type: ignore[arg-type]
        return PhaseRequest(safe.phase, RequestSource.FALLBACK)
