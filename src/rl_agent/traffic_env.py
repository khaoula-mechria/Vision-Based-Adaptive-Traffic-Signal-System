"""Gymnasium-style environment for signal control.

Observation space
    ``Box(0, 1, (ObservationEncoder.dim,))`` - see ``observation.py``.
Action space
    ``Discrete(num_phases)`` - *request* for a phase index of the ``IntersectionModel``
    (default 0=NS, 1=EW, 2=PED). The action is never executed directly: the
    ``SignalSafetyManager`` accepts, delays or overrides it and performs the yellow / all-red
    transitions itself. Requesting the current phase means "extend".
Step
    One step = ``decision_interval_s`` of simulated time (default 5 s).
Termination
    Episodes never *terminate*; they are *truncated* at ``episode_s`` (time limit). Agents must
    bootstrap through truncation.
Determinism
    ``reset(seed=...)`` fully determines demand in both backends.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..safety.auditor import SignalAuditor
from ..safety.signal_safety import PhaseContext, SignalSafetyManager, SignalTiming
from ..simulation.backend import TrafficBackend
from ..state import IntersectionObservation, PhaseRequest, RequestSource
from .observation import ObservationEncoder
from .reward import IntervalStats, RewardCalculator, RewardWeights

try:  # gymnasium is optional; the env works without it
    import gymnasium as gym
    from gymnasium import spaces

    _Base = gym.Env
except ImportError:  # pragma: no cover - exercised only without gymnasium
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]
    _Base = object  # type: ignore[misc,assignment]


class TrafficSignalEnv(_Base):
    metadata = {"render_modes": []}

    def __init__(
        self,
        backend: TrafficBackend,
        timing: Optional[SignalTiming] = None,
        reward_weights: Optional[RewardWeights] = None,
        decision_interval_s: float = 5.0,
        episode_s: float = 1800.0,
        seed: int = 0,
        encoder: Optional[ObservationEncoder] = None,
    ) -> None:
        if decision_interval_s < backend.step_length_s:
            raise ValueError("decision_interval_s must be >= the backend step length")
        self.backend = backend
        self.model = backend.model
        self.timing = timing or SignalTiming()
        self.decision_interval_s = decision_interval_s
        self.episode_s = episode_s
        self.encoder = encoder or ObservationEncoder(self.model)
        self.reward_fn = RewardCalculator(reward_weights)
        self._seed = seed
        self.manager: SignalSafetyManager = SignalSafetyManager(self.model, self.timing)
        self.auditor = SignalAuditor(self.model, self.timing)
        self.intersection_obs: Optional[IntersectionObservation] = None
        self._done = True
        self._ped_conflict_steps = 0
        self._rejections = 0
        self._decisions = 0
        if spaces is not None:
            self.observation_space = spaces.Box(0.0, 1.0, (self.encoder.dim,), np.float32)
            self.action_space = spaces.Discrete(self.model.num_phases)
        self.observation_dim = self.encoder.dim
        self.action_dim = self.model.num_phases

    # ------------------------------------------------------------------ API
    def context(self) -> PhaseContext:
        return self.manager.context(self.backend.time_s)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict]:
        if _Base is not object:
            super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self.backend.reset(self._seed)
        self.manager = SignalSafetyManager(self.model, self.timing, start_time_s=self.backend.time_s)
        self.auditor = SignalAuditor(self.model, self.timing)
        self._ped_conflict_steps = 0
        self._rejections = 0
        self._decisions = 0
        self._done = False
        self.intersection_obs = self.backend.observe()
        return self._encode(), {"time_s": self.backend.time_s}

    def _encode(self) -> np.ndarray:
        return self.encoder.encode(self.intersection_obs, self.context())  # type: ignore[arg-type]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset()")
        req = PhaseRequest(int(action), RequestSource.RL)
        n_ticks = max(1, int(round(self.decision_interval_s / self.backend.step_length_s)))
        changes0 = self.manager.stats["phase_changes"]
        viol0 = self.auditor.total

        veh_wait = ped_wait = emerg_wait = 0.0
        discharged = 0
        queue_acc = 0.0
        rejected = False
        ticks = 0
        reasons = []
        for _ in range(n_ticks):
            cmd = self.manager.step(self.backend.time_s, req, self.intersection_obs)
            self.auditor.observe_command(cmd)
            if cmd.decision is not None:
                self._decisions += 1
                if not cmd.decision.accepted:
                    rejected = True
                    self._rejections += 1
                    reasons.append(cmd.decision.reason)
            stats = self.backend.step(cmd)
            self.intersection_obs = self.backend.observe()
            veh_wait += stats.vehicle_wait_s
            ped_wait += stats.pedestrian_wait_s
            emerg_wait += stats.emergency_wait_s
            discharged += stats.discharged
            self._ped_conflict_steps += int(stats.ped_conflict)
            queue_acc += sum(a.queue for a in self.intersection_obs.approaches.values())
            ticks += 1
            if self.backend.time_s >= self.episode_s:
                break

        obs = self.intersection_obs
        assert obs is not None
        interval = IntervalStats(
            vehicle_wait_s=veh_wait,
            pedestrian_wait_s=ped_wait,
            mean_queue=queue_acc / max(1, ticks),
            discharged=discharged,
            max_approach_wait_s=max((a.max_wait_s for a in obs.approaches.values()), default=0.0),
            phase_changes=self.manager.stats["phase_changes"] - changes0,
            rejected=rejected,
            violations=self.auditor.total - viol0,
            emergency_wait_s=emerg_wait,
            duration_s=ticks * self.backend.step_length_s,
        )
        reward = self.reward_fn(interval)
        truncated = self.backend.time_s >= self.episode_s
        self._done = truncated
        info: Dict[str, Any] = {
            "time_s": self.backend.time_s,
            "request_rejected": rejected,
            "rejection_reasons": reasons,
            "phase": self.manager.phase,
            "stage": self.manager.stage.value,
            "reward_breakdown": self.reward_fn.breakdown(interval),
            "safety_violations": self.auditor.total,
        }
        if truncated:
            info["episode"] = self.summary()
        return self._encode(), reward, False, truncated, info

    def summary(self) -> Dict[str, float]:
        """Episode metrics: backend measurements plus control/safety counters."""
        m = dict(self.backend.episode_metrics())
        m["phase_switches"] = float(self.manager.stats["phase_changes"])
        m["safety_violations"] = float(self.auditor.total)
        m["ped_conflict_steps"] = float(self._ped_conflict_steps)
        m["requests_overridden"] = float(self._rejections)
        m["decisions"] = float(self._decisions)
        m["conflict_hold_timeouts"] = float(self.manager.stats["conflict_hold_timeouts"])
        return m

    def close(self) -> None:
        self.backend.close()
