"""Lightweight stochastic queue model of a signalised intersection.

Intended for unit tests, smoke tests and very fast experiments. It is a *model*, not a
substitute for SUMO: vehicles are queued FIFO per approach and discharged at a saturation
flow rate while the approach displays green (half rate on yellow); pedestrians cross only
during walk. Results from this backend must not be reported as traffic-simulation results.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Mapping, Optional, Union

import numpy as np

from ..safety.intersection import IntersectionModel
from ..state import (
    ApproachObservation,
    CrosswalkObservation,
    IntersectionObservation,
    SignalCommand,
)
from .backend import SensorNoise, StepStats, TrafficBackend


@dataclass
class QueueModelConfig:
    arrival_rate_veh_s: Union[float, Mapping[str, float]] = 0.12  # per approach
    ped_arrival_rate_s: Union[float, Mapping[str, float]] = 0.02  # per crosswalk
    emergency_rate_s: float = 1.0 / 1200.0  # per second, any approach
    saturation_flow_veh_s: float = 0.5
    startup_lost_s: float = 2.0
    crossing_time_s: float = 6.0
    box_clear_s: float = 2.0
    episode_s: float = 1800.0


@dataclass
class _Veh:
    arrival: float
    emergency: bool = False


class QueueModelBackend(TrafficBackend):
    step_length_s = 1.0

    def __init__(
        self,
        model: IntersectionModel,
        config: Optional[QueueModelConfig] = None,
        noise: Optional[SensorNoise] = None,
        seed: int = 0,
    ) -> None:
        super().__init__(model, noise, seed)
        self.cfg = config or QueueModelConfig()
        self._seed = seed
        self.reset(seed)

    @property
    def time_s(self) -> float:
        return self._t

    def _rate(self, spec, name: str) -> float:
        return float(spec[name]) if isinstance(spec, Mapping) else float(spec)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._seed = seed
        self._rng = random.Random(self._seed)
        self._noise_rng = random.Random(self._seed + 7919)
        self._t = 0.0
        m = self.model
        self._q: Dict[str, Deque[_Veh]] = {a: deque() for a in m.approaches}
        self._green_for: Dict[str, float] = {a: 0.0 for a in m.approaches}
        self._credit: Dict[str, float] = {a: 0.0 for a in m.approaches}
        self._ped_wait: Dict[str, Deque[float]] = {c: deque() for c in m.crosswalks}
        self._crossing: Dict[str, List[float]] = {c: [] for c in m.crosswalks}
        self._box_until = -1.0
        self._veh_waits: List[float] = []
        self._ped_waits: List[float] = []
        self._emergency_delays: List[float] = []
        self._queue_sum = 0.0
        self._steps = 0
        self._conflict_steps = 0
        self._discharged = 0
        self._veh_wait_total = 0.0
        self._ped_wait_total = 0.0

    def _poisson(self, lam: float) -> int:
        if lam <= 0:
            return 0
        L, k, p = np.exp(-lam), 0, 1.0
        while True:
            p *= self._rng.random()
            if p <= L:
                return k
            k += 1

    def _truth(self) -> IntersectionObservation:
        t = self._t
        approaches = {}
        for a, q in self._q.items():
            waits = [t - v.arrival for v in q]
            approaches[a] = ApproachObservation(
                queue=float(len(q)),
                max_wait_s=max(waits, default=0.0),
                mean_wait_s=float(np.mean(waits)) if waits else 0.0,
                emergency=any(v.emergency for v in q),
            )
        crosswalks = {}
        for c in self.model.crosswalks:
            waits = [t - w for w in self._ped_wait[c]]
            crosswalks[c] = CrosswalkObservation(
                waiting=float(len(waits)),
                max_wait_s=max(waits, default=0.0),
                occupied=float(sum(1 for e in self._crossing[c] if e > t)),
            )
        return IntersectionObservation(
            time_s=t,
            approaches=approaches,
            crosswalks=crosswalks,
            box_vehicles=1 if t < self._box_until else 0,
        )

    def step(self, cmd: SignalCommand) -> StepStats:
        cfg, dt, m = self.cfg, self.step_length_s, self.model
        stats = StepStats()
        t = self._t

        # arrivals
        for a in m.approaches:
            for _ in range(self._poisson(self._rate(cfg.arrival_rate_veh_s, a) * dt)):
                self._q[a].append(_Veh(t))
        if self._rng.random() < cfg.emergency_rate_s * dt and m.approaches:
            a = self._rng.choice(m.approaches)
            self._q[a].append(_Veh(t, emergency=True))
        for c in m.crosswalks:
            for _ in range(self._poisson(self._rate(cfg.ped_arrival_rate_s, c) * dt)):
                self._ped_wait[c].append(t)

        # true pedestrian occupancy (for conflict ground truth and vehicle yielding)
        occupied = {c for c in m.crosswalks if any(e > t for e in self._crossing[c])}
        for c in m.crosswalks:
            self._crossing[c] = [e for e in self._crossing[c] if e > t]

        # vehicle discharge
        for a in m.approaches:
            light = cmd.approach_lights.get(a, "r")
            blocked = any(m.conflict(a, c) for c in occupied)
            if light in ("G", "y") and not blocked:
                self._green_for[a] += dt
                if self._green_for[a] > cfg.startup_lost_s:
                    self._credit[a] += cfg.saturation_flow_veh_s * dt * (1.0 if light == "G" else 0.5)
                while self._credit[a] >= 1.0 and self._q[a]:
                    v = self._q[a].popleft()
                    self._credit[a] -= 1.0
                    stats.discharged += 1
                    self._veh_waits.append(t - v.arrival)
                    if v.emergency:
                        self._emergency_delays.append(t - v.arrival)
                    self._box_until = t + cfg.box_clear_s
                if not self._q[a]:
                    self._credit[a] = min(self._credit[a], 1.0)
            else:
                self._green_for[a] = 0.0
                self._credit[a] = 0.0

        # pedestrians start crossing only during walk
        for c in m.crosswalks:
            if cmd.crosswalk_lights.get(c) == "W":
                while self._ped_wait[c]:
                    arrived = self._ped_wait[c].popleft()
                    self._ped_waits.append(t - arrived)
                    self._crossing[c].append(t + cfg.crossing_time_s)

        # ground-truth conflict: pedestrian in crosswalk while a conflicting approach is non-red
        occupied_now = {c for c in m.crosswalks if self._crossing[c]}
        for c in occupied_now:
            if any(m.conflict(c, a) and cmd.approach_lights.get(a, "r") in ("G", "y") for a in m.approaches):
                stats.ped_conflict = True
                self._conflict_steps += 1
                break

        # accrual
        waiting = sum(len(q) for q in self._q.values())
        stats.vehicle_wait_s = waiting * dt
        stats.pedestrian_wait_s = sum(len(w) for w in self._ped_wait.values()) * dt
        stats.emergency_wait_s = sum(1 for q in self._q.values() for v in q if v.emergency) * dt
        self._veh_wait_total += stats.vehicle_wait_s
        self._ped_wait_total += stats.pedestrian_wait_s
        self._queue_sum += waiting
        self._discharged += stats.discharged
        self._steps += 1
        self._t += dt
        return stats

    def episode_metrics(self) -> Dict[str, float]:
        t = self._t
        veh_waits = self._veh_waits + [t - v.arrival for q in self._q.values() for v in q]
        ped_waits = self._ped_waits + [t - w for q in self._ped_wait.values() for w in q]
        return {
            "mean_vehicle_wait_s": float(np.mean(veh_waits)) if veh_waits else 0.0,
            "max_vehicle_wait_s": float(max(veh_waits, default=0.0)),
            "mean_pedestrian_wait_s": float(np.mean(ped_waits)) if ped_waits else 0.0,
            "max_pedestrian_wait_s": float(max(ped_waits, default=0.0)),
            "mean_queue": self._queue_sum / max(1, self._steps),
            "throughput_veh": float(self._discharged),
            "mean_emergency_delay_s": float(np.mean(self._emergency_delays)) if self._emergency_delays else 0.0,
            "emergency_vehicles_served": float(len(self._emergency_delays)),
            "ped_conflict_steps": float(self._conflict_steps),
        }

    def close(self) -> None:
        pass
