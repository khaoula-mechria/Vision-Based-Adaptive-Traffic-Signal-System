"""Reward for the signal-control RL agent.

Design notes
------------
* Computed from per-interval *deltas* (waiting accrued, vehicles discharged, switches),
  not cumulative counters.
* Fixed scales, no running-mean normalisation: the reward stays stationary across episodes.
* Safety is **not** enforced through this reward. Unsafe actions are filtered by
  ``SignalSafetyManager`` before execution; the ``rejected`` / ``violation`` terms only
  discourage proposing actions the safety layer has to override and make violations visible
  in training curves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class RewardWeights:
    vehicle_wait: float = 1.0
    pedestrian_wait: float = 1.5
    queue: float = 0.5
    throughput: float = 0.5
    fairness: float = 0.5  # penalise the longest wait on any approach (starvation)
    switching: float = 0.2  # per actual phase change in the interval
    rejected: float = 0.1  # action had to be overridden by the safety layer
    violation: float = 10.0  # auditor-detected violation (should never occur)
    emergency: float = 1.0  # emergency vehicle waiting
    clip: float = 5.0


@dataclass
class IntervalStats:
    """Quantities measured over one decision interval."""

    vehicle_wait_s: float  # vehicle-seconds of waiting accrued
    pedestrian_wait_s: float  # person-seconds of waiting accrued
    mean_queue: float  # mean halting vehicles (all approaches) over the interval
    discharged: float  # vehicles that passed the stop line
    max_approach_wait_s: float
    phase_changes: int
    rejected: bool
    violations: int
    emergency_wait_s: float
    duration_s: float


class RewardCalculator:
    # Scales chosen so each term is O(1) for a moderately loaded intersection at 5 s intervals.
    VEH_WAIT_SCALE = 100.0
    PED_WAIT_SCALE = 50.0
    QUEUE_SCALE = 20.0
    THROUGHPUT_SCALE = 5.0
    WAIT_CAP_S = 120.0

    def __init__(self, weights: Optional[RewardWeights] = None) -> None:
        self.w = weights or RewardWeights()

    def breakdown(self, s: IntervalStats) -> Dict[str, float]:
        w = self.w
        k = max(s.duration_s, 1e-6) / 5.0  # scale accrual terms to the interval length
        return {
            "vehicle_wait": -w.vehicle_wait * s.vehicle_wait_s / (self.VEH_WAIT_SCALE * k),
            "pedestrian_wait": -w.pedestrian_wait * s.pedestrian_wait_s / (self.PED_WAIT_SCALE * k),
            "queue": -w.queue * s.mean_queue / self.QUEUE_SCALE,
            "throughput": w.throughput * s.discharged / (self.THROUGHPUT_SCALE * k),
            "fairness": -w.fairness * min(s.max_approach_wait_s, self.WAIT_CAP_S) / self.WAIT_CAP_S,
            "switching": -w.switching * s.phase_changes,
            "rejected": -w.rejected if s.rejected else 0.0,
            "violation": -w.violation * s.violations,
            "emergency": -w.emergency * s.emergency_wait_s / max(s.duration_s, 1.0),
        }

    def __call__(self, s: IntervalStats) -> float:
        total = float(sum(self.breakdown(s).values()))
        return float(np.clip(total, -2 * self.w.clip, self.w.clip))
