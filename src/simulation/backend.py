"""Common interface for traffic backends (SUMO, the built-in queue model, or a real camera
pipeline) plus a sensor-noise model used to test behaviour under detection uncertainty."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Dict, Optional

from ..safety.intersection import IntersectionModel
from ..state import (
    IntersectionObservation,
    SignalCommand,
)


@dataclass
class StepStats:
    """Ground-truth quantities accrued during one simulation step."""

    vehicle_wait_s: float = 0.0  # vehicle-seconds of waiting
    pedestrian_wait_s: float = 0.0  # person-seconds of waiting
    discharged: int = 0  # vehicles that passed the stop line / left the network
    emergency_wait_s: float = 0.0
    ped_conflict: bool = False  # a pedestrian was truly in a crosswalk while a conflicting
    #                             approach displayed green/yellow (ground truth, not sensed)


@dataclass(frozen=True)
class SensorNoise:
    """Degradation applied to what the controller *sees* (ground truth is unaffected).

    ``*_detect_prob`` thins counts binomially; ``dropout_prob`` is the per-step probability
    that the whole perception system is unavailable (``sensors_ok=False``).
    ``ped_uncertain`` is set automatically when ``ped_detect_prob`` < 0.95.
    """

    vehicle_detect_prob: float = 1.0
    ped_detect_prob: float = 1.0
    emergency_detect_prob: float = 1.0
    dropout_prob: float = 0.0

    @property
    def is_clean(self) -> bool:
        return (self.vehicle_detect_prob == 1.0 and self.ped_detect_prob == 1.0
                and self.emergency_detect_prob == 1.0 and self.dropout_prob == 0.0)


def _thin(count: float, p: float, rng: random.Random) -> float:
    if p >= 1.0:
        return count
    return float(sum(rng.random() < p for _ in range(int(round(count)))))


def apply_sensor_noise(obs: IntersectionObservation, noise: SensorNoise, rng: random.Random) -> IntersectionObservation:
    if noise.is_clean:
        return obs
    if noise.dropout_prob > 0 and rng.random() < noise.dropout_prob:
        return replace(obs, sensors_ok=False)
    approaches = {
        n: replace(
            a,
            queue=_thin(a.queue, noise.vehicle_detect_prob, rng),
            emergency=a.emergency and rng.random() < noise.emergency_detect_prob,
        )
        for n, a in obs.approaches.items()
    }
    crosswalks = {
        n: replace(
            c,
            waiting=_thin(c.waiting, noise.ped_detect_prob, rng),
            occupied=_thin(c.occupied, noise.ped_detect_prob, rng),
        )
        for n, c in obs.crosswalks.items()
    }
    return replace(
        obs,
        approaches=approaches,
        crosswalks=crosswalks,
        ped_uncertain=obs.ped_uncertain or noise.ped_detect_prob < 0.95,
    )


class TrafficBackend(ABC):
    """A traffic source/sink advanced in fixed steps under a ``SignalCommand``."""

    step_length_s: float = 1.0
    model: IntersectionModel

    def __init__(self, model: IntersectionModel, noise: Optional[SensorNoise] = None, seed: int = 0) -> None:
        self.model = model
        self.noise = noise or SensorNoise()
        self._noise_rng = random.Random(seed + 7919)

    @property
    @abstractmethod
    def time_s(self) -> float: ...

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        """Start a new deterministic episode."""

    @abstractmethod
    def _truth(self) -> IntersectionObservation:
        """Ground-truth observation."""

    def observe(self) -> IntersectionObservation:
        return apply_sensor_noise(self._truth(), self.noise, self._noise_rng)

    @abstractmethod
    def step(self, cmd: SignalCommand) -> StepStats:
        """Advance one ``step_length_s`` with the given lights."""

    @abstractmethod
    def episode_metrics(self) -> Dict[str, float]:
        """Metrics accumulated since the last reset (measured, not estimated)."""

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> "TrafficBackend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
