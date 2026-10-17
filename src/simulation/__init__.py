"""Traffic backends: SUMO (real simulator) and a lightweight queue model."""

from .backend import SensorNoise, StepStats, TrafficBackend
from .queue_model import QueueModelBackend, QueueModelConfig
from .sumo_scenario import SCENARIOS, Scenario, sumo_available

__all__ = ["TrafficBackend", "SensorNoise", "StepStats", "QueueModelBackend", "QueueModelConfig",
           "Scenario", "SCENARIOS", "sumo_available", "SumoBackend"]


def __getattr__(name):  # traci is imported only when the SUMO backend is used
    if name == "SumoBackend":
        from .sumo_backend import SumoBackend

        return SumoBackend
    raise AttributeError(name)
