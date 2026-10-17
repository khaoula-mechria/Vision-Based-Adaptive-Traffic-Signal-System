"""Vision-Based Adaptive Traffic Signal System (research prototype).

Imports are lazy so that ``import src`` does not pull in torch / ultralytics / traci.
Not certified for control of public-road traffic.
"""

__version__ = "2.0.0"

_LAZY = {
    "ObjectDetector": ".perception",
    "MultiObjectTracker": ".tracking",
    "QueueEstimator": ".queue_estimation",
    "PedestrianSafetyManager": ".safety",
    "SignalSafetyManager": ".safety",
    "EmergencyVehicleHandler": ".emergency",
    "TrafficSignalAgent": ".rl_agent",
    "TrafficSignalEnv": ".rl_agent",
    "SumoBackend": ".simulation",
    "AdaptiveTrafficController": ".controller",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
