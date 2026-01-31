"""
Vision-Based Adaptive Traffic Signal System
Main package initialization
"""

__version__ = "1.0.0"
__author__ = "Traffic AI Team"

from .perception import ObjectDetector
from .tracking import MultiObjectTracker
from .queue_estimation import QueueEstimator
from .safety import PedestrianSafetyManager
from .emergency import EmergencyVehicleHandler
from .rl_agent import TrafficSignalAgent
from .simulation import SUMOInterface

__all__ = [
    "ObjectDetector",
    "MultiObjectTracker", 
    "QueueEstimator",
    "PedestrianSafetyManager",
    "EmergencyVehicleHandler",
    "TrafficSignalAgent",
    "SUMOInterface",
]
