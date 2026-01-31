"""
Emergency Vehicle handling module initialization
"""

from .emergency_handler import EmergencyVehicleHandler
from .emergency_state import EmergencyState, EmergencyVehicle

__all__ = ["EmergencyVehicleHandler", "EmergencyState", "EmergencyVehicle"]
