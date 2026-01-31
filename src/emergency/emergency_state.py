"""
Emergency vehicle state tracking
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum
import time


class EmergencyType(Enum):
    """Type of emergency vehicle"""
    AMBULANCE = "ambulance"
    FIRE_TRUCK = "fire_truck"
    POLICE = "police"
    UNKNOWN = "unknown"


class PreemptionState(Enum):
    """State of signal preemption"""
    INACTIVE = "inactive"
    DETECTING = "detecting"  # Confirmed detection, preparing
    ACTIVE = "active"  # Green corridor active
    COOLDOWN = "cooldown"  # Restoring normal operation


@dataclass
class EmergencyVehicle:
    """Tracked emergency vehicle"""
    track_id: int
    vehicle_type: EmergencyType = EmergencyType.UNKNOWN
    
    # Position and motion
    position: Tuple[float, float] = (0.0, 0.0)
    velocity: Tuple[float, float] = (0.0, 0.0)
    approach_direction: str = "unknown"  # 'north', 'south', 'east', 'west'
    
    # Detection info
    confidence: float = 0.0
    has_flashing_lights: bool = False
    detection_count: int = 0  # Consecutive detections
    
    # Timing
    first_detected: float = field(default_factory=time.time)
    last_detected: float = field(default_factory=time.time)
    
    # Distance estimation
    estimated_distance_meters: float = 0.0
    estimated_arrival_seconds: float = 0.0
    
    def update_detection(
        self,
        position: Tuple[float, float],
        velocity: Tuple[float, float],
        confidence: float,
        timestamp: float,
    ):
        """Update detection info"""
        self.position = position
        self.velocity = velocity
        self.confidence = confidence
        self.last_detected = timestamp
        self.detection_count += 1
        
        # Estimate approach direction from velocity
        vx, vy = velocity
        if abs(vx) > abs(vy):
            self.approach_direction = "west" if vx > 0 else "east"
        else:
            self.approach_direction = "north" if vy > 0 else "south"
    
    def is_confirmed(self, min_detections: int = 3) -> bool:
        """Check if emergency vehicle is confirmed (consistent detections)"""
        return self.detection_count >= min_detections
    
    def get_tracking_duration(self) -> float:
        """Get how long this vehicle has been tracked"""
        return self.last_detected - self.first_detected
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "track_id": self.track_id,
            "vehicle_type": self.vehicle_type.value,
            "position": self.position,
            "velocity": self.velocity,
            "approach_direction": self.approach_direction,
            "confidence": self.confidence,
            "has_flashing_lights": self.has_flashing_lights,
            "detection_count": self.detection_count,
            "is_confirmed": self.is_confirmed(),
            "estimated_distance_meters": self.estimated_distance_meters,
            "estimated_arrival_seconds": self.estimated_arrival_seconds,
        }


@dataclass
class EmergencyState:
    """Current emergency state for the intersection"""
    preemption_state: PreemptionState = PreemptionState.INACTIVE
    
    # Active emergency vehicle info
    active_vehicle_id: Optional[int] = None
    active_direction: str = "unknown"
    
    # Timing
    preemption_start_time: float = 0.0
    green_corridor_duration: float = 15.0
    cooldown_duration: float = 10.0
    
    # State tracking
    time_in_state: float = 0.0
    vehicles_detected: int = 0
    
    def start_preemption(
        self,
        vehicle_id: int,
        direction: str,
        timestamp: float,
    ):
        """Start emergency preemption"""
        self.preemption_state = PreemptionState.DETECTING
        self.active_vehicle_id = vehicle_id
        self.active_direction = direction
        self.preemption_start_time = timestamp
        self.time_in_state = 0.0
    
    def activate_green_corridor(self, timestamp: float):
        """Activate green corridor for emergency vehicle"""
        self.preemption_state = PreemptionState.ACTIVE
        self.preemption_start_time = timestamp
        self.time_in_state = 0.0
    
    def start_cooldown(self, timestamp: float):
        """Start cooldown period after emergency passes"""
        self.preemption_state = PreemptionState.COOLDOWN
        self.preemption_start_time = timestamp
        self.time_in_state = 0.0
    
    def deactivate(self):
        """Deactivate emergency mode"""
        self.preemption_state = PreemptionState.INACTIVE
        self.active_vehicle_id = None
        self.active_direction = "unknown"
        self.time_in_state = 0.0
    
    def update_time(self, current_time: float):
        """Update time in current state"""
        if self.preemption_start_time > 0:
            self.time_in_state = current_time - self.preemption_start_time
    
    def is_active(self) -> bool:
        """Check if emergency mode is active"""
        return self.preemption_state in [PreemptionState.DETECTING, PreemptionState.ACTIVE]
    
    def should_transition(self, current_time: float) -> Optional[PreemptionState]:
        """
        Check if state should transition.
        
        Returns:
            Next state if transition needed, None otherwise
        """
        self.update_time(current_time)
        
        if self.preemption_state == PreemptionState.DETECTING:
            # Short delay before green corridor
            if self.time_in_state > 2.0:  # 2 second detection delay
                return PreemptionState.ACTIVE
        
        elif self.preemption_state == PreemptionState.ACTIVE:
            # Check if green corridor duration exceeded
            if self.time_in_state > self.green_corridor_duration:
                return PreemptionState.COOLDOWN
        
        elif self.preemption_state == PreemptionState.COOLDOWN:
            # Check if cooldown complete
            if self.time_in_state > self.cooldown_duration:
                return PreemptionState.INACTIVE
        
        return None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "preemption_state": self.preemption_state.value,
            "active_vehicle_id": self.active_vehicle_id,
            "active_direction": self.active_direction,
            "time_in_state": self.time_in_state,
            "is_active": self.is_active(),
            "vehicles_detected": self.vehicles_detected,
        }
