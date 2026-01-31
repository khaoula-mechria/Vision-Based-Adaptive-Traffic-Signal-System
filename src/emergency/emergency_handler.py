"""
Emergency Vehicle Handler
Detects emergency vehicles and manages signal preemption for green corridors
"""

import time
from typing import Dict, List, Optional, Tuple
import logging

from .emergency_state import (
    EmergencyVehicle,
    EmergencyState,
    EmergencyType,
    PreemptionState,
)
from ..tracking.track import Track


logger = logging.getLogger(__name__)


class EmergencyVehicleHandler:
    """
    Handles emergency vehicle detection and signal preemption.
    
    Features:
    - Detects emergency vehicles using visual cues
    - Tracks approaching emergency vehicles
    - Preempts normal signal cycles for green corridor
    - Smoothly restores traffic flow after priority override
    """
    
    def __init__(
        self,
        detection_confidence: float = 0.7,
        min_detections: int = 3,
        preemption_delay: float = 2.0,
        green_corridor_duration: float = 15.0,
        cooldown_period: float = 10.0,
        meters_per_pixel: float = 0.1,
    ):
        """
        Initialize the emergency vehicle handler.
        
        Args:
            detection_confidence: Minimum confidence for emergency detection
            min_detections: Minimum consecutive detections to confirm
            preemption_delay: Delay before signal change in seconds
            green_corridor_duration: Duration of green corridor in seconds
            cooldown_period: Cooldown period after emergency passes
            meters_per_pixel: Scale for distance estimation
        """
        self.detection_confidence = detection_confidence
        self.min_detections = min_detections
        self.preemption_delay = preemption_delay
        self.green_corridor_duration = green_corridor_duration
        self.cooldown_period = cooldown_period
        self.meters_per_pixel = meters_per_pixel
        
        # Tracked emergency vehicles
        self.vehicles: Dict[int, EmergencyVehicle] = {}
        
        # Current emergency state
        self.state = EmergencyState(
            green_corridor_duration=green_corridor_duration,
            cooldown_duration=cooldown_period,
        )
        
        # Signal phase to direction mapping
        self.direction_phases: Dict[str, str] = {
            "north": "NS_Green",
            "south": "NS_Green",
            "east": "EW_Green",
            "west": "EW_Green",
        }
        
        # Event history
        self._preemption_history: List[dict] = []
    
    def update(
        self,
        tracks: List[Track],
        timestamp: Optional[float] = None,
    ) -> EmergencyState:
        """
        Update emergency detection with tracked vehicles.
        
        Args:
            tracks: List of all tracked objects
            timestamp: Current timestamp
            
        Returns:
            Current emergency state
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Get emergency vehicle tracks
        emergency_tracks = [t for t in tracks if t.is_emergency]
        
        # Update tracked emergency vehicles
        current_ids = set()
        for track in emergency_tracks:
            track_id = track.track_id
            current_ids.add(track_id)
            
            if track_id not in self.vehicles:
                # New emergency vehicle detected
                self.vehicles[track_id] = EmergencyVehicle(
                    track_id=track_id,
                    first_detected=timestamp,
                )
                logger.info(f"New emergency vehicle detected: {track_id}")
            
            # Update vehicle info
            vehicle = self.vehicles[track_id]
            vehicle.update_detection(
                position=track.get_center(),
                velocity=track.get_velocity(),
                confidence=track.confidence,
                timestamp=timestamp,
            )
            
            # Estimate distance and arrival time
            self._estimate_arrival(vehicle)
        
        # Remove stale vehicles
        stale_ids = []
        for track_id, vehicle in self.vehicles.items():
            if track_id not in current_ids:
                if timestamp - vehicle.last_detected > 5.0:  # 5 second timeout
                    stale_ids.append(track_id)
        
        for track_id in stale_ids:
            logger.info(f"Emergency vehicle {track_id} no longer detected")
            del self.vehicles[track_id]
        
        # Update state count
        self.state.vehicles_detected = len(self.vehicles)
        
        # Handle state transitions
        self._update_state(timestamp)
        
        return self.state
    
    def _estimate_arrival(self, vehicle: EmergencyVehicle):
        """Estimate distance and arrival time for emergency vehicle"""
        # Estimate distance based on position (simplified)
        # In production, this would use perspective transformation
        vx, vy = vehicle.velocity
        speed_px = (vx**2 + vy**2)**0.5
        speed_m = speed_px * self.meters_per_pixel
        
        # Rough distance estimate based on position in frame
        # Assume center is intersection
        x, y = vehicle.position
        distance_px = (x**2 + y**2)**0.5  # Distance from center
        vehicle.estimated_distance_meters = distance_px * self.meters_per_pixel
        
        # Arrival time
        if speed_m > 0:
            vehicle.estimated_arrival_seconds = vehicle.estimated_distance_meters / speed_m
        else:
            vehicle.estimated_arrival_seconds = float('inf')
    
    def _update_state(self, timestamp: float):
        """Update preemption state machine"""
        # Check for state transitions first
        next_state = self.state.should_transition(timestamp)
        if next_state:
            self._transition_to_state(next_state, timestamp)
        
        # If not active, check for new emergency
        if not self.state.is_active():
            confirmed_vehicle = self._get_confirmed_approaching_vehicle()
            if confirmed_vehicle:
                self._start_preemption(confirmed_vehicle, timestamp)
    
    def _get_confirmed_approaching_vehicle(self) -> Optional[EmergencyVehicle]:
        """Get the nearest confirmed approaching emergency vehicle"""
        confirmed = [
            v for v in self.vehicles.values()
            if v.is_confirmed(self.min_detections) and v.confidence >= self.detection_confidence
        ]
        
        if not confirmed:
            return None
        
        # Return nearest vehicle
        return min(confirmed, key=lambda v: v.estimated_distance_meters)
    
    def _start_preemption(self, vehicle: EmergencyVehicle, timestamp: float):
        """Start emergency preemption sequence"""
        logger.warning(
            f"EMERGENCY PREEMPTION: Vehicle {vehicle.track_id} approaching from "
            f"{vehicle.approach_direction}, ETA: {vehicle.estimated_arrival_seconds:.1f}s"
        )
        
        self.state.start_preemption(
            vehicle_id=vehicle.track_id,
            direction=vehicle.approach_direction,
            timestamp=timestamp,
        )
        
        # Record event
        self._preemption_history.append({
            "event": "preemption_start",
            "vehicle_id": vehicle.track_id,
            "direction": vehicle.approach_direction,
            "timestamp": timestamp,
        })
    
    def _transition_to_state(self, new_state: PreemptionState, timestamp: float):
        """Transition to a new preemption state"""
        old_state = self.state.preemption_state
        
        if new_state == PreemptionState.ACTIVE:
            self.state.activate_green_corridor(timestamp)
            logger.warning(f"GREEN CORRIDOR ACTIVE: Direction {self.state.active_direction}")
            
            self._preemption_history.append({
                "event": "green_corridor_active",
                "direction": self.state.active_direction,
                "timestamp": timestamp,
            })
        
        elif new_state == PreemptionState.COOLDOWN:
            self.state.start_cooldown(timestamp)
            logger.info("Emergency passed, entering cooldown")
            
            self._preemption_history.append({
                "event": "cooldown_start",
                "timestamp": timestamp,
            })
        
        elif new_state == PreemptionState.INACTIVE:
            self.state.deactivate()
            logger.info("Emergency preemption complete, resuming normal operation")
            
            self._preemption_history.append({
                "event": "preemption_end",
                "timestamp": timestamp,
            })
    
    def get_required_phase(self) -> Optional[str]:
        """
        Get the required signal phase for emergency passage.
        
        Returns:
            Required phase name if preemption active, None otherwise
        """
        if not self.state.is_active():
            return None
        
        direction = self.state.active_direction
        return self.direction_phases.get(direction)
    
    def get_blocked_directions(self) -> List[str]:
        """
        Get directions that should not have green during emergency.
        
        Returns:
            List of blocked direction names
        """
        if not self.state.is_active():
            return []
        
        active_direction = self.state.active_direction
        
        # Block perpendicular directions
        if active_direction in ["north", "south"]:
            return ["east", "west"]
        elif active_direction in ["east", "west"]:
            return ["north", "south"]
        
        return []
    
    def is_preemption_active(self) -> bool:
        """Check if emergency preemption is active"""
        return self.state.is_active()
    
    def get_clearance_time(self) -> float:
        """
        Get time until current preemption ends.
        
        Returns:
            Seconds until preemption ends, 0 if not active
        """
        if not self.state.is_active():
            return 0.0
        
        if self.state.preemption_state == PreemptionState.DETECTING:
            remaining = self.preemption_delay - self.state.time_in_state
        elif self.state.preemption_state == PreemptionState.ACTIVE:
            remaining = self.green_corridor_duration - self.state.time_in_state
        else:
            remaining = 0.0
        
        return max(0.0, remaining)
    
    def force_end_preemption(self, timestamp: Optional[float] = None):
        """
        Force end of emergency preemption.
        
        Use with caution - normally preemption ends automatically.
        """
        if timestamp is None:
            timestamp = time.time()
        
        logger.warning("Emergency preemption force-ended")
        self.state.start_cooldown(timestamp)
    
    def get_active_vehicles(self) -> List[EmergencyVehicle]:
        """Get list of currently tracked emergency vehicles"""
        return list(self.vehicles.values())
    
    def get_preemption_history(
        self,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """Get preemption event history"""
        if limit:
            return self._preemption_history[-limit:]
        return self._preemption_history.copy()
    
    def get_summary(self) -> dict:
        """Get summary of emergency handler state"""
        return {
            "state": self.state.to_dict(),
            "active_vehicles": [v.to_dict() for v in self.vehicles.values()],
            "required_phase": self.get_required_phase(),
            "blocked_directions": self.get_blocked_directions(),
            "clearance_time": self.get_clearance_time(),
        }
    
    def reset(self):
        """Reset emergency handler state"""
        self.vehicles.clear()
        self.state = EmergencyState(
            green_corridor_duration=self.green_corridor_duration,
            cooldown_duration=self.cooldown_period,
        )
        self._preemption_history.clear()
