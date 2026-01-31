"""
Lane data structure and state tracking
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set
from enum import Enum
import time
import numpy as np


class CongestionLevel(Enum):
    """Congestion level classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LaneState:
    """Current state of a lane"""
    queue_length: int = 0  # Number of vehicles in queue
    queue_length_meters: float = 0.0  # Physical queue length
    occupancy: float = 0.0  # Percentage of lane occupied (0-1)
    average_waiting_time: float = 0.0  # Average waiting time in seconds
    max_waiting_time: float = 0.0  # Maximum waiting time in seconds
    arrival_rate: float = 0.0  # Vehicles per minute
    departure_rate: float = 0.0  # Vehicles per minute
    congestion_level: CongestionLevel = CongestionLevel.LOW
    
    # Time tracking
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "queue_length": self.queue_length,
            "queue_length_meters": self.queue_length_meters,
            "occupancy": self.occupancy,
            "average_waiting_time": self.average_waiting_time,
            "max_waiting_time": self.max_waiting_time,
            "arrival_rate": self.arrival_rate,
            "departure_rate": self.departure_rate,
            "congestion_level": self.congestion_level.value,
            "timestamp": self.timestamp,
        }


@dataclass
class Lane:
    """
    Represents a traffic lane with queue and flow estimation.
    """
    lane_id: int
    direction: str  # 'north', 'south', 'east', 'west'
    lane_type: str = "through"  # 'through', 'left_turn', 'right_turn'
    
    # Lane geometry (in pixels)
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    
    # Lane parameters
    length_meters: float = 100.0
    width_meters: float = 3.5
    vehicle_spacing_meters: float = 7.0  # Average space per vehicle
    
    # Current state
    state: LaneState = field(default_factory=LaneState)
    
    # Historical data for rate calculation
    _vehicle_ids: Set[int] = field(default_factory=set)
    _arrival_times: List[float] = field(default_factory=list)
    _departure_times: List[float] = field(default_factory=list)
    _waiting_times: List[float] = field(default_factory=list)
    
    # Congestion thresholds
    _congestion_low: int = 5
    _congestion_medium: int = 10
    _congestion_high: int = 20
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is within the lane boundaries"""
        return (self.x1 <= x <= self.x2) and (self.y1 <= y <= self.y2)
    
    def update_queue(
        self,
        vehicle_count: int,
        waiting_times: List[float],
        timestamp: float,
    ):
        """
        Update queue state with new measurements.
        
        Args:
            vehicle_count: Number of vehicles in lane
            waiting_times: List of waiting times for vehicles
            timestamp: Current timestamp
        """
        self.state.queue_length = vehicle_count
        self.state.queue_length_meters = vehicle_count * self.vehicle_spacing_meters
        self.state.occupancy = min(1.0, self.state.queue_length_meters / self.length_meters)
        
        if waiting_times:
            self.state.average_waiting_time = np.mean(waiting_times)
            self.state.max_waiting_time = max(waiting_times)
            self._waiting_times.extend(waiting_times)
        
        # Update congestion level
        self._update_congestion_level()
        
        self.state.timestamp = timestamp
    
    def record_arrival(self, track_id: int, timestamp: float):
        """Record vehicle arrival in lane"""
        if track_id not in self._vehicle_ids:
            self._vehicle_ids.add(track_id)
            self._arrival_times.append(timestamp)
            
            # Keep only recent arrivals for rate calculation
            self._trim_history()
    
    def record_departure(self, track_id: int, timestamp: float):
        """Record vehicle departure from lane"""
        if track_id in self._vehicle_ids:
            self._vehicle_ids.remove(track_id)
            self._departure_times.append(timestamp)
    
    def calculate_arrival_rate(self, window_seconds: float = 60.0) -> float:
        """
        Calculate vehicle arrival rate.
        
        Args:
            window_seconds: Time window for rate calculation
            
        Returns:
            Arrivals per minute
        """
        if not self._arrival_times:
            return 0.0
        
        current_time = time.time()
        cutoff = current_time - window_seconds
        recent_arrivals = [t for t in self._arrival_times if t > cutoff]
        
        if len(recent_arrivals) < 2:
            return 0.0
        
        duration_minutes = window_seconds / 60.0
        self.state.arrival_rate = len(recent_arrivals) / duration_minutes
        return self.state.arrival_rate
    
    def calculate_departure_rate(self, window_seconds: float = 60.0) -> float:
        """
        Calculate vehicle departure rate.
        
        Args:
            window_seconds: Time window for rate calculation
            
        Returns:
            Departures per minute
        """
        if not self._departure_times:
            return 0.0
        
        current_time = time.time()
        cutoff = current_time - window_seconds
        recent_departures = [t for t in self._departure_times if t > cutoff]
        
        if len(recent_departures) < 2:
            return 0.0
        
        duration_minutes = window_seconds / 60.0
        self.state.departure_rate = len(recent_departures) / duration_minutes
        return self.state.departure_rate
    
    def _update_congestion_level(self):
        """Update congestion level based on queue length"""
        if self.state.queue_length >= self._congestion_high:
            self.state.congestion_level = CongestionLevel.CRITICAL
        elif self.state.queue_length >= self._congestion_medium:
            self.state.congestion_level = CongestionLevel.HIGH
        elif self.state.queue_length >= self._congestion_low:
            self.state.congestion_level = CongestionLevel.MEDIUM
        else:
            self.state.congestion_level = CongestionLevel.LOW
    
    def _trim_history(self, max_age_seconds: float = 300.0):
        """Remove old entries from history"""
        current_time = time.time()
        cutoff = current_time - max_age_seconds
        
        self._arrival_times = [t for t in self._arrival_times if t > cutoff]
        self._departure_times = [t for t in self._departure_times if t > cutoff]
    
    def get_flow_imbalance(self) -> float:
        """
        Calculate flow imbalance (arrival - departure rate).
        Positive values indicate queue building.
        
        Returns:
            Flow imbalance in vehicles per minute
        """
        return self.state.arrival_rate - self.state.departure_rate
    
    def reset(self):
        """Reset lane state"""
        self.state = LaneState()
        self._vehicle_ids.clear()
        self._arrival_times.clear()
        self._departure_times.clear()
        self._waiting_times.clear()
    
    def to_dict(self) -> dict:
        """Convert lane to dictionary"""
        return {
            "lane_id": self.lane_id,
            "direction": self.direction,
            "lane_type": self.lane_type,
            "geometry": {
                "x1": self.x1,
                "y1": self.y1,
                "x2": self.x2,
                "y2": self.y2,
            },
            "parameters": {
                "length_meters": self.length_meters,
                "width_meters": self.width_meters,
            },
            "state": self.state.to_dict(),
            "flow_imbalance": self.get_flow_imbalance(),
        }
