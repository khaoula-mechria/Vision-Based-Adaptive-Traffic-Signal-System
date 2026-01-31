"""
Track data structure for multi-object tracking
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time
import numpy as np


@dataclass
class Track:
    """
    Represents a tracked object with persistent ID.
    Maintains history of positions and timestamps for analysis.
    """
    track_id: int
    class_name: str
    class_id: int
    
    # Current state
    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4))
    confidence: float = 0.0
    
    # Track status
    hits: int = 0
    age: int = 0
    time_since_update: int = 0
    
    # History for analysis
    position_history: List[Tuple[float, float]] = field(default_factory=list)
    timestamp_history: List[float] = field(default_factory=list)
    
    # Timing information
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    
    # Lane assignment
    lane_id: Optional[int] = None
    
    # Vehicle-specific
    is_vehicle: bool = False
    is_pedestrian: bool = False
    is_cyclist: bool = False
    is_emergency: bool = False
    
    # Waiting time tracking
    waiting_start_time: Optional[float] = None
    total_waiting_time: float = 0.0
    is_waiting: bool = False
    
    def update_position(self, bbox: np.ndarray, timestamp: float):
        """Update track position and history"""
        self.bbox = bbox
        self.last_seen = timestamp
        
        # Calculate center
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Add to history
        self.position_history.append((center_x, center_y))
        self.timestamp_history.append(timestamp)
        
        # Keep history bounded
        max_history = 100
        if len(self.position_history) > max_history:
            self.position_history = self.position_history[-max_history:]
            self.timestamp_history = self.timestamp_history[-max_history:]
    
    def get_velocity(self) -> Tuple[float, float]:
        """
        Calculate velocity from recent position history.
        
        Returns:
            Tuple of (vx, vy) in pixels per second
        """
        if len(self.position_history) < 2:
            return (0.0, 0.0)
        
        # Use recent positions for velocity calculation
        n_points = min(5, len(self.position_history))
        recent_positions = self.position_history[-n_points:]
        recent_times = self.timestamp_history[-n_points:]
        
        # Calculate average velocity
        dx = recent_positions[-1][0] - recent_positions[0][0]
        dy = recent_positions[-1][1] - recent_positions[0][1]
        dt = recent_times[-1] - recent_times[0]
        
        if dt > 0:
            return (dx / dt, dy / dt)
        return (0.0, 0.0)
    
    def get_speed(self) -> float:
        """Get speed magnitude in pixels per second"""
        vx, vy = self.get_velocity()
        return np.sqrt(vx**2 + vy**2)
    
    def is_stationary(self, threshold: float = 5.0) -> bool:
        """Check if track is stationary (stopped)"""
        return self.get_speed() < threshold
    
    def start_waiting(self, timestamp: float):
        """Mark track as waiting (stopped in queue)"""
        if not self.is_waiting:
            self.is_waiting = True
            self.waiting_start_time = timestamp
    
    def stop_waiting(self, timestamp: float):
        """Mark track as no longer waiting"""
        if self.is_waiting and self.waiting_start_time is not None:
            self.total_waiting_time += timestamp - self.waiting_start_time
            self.is_waiting = False
            self.waiting_start_time = None
    
    def get_current_waiting_time(self, current_time: float) -> float:
        """Get current waiting time including active wait"""
        total = self.total_waiting_time
        if self.is_waiting and self.waiting_start_time is not None:
            total += current_time - self.waiting_start_time
        return total
    
    def get_track_duration(self) -> float:
        """Get total duration this object has been tracked"""
        return self.last_seen - self.first_seen
    
    def get_center(self) -> Tuple[float, float]:
        """Get current center position"""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )
    
    def to_dict(self) -> dict:
        """Convert track to dictionary"""
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "bbox": self.bbox.tolist() if isinstance(self.bbox, np.ndarray) else list(self.bbox),
            "confidence": self.confidence,
            "hits": self.hits,
            "age": self.age,
            "time_since_update": self.time_since_update,
            "lane_id": self.lane_id,
            "is_vehicle": self.is_vehicle,
            "is_pedestrian": self.is_pedestrian,
            "is_cyclist": self.is_cyclist,
            "is_emergency": self.is_emergency,
            "total_waiting_time": self.total_waiting_time,
            "is_waiting": self.is_waiting,
            "velocity": self.get_velocity(),
            "speed": self.get_speed(),
        }
