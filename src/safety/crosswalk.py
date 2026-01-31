"""
Crosswalk data structure and state tracking
"""

from dataclasses import dataclass, field
from typing import List, Set, Optional, Tuple
from enum import Enum
import time


class CrosswalkPhase(Enum):
    """Crosswalk signal phase"""
    DONT_WALK = "dont_walk"
    WALK = "walk"
    FLASHING_DONT_WALK = "flashing_dont_walk"


@dataclass
class CrosswalkState:
    """Current state of a crosswalk"""
    phase: CrosswalkPhase = CrosswalkPhase.DONT_WALK
    pedestrian_count: int = 0
    pedestrians_in_crosswalk: int = 0
    estimated_clear_time: float = 0.0  # seconds to clear crosswalk
    
    # Timing
    phase_start_time: float = field(default_factory=time.time)
    minimum_crossing_time: float = 15.0  # Minimum seconds for crossing
    remaining_time: float = 0.0
    
    # Safety flags
    is_safe_to_change: bool = True
    requires_extension: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "phase": self.phase.value,
            "pedestrian_count": self.pedestrian_count,
            "pedestrians_in_crosswalk": self.pedestrians_in_crosswalk,
            "estimated_clear_time": self.estimated_clear_time,
            "minimum_crossing_time": self.minimum_crossing_time,
            "remaining_time": self.remaining_time,
            "is_safe_to_change": self.is_safe_to_change,
            "requires_extension": self.requires_extension,
        }


@dataclass
class Crosswalk:
    """
    Represents a pedestrian crosswalk with safety management.
    """
    crosswalk_id: int
    direction: str  # 'north', 'south', 'east', 'west'
    
    # Crosswalk geometry (in pixels)
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    
    # Physical dimensions
    width_meters: float = 10.0  # Crossing width
    length_meters: float = 3.0  # Crosswalk length (stripe width)
    
    # Timing parameters
    crossing_speed: float = 1.2  # Average walking speed (m/s)
    min_time_per_meter: float = 1.5  # Minimum seconds per meter
    clearance_buffer: float = 5.0  # Additional clearance time
    
    # Current state
    state: CrosswalkState = field(default_factory=CrosswalkState)
    
    # Pedestrian tracking
    _pedestrian_ids: Set[int] = field(default_factory=set)
    _pedestrian_positions: dict = field(default_factory=dict)
    
    # Request tracking
    _pending_requests: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate minimum crossing time based on width"""
        self.state.minimum_crossing_time = self.calculate_minimum_crossing_time()
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is within the crosswalk boundaries"""
        return (self.x1 <= x <= self.x2) and (self.y1 <= y <= self.y2)
    
    def calculate_minimum_crossing_time(
        self,
        crowd_size: Optional[int] = None,
    ) -> float:
        """
        Calculate minimum crossing time based on width and crowd size.
        
        Larger crowds may slow crossing and require more time.
        
        Args:
            crowd_size: Number of pedestrians (uses current count if None)
            
        Returns:
            Minimum crossing time in seconds
        """
        if crowd_size is None:
            crowd_size = self.state.pedestrian_count
        
        # Base time from crossing width
        base_time = self.width_meters * self.min_time_per_meter
        
        # Crowd adjustment (slower with more people)
        crowd_factor = 1.0 + (crowd_size * 0.05)  # 5% slower per pedestrian
        crowd_factor = min(crowd_factor, 2.0)  # Cap at 2x slowdown
        
        # Add clearance buffer
        crossing_time = base_time * crowd_factor + self.clearance_buffer
        
        return crossing_time
    
    def update_pedestrians(
        self,
        pedestrian_positions: List[Tuple[int, float, float]],
        timestamp: float,
    ):
        """
        Update pedestrian tracking in crosswalk.
        
        Args:
            pedestrian_positions: List of (track_id, x, y) tuples
            timestamp: Current timestamp
        """
        current_ids = set()
        in_crosswalk = 0
        
        for track_id, x, y in pedestrian_positions:
            if self.contains_point(x, y):
                current_ids.add(track_id)
                in_crosswalk += 1
                self._pedestrian_positions[track_id] = (x, y, timestamp)
        
        self._pedestrian_ids = current_ids
        self.state.pedestrians_in_crosswalk = in_crosswalk
        
        # Update minimum crossing time for current crowd
        self.state.minimum_crossing_time = self.calculate_minimum_crossing_time(in_crosswalk)
    
    def set_pedestrian_count(self, count: int):
        """Set total pedestrian count waiting or crossing"""
        self.state.pedestrian_count = count
        self.state.minimum_crossing_time = self.calculate_minimum_crossing_time(count)
    
    def set_phase(self, phase: CrosswalkPhase, timestamp: Optional[float] = None):
        """
        Set crosswalk phase.
        
        Args:
            phase: New phase
            timestamp: Phase start time
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.state.phase = phase
        self.state.phase_start_time = timestamp
        
        if phase == CrosswalkPhase.WALK:
            self.state.remaining_time = self.state.minimum_crossing_time
        elif phase == CrosswalkPhase.FLASHING_DONT_WALK:
            # Flashing phase gives remaining clearance time
            self.state.remaining_time = self.clearance_buffer
    
    def update_remaining_time(self, current_time: float):
        """Update remaining time in current phase"""
        elapsed = current_time - self.state.phase_start_time
        
        if self.state.phase == CrosswalkPhase.WALK:
            total_time = self.state.minimum_crossing_time
        elif self.state.phase == CrosswalkPhase.FLASHING_DONT_WALK:
            total_time = self.clearance_buffer
        else:
            total_time = 0
        
        self.state.remaining_time = max(0, total_time - elapsed)
    
    def check_safe_to_change(self, current_time: float) -> bool:
        """
        Check if it's safe to change phase (no pedestrians in crosswalk).
        
        This is a HARD safety constraint that overrides optimization.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            True if safe to change phase
        """
        self.update_remaining_time(current_time)
        
        # Not safe if pedestrians are in the crosswalk
        if self.state.pedestrians_in_crosswalk > 0:
            self.state.is_safe_to_change = False
            return False
        
        # Not safe if walk phase hasn't completed minimum time
        if self.state.phase == CrosswalkPhase.WALK:
            elapsed = current_time - self.state.phase_start_time
            if elapsed < self.state.minimum_crossing_time:
                self.state.is_safe_to_change = False
                return False
        
        self.state.is_safe_to_change = True
        return True
    
    def check_extension_required(
        self,
        high_traffic_threshold: int = 10,
    ) -> bool:
        """
        Check if pedestrian green phase should be extended.
        
        Args:
            high_traffic_threshold: Pedestrian count for high traffic
            
        Returns:
            True if extension is required
        """
        # Extend if pedestrians still in crosswalk
        if self.state.pedestrians_in_crosswalk > 0:
            self.state.requires_extension = True
            return True
        
        # Extend during high foot traffic
        if self.state.pedestrian_count >= high_traffic_threshold:
            self.state.requires_extension = True
            return True
        
        self.state.requires_extension = False
        return False
    
    def estimate_clear_time(self, current_time: float) -> float:
        """
        Estimate time for crosswalk to clear.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Estimated seconds to clear
        """
        if self.state.pedestrians_in_crosswalk == 0:
            return 0.0
        
        # Estimate based on crossing width and walking speed
        clear_time = self.width_meters / self.crossing_speed
        
        # Add buffer for uncertainty
        clear_time *= 1.2
        
        self.state.estimated_clear_time = clear_time
        return clear_time
    
    def add_crossing_request(self, timestamp: float):
        """Add a pedestrian crossing request (button press)"""
        self._pending_requests.append(timestamp)
        
        # Keep only recent requests
        cutoff = timestamp - 120  # 2 minute window
        self._pending_requests = [t for t in self._pending_requests if t > cutoff]
    
    def has_pending_request(self, max_age: float = 60.0) -> bool:
        """Check if there are pending crossing requests"""
        if not self._pending_requests:
            return False
        
        current_time = time.time()
        return any(current_time - t < max_age for t in self._pending_requests)
    
    def clear_requests(self):
        """Clear pending crossing requests (after walk phase)"""
        self._pending_requests.clear()
    
    def reset(self):
        """Reset crosswalk state"""
        self.state = CrosswalkState()
        self.state.minimum_crossing_time = self.calculate_minimum_crossing_time()
        self._pedestrian_ids.clear()
        self._pedestrian_positions.clear()
        self._pending_requests.clear()
    
    def to_dict(self) -> dict:
        """Convert crosswalk to dictionary"""
        return {
            "crosswalk_id": self.crosswalk_id,
            "direction": self.direction,
            "geometry": {
                "x1": self.x1,
                "y1": self.y1,
                "x2": self.x2,
                "y2": self.y2,
            },
            "dimensions": {
                "width_meters": self.width_meters,
                "length_meters": self.length_meters,
            },
            "state": self.state.to_dict(),
            "pending_requests": len(self._pending_requests),
        }
