"""
Queue Estimation Module
Estimates lane-level queue lengths, waiting times, and congestion patterns
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from .lane import Lane, LaneState, CongestionLevel
from ..tracking.track import Track


class QueueEstimator:
    """
    Estimates queue length and flow metrics per lane.
    
    Provides:
    - Queue length estimation using spatial position and density
    - Vehicle arrival and departure rates
    - Average and maximum waiting times
    - Congestion pattern detection
    - Lane imbalance identification
    """
    
    def __init__(
        self,
        vehicle_length_meters: float = 4.5,
        meters_per_pixel: float = 0.1,
        stationary_threshold: float = 5.0,
        congestion_thresholds: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize queue estimator.
        
        Args:
            vehicle_length_meters: Average vehicle length
            meters_per_pixel: Scale factor for pixel to meter conversion
            stationary_threshold: Speed threshold for stationary detection (px/s)
            congestion_thresholds: Congestion level thresholds
        """
        self.vehicle_length_meters = vehicle_length_meters
        self.meters_per_pixel = meters_per_pixel
        self.stationary_threshold = stationary_threshold
        
        self.congestion_thresholds = congestion_thresholds or {
            "low": 5,
            "medium": 10,
            "high": 20,
        }
        
        # Lane definitions
        self.lanes: Dict[int, Lane] = {}
        
        # Track to lane assignment
        self._track_lane_map: Dict[int, int] = {}
        
        # Historical metrics
        self._metrics_history: List[dict] = []
        
    def add_lane(
        self,
        lane_id: int,
        direction: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        lane_type: str = "through",
        length_meters: float = 100.0,
    ):
        """
        Add a lane definition.
        
        Args:
            lane_id: Unique lane identifier
            direction: Lane direction ('north', 'south', 'east', 'west')
            x1, y1: Top-left corner in pixels
            x2, y2: Bottom-right corner in pixels
            lane_type: Type of lane ('through', 'left_turn', 'right_turn')
            length_meters: Physical lane length
        """
        lane = Lane(
            lane_id=lane_id,
            direction=direction,
            lane_type=lane_type,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            length_meters=length_meters,
        )
        
        # Set congestion thresholds
        lane._congestion_low = self.congestion_thresholds["low"]
        lane._congestion_medium = self.congestion_thresholds["medium"]
        lane._congestion_high = self.congestion_thresholds["high"]
        
        self.lanes[lane_id] = lane
    
    def setup_default_intersection(
        self,
        frame_width: int,
        frame_height: int,
        num_lanes_per_direction: int = 2,
    ):
        """
        Set up default lane definitions for a standard intersection.
        
        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            num_lanes_per_direction: Number of lanes per direction
        """
        center_x = frame_width / 2
        center_y = frame_height / 2
        lane_width = 50  # pixels
        
        lane_id = 0
        
        # North approach (vehicles coming from top)
        for i in range(num_lanes_per_direction):
            self.add_lane(
                lane_id=lane_id,
                direction="north",
                x1=center_x - (num_lanes_per_direction - i) * lane_width,
                y1=0,
                x2=center_x - (num_lanes_per_direction - i - 1) * lane_width,
                y2=center_y - 50,
                lane_type="through" if i > 0 else "left_turn",
            )
            lane_id += 1
        
        # South approach (vehicles coming from bottom)
        for i in range(num_lanes_per_direction):
            self.add_lane(
                lane_id=lane_id,
                direction="south",
                x1=center_x + i * lane_width,
                y1=center_y + 50,
                x2=center_x + (i + 1) * lane_width,
                y2=frame_height,
                lane_type="through" if i > 0 else "left_turn",
            )
            lane_id += 1
        
        # East approach (vehicles coming from right)
        for i in range(num_lanes_per_direction):
            self.add_lane(
                lane_id=lane_id,
                direction="east",
                x1=center_x + 50,
                y1=center_y - (num_lanes_per_direction - i) * lane_width,
                x2=frame_width,
                y2=center_y - (num_lanes_per_direction - i - 1) * lane_width,
                lane_type="through" if i > 0 else "left_turn",
            )
            lane_id += 1
        
        # West approach (vehicles coming from left)
        for i in range(num_lanes_per_direction):
            self.add_lane(
                lane_id=lane_id,
                direction="west",
                x1=0,
                y1=center_y + i * lane_width,
                x2=center_x - 50,
                y2=center_y + (i + 1) * lane_width,
                lane_type="through" if i > 0 else "left_turn",
            )
            lane_id += 1
    
    def update(
        self,
        tracks: List[Track],
        timestamp: Optional[float] = None,
    ) -> Dict[int, LaneState]:
        """
        Update queue estimation with tracked vehicles.
        
        Args:
            tracks: List of tracked objects
            timestamp: Current timestamp
            
        Returns:
            Dictionary mapping lane_id to LaneState
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Get only vehicle tracks
        vehicle_tracks = [t for t in tracks if t.is_vehicle]
        
        # Assign tracks to lanes
        lane_tracks: Dict[int, List[Track]] = {lane_id: [] for lane_id in self.lanes}
        
        for track in vehicle_tracks:
            lane_id = self._assign_to_lane(track)
            if lane_id is not None:
                lane_tracks[lane_id].append(track)
                
                # Record arrival if new to lane
                if track.track_id not in self._track_lane_map or self._track_lane_map[track.track_id] != lane_id:
                    self.lanes[lane_id].record_arrival(track.track_id, timestamp)
                    
                    # Record departure from previous lane
                    if track.track_id in self._track_lane_map:
                        prev_lane = self._track_lane_map[track.track_id]
                        if prev_lane in self.lanes:
                            self.lanes[prev_lane].record_departure(track.track_id, timestamp)
                
                self._track_lane_map[track.track_id] = lane_id
                
                # Update waiting status
                if track.is_stationary(self.stationary_threshold):
                    track.start_waiting(timestamp)
                else:
                    track.stop_waiting(timestamp)
        
        # Update each lane
        lane_states = {}
        for lane_id, lane in self.lanes.items():
            lane_vehicles = lane_tracks[lane_id]
            
            # Calculate waiting times
            waiting_times = [
                t.get_current_waiting_time(timestamp)
                for t in lane_vehicles
                if t.is_waiting
            ]
            
            # Update lane state
            lane.update_queue(
                vehicle_count=len(lane_vehicles),
                waiting_times=waiting_times,
                timestamp=timestamp,
            )
            
            # Calculate rates
            lane.calculate_arrival_rate()
            lane.calculate_departure_rate()
            
            lane_states[lane_id] = lane.state
        
        # Store metrics history
        self._store_metrics(lane_states, timestamp)
        
        return lane_states
    
    def _assign_to_lane(self, track: Track) -> Optional[int]:
        """
        Assign a track to a lane based on position.
        
        Args:
            track: Track to assign
            
        Returns:
            Lane ID or None if not in any lane
        """
        center_x, center_y = track.get_center()
        
        for lane_id, lane in self.lanes.items():
            if lane.contains_point(center_x, center_y):
                return lane_id
        
        return None
    
    def _store_metrics(self, lane_states: Dict[int, LaneState], timestamp: float):
        """Store metrics for historical analysis"""
        metrics = {
            "timestamp": timestamp,
            "lanes": {lid: state.to_dict() for lid, state in lane_states.items()},
            "total_queue": sum(s.queue_length for s in lane_states.values()),
            "avg_waiting_time": np.mean([s.average_waiting_time for s in lane_states.values()]) if lane_states else 0,
        }
        
        self._metrics_history.append(metrics)
        
        # Keep only recent history
        max_history = 1000
        if len(self._metrics_history) > max_history:
            self._metrics_history = self._metrics_history[-max_history:]
    
    def get_lane_state(self, lane_id: int) -> Optional[LaneState]:
        """Get current state of a specific lane"""
        if lane_id in self.lanes:
            return self.lanes[lane_id].state
        return None
    
    def get_all_lane_states(self) -> Dict[int, LaneState]:
        """Get current state of all lanes"""
        return {lid: lane.state for lid, lane in self.lanes.items()}
    
    def get_direction_states(self) -> Dict[str, LaneState]:
        """
        Get aggregated state per direction.
        
        Returns:
            Dictionary mapping direction to aggregated LaneState
        """
        direction_states: Dict[str, Dict] = {}
        
        for lane in self.lanes.values():
            direction = lane.direction
            if direction not in direction_states:
                direction_states[direction] = {
                    "queue_lengths": [],
                    "waiting_times": [],
                    "arrival_rates": [],
                    "departure_rates": [],
                }
            
            direction_states[direction]["queue_lengths"].append(lane.state.queue_length)
            direction_states[direction]["waiting_times"].append(lane.state.average_waiting_time)
            direction_states[direction]["arrival_rates"].append(lane.state.arrival_rate)
            direction_states[direction]["departure_rates"].append(lane.state.departure_rate)
        
        # Aggregate
        result = {}
        for direction, data in direction_states.items():
            state = LaneState()
            state.queue_length = sum(data["queue_lengths"])
            state.average_waiting_time = np.mean(data["waiting_times"]) if data["waiting_times"] else 0
            state.arrival_rate = sum(data["arrival_rates"])
            state.departure_rate = sum(data["departure_rates"])
            result[direction] = state
        
        return result
    
    def get_congestion_map(self) -> Dict[int, CongestionLevel]:
        """Get congestion level for each lane"""
        return {lid: lane.state.congestion_level for lid, lane in self.lanes.items()}
    
    def identify_imbalanced_lanes(self, threshold: float = 2.0) -> List[int]:
        """
        Identify lanes with significant flow imbalance.
        
        Args:
            threshold: Minimum imbalance to flag (vehicles per minute)
            
        Returns:
            List of lane IDs with imbalance
        """
        imbalanced = []
        for lane_id, lane in self.lanes.items():
            imbalance = lane.get_flow_imbalance()
            if abs(imbalance) > threshold:
                imbalanced.append(lane_id)
        return imbalanced
    
    def get_total_queue_length(self) -> int:
        """Get total queue length across all lanes"""
        return sum(lane.state.queue_length for lane in self.lanes.values())
    
    def get_max_waiting_time(self) -> float:
        """Get maximum waiting time across all lanes"""
        if not self.lanes:
            return 0.0
        return max(lane.state.max_waiting_time for lane in self.lanes.values())
    
    def get_summary(self) -> dict:
        """Get summary of queue estimation state"""
        states = self.get_all_lane_states()
        congestion = self.get_congestion_map()
        
        return {
            "total_lanes": len(self.lanes),
            "total_queue": self.get_total_queue_length(),
            "max_waiting_time": self.get_max_waiting_time(),
            "average_waiting_time": np.mean([s.average_waiting_time for s in states.values()]) if states else 0,
            "congested_lanes": sum(1 for c in congestion.values() if c in [CongestionLevel.HIGH, CongestionLevel.CRITICAL]),
            "imbalanced_lanes": len(self.identify_imbalanced_lanes()),
            "lanes": {lid: lane.to_dict() for lid, lane in self.lanes.items()},
        }
    
    def reset(self):
        """Reset all lane states"""
        for lane in self.lanes.values():
            lane.reset()
        self._track_lane_map.clear()
        self._metrics_history.clear()
