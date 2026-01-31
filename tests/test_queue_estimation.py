"""
Unit tests for queue estimation module
"""

import pytest
import numpy as np
import time

from src.queue_estimation.lane import Lane, LaneState, CongestionLevel
from src.queue_estimation.queue_estimator import QueueEstimator
from src.tracking.track import Track


class TestLaneState:
    """Tests for LaneState class"""
    
    def test_default_state(self):
        """Test default state values"""
        state = LaneState()
        
        assert state.queue_length == 0
        assert state.congestion_level == CongestionLevel.LOW
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        state = LaneState(queue_length=5, average_waiting_time=15.0)
        
        d = state.to_dict()
        assert d["queue_length"] == 5
        assert d["average_waiting_time"] == 15.0


class TestLane:
    """Tests for Lane class"""
    
    def test_lane_creation(self):
        """Test lane creation"""
        lane = Lane(
            lane_id=0,
            direction="north",
            x1=0, y1=0, x2=100, y2=200,
        )
        
        assert lane.lane_id == 0
        assert lane.direction == "north"
    
    def test_contains_point(self):
        """Test point containment"""
        lane = Lane(
            lane_id=0, direction="north",
            x1=0, y1=0, x2=100, y2=200,
        )
        
        assert lane.contains_point(50, 100) is True
        assert lane.contains_point(150, 100) is False
    
    def test_update_queue(self):
        """Test queue update"""
        lane = Lane(lane_id=0, direction="north")
        
        lane.update_queue(
            vehicle_count=5,
            waiting_times=[10.0, 15.0, 20.0],
            timestamp=time.time(),
        )
        
        assert lane.state.queue_length == 5
        assert lane.state.average_waiting_time == 15.0
        assert lane.state.max_waiting_time == 20.0
    
    def test_congestion_levels(self):
        """Test congestion level calculation"""
        lane = Lane(lane_id=0, direction="north")
        
        # Low congestion
        lane.update_queue(vehicle_count=3, waiting_times=[], timestamp=time.time())
        assert lane.state.congestion_level == CongestionLevel.LOW
        
        # Medium congestion
        lane.update_queue(vehicle_count=7, waiting_times=[], timestamp=time.time())
        assert lane.state.congestion_level == CongestionLevel.MEDIUM
        
        # High congestion
        lane.update_queue(vehicle_count=15, waiting_times=[], timestamp=time.time())
        assert lane.state.congestion_level == CongestionLevel.HIGH
        
        # Critical congestion
        lane.update_queue(vehicle_count=25, waiting_times=[], timestamp=time.time())
        assert lane.state.congestion_level == CongestionLevel.CRITICAL


class TestQueueEstimator:
    """Tests for QueueEstimator class"""
    
    def test_add_lane(self):
        """Test adding lanes"""
        estimator = QueueEstimator()
        
        estimator.add_lane(
            lane_id=0,
            direction="north",
            x1=0, y1=0, x2=100, y2=200,
        )
        
        assert len(estimator.lanes) == 1
        assert 0 in estimator.lanes
    
    def test_setup_default_intersection(self):
        """Test default intersection setup"""
        estimator = QueueEstimator()
        estimator.setup_default_intersection(
            frame_width=640,
            frame_height=480,
            num_lanes_per_direction=2,
        )
        
        # Should have 8 lanes (2 per direction * 4 directions)
        assert len(estimator.lanes) == 8
    
    def test_get_total_queue(self):
        """Test total queue calculation"""
        estimator = QueueEstimator()
        
        estimator.add_lane(0, "north", 0, 0, 100, 200)
        estimator.add_lane(1, "south", 0, 0, 100, 200)
        
        estimator.lanes[0].update_queue(5, [], time.time())
        estimator.lanes[1].update_queue(3, [], time.time())
        
        assert estimator.get_total_queue_length() == 8
    
    def test_get_congestion_map(self):
        """Test congestion map generation"""
        estimator = QueueEstimator()
        
        estimator.add_lane(0, "north", 0, 0, 100, 200)
        estimator.add_lane(1, "south", 0, 0, 100, 200)
        
        estimator.lanes[0].update_queue(3, [], time.time())  # Low
        estimator.lanes[1].update_queue(15, [], time.time())  # High
        
        congestion = estimator.get_congestion_map()
        
        assert congestion[0] == CongestionLevel.LOW
        assert congestion[1] == CongestionLevel.HIGH
