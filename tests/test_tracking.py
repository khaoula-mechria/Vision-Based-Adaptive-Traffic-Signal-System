"""
Unit tests for tracking module
"""

import numpy as np
import pytest

from src.tracking.track import Track
from src.tracking.kalman_filter import KalmanBoxTracker, iou, iou_batch


class TestTrack:
    """Tests for Track class"""
    
    def test_track_creation(self):
        """Test track creation"""
        track = Track(
            track_id=1,
            class_name="car",
            class_id=2,
        )
        
        assert track.track_id == 1
        assert track.class_name == "car"
        assert track.is_waiting is False
    
    def test_update_position(self):
        """Test position update"""
        track = Track(track_id=1, class_name="car", class_id=2)
        
        bbox = np.array([0, 0, 100, 100])
        track.update_position(bbox, 1.0)
        
        assert len(track.position_history) == 1
        assert track.position_history[0] == (50, 50)
    
    def test_velocity_calculation(self):
        """Test velocity calculation from history"""
        track = Track(track_id=1, class_name="car", class_id=2)
        
        # Add position history
        track.update_position(np.array([0, 0, 100, 100]), 0.0)
        track.update_position(np.array([10, 0, 110, 100]), 1.0)
        
        vx, vy = track.get_velocity()
        assert vx == 10.0  # Moved 10 pixels in 1 second
        assert vy == 0.0
    
    def test_is_stationary(self):
        """Test stationary detection"""
        track = Track(track_id=1, class_name="car", class_id=2)
        
        # Stationary vehicle
        track.update_position(np.array([0, 0, 100, 100]), 0.0)
        track.update_position(np.array([0, 0, 100, 100]), 1.0)
        
        assert bool(track.is_stationary(threshold=5.0)) is True
    
    def test_waiting_time_tracking(self):
        """Test waiting time calculation"""
        track = Track(track_id=1, class_name="car", class_id=2)
        
        # Start waiting
        track.start_waiting(0.0)
        assert track.is_waiting is True
        
        # Check waiting time
        current_wait = track.get_current_waiting_time(10.0)
        assert current_wait == 10.0
        
        # Stop waiting
        track.stop_waiting(15.0)
        assert track.is_waiting is False
        assert track.total_waiting_time == 15.0


class TestKalmanFilter:
    """Tests for Kalman filter tracking"""
    
    def test_kalman_init(self):
        """Test Kalman filter initialization"""
        bbox = np.array([0, 0, 100, 100])
        tracker = KalmanBoxTracker(bbox)
        
        assert tracker.time_since_update == 0
        assert tracker.hits == 0
    
    def test_predict(self):
        """Test prediction step"""
        bbox = np.array([0, 0, 100, 100])
        tracker = KalmanBoxTracker(bbox)
        
        predicted = tracker.predict()
        
        assert len(predicted) == 4
        assert tracker.age == 1
    
    def test_update(self):
        """Test update step"""
        bbox = np.array([0, 0, 100, 100])
        tracker = KalmanBoxTracker(bbox)
        
        tracker.predict()
        tracker.update(np.array([5, 5, 105, 105]))
        
        assert tracker.hits == 1
        assert tracker.time_since_update == 0


class TestIOU:
    """Tests for IoU calculation"""
    
    def test_perfect_overlap(self):
        """Test perfect overlap case"""
        bb1 = np.array([0, 0, 100, 100])
        bb2 = np.array([0, 0, 100, 100])
        
        assert iou(bb1, bb2) == 1.0
    
    def test_no_overlap(self):
        """Test no overlap case"""
        bb1 = np.array([0, 0, 50, 50])
        bb2 = np.array([100, 100, 150, 150])
        
        assert iou(bb1, bb2) == 0.0
    
    def test_partial_overlap(self):
        """Test partial overlap"""
        bb1 = np.array([0, 0, 100, 100])
        bb2 = np.array([50, 50, 150, 150])
        
        result = iou(bb1, bb2)
        assert 0 < result < 1
    
    def test_iou_batch(self):
        """Test batch IoU calculation"""
        bb_test = np.array([
            [0, 0, 100, 100],
            [200, 200, 300, 300],
        ])
        bb_gt = np.array([
            [0, 0, 100, 100],
            [50, 50, 150, 150],
        ])
        
        result = iou_batch(bb_test, bb_gt)
        
        assert result.shape == (2, 2)
        assert result[0, 0] == 1.0  # Perfect overlap
