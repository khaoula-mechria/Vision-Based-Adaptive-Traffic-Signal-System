"""
Unit tests for perception module
"""

import numpy as np
import pytest

from src.perception.detection_result import BoundingBox, Detection, DetectionResult


class TestBoundingBox:
    """Tests for BoundingBox class"""
    
    def test_center_calculation(self):
        """Test center point calculation"""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        assert bbox.center == (50, 50)
    
    def test_width_height(self):
        """Test width and height calculation"""
        bbox = BoundingBox(x1=10, y1=20, x2=50, y2=80)
        assert bbox.width == 40
        assert bbox.height == 60
    
    def test_area(self):
        """Test area calculation"""
        bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        assert bbox.area == 100
    
    def test_to_array(self):
        """Test conversion to numpy array"""
        bbox = BoundingBox(x1=1, y1=2, x2=3, y2=4)
        arr = bbox.to_array()
        assert isinstance(arr, np.ndarray)
        assert list(arr) == [1, 2, 3, 4]
    
    def test_from_array(self):
        """Test creation from numpy array"""
        arr = np.array([1, 2, 3, 4])
        bbox = BoundingBox.from_array(arr)
        assert bbox.x1 == 1
        assert bbox.y1 == 2
        assert bbox.x2 == 3
        assert bbox.y2 == 4


class TestDetection:
    """Tests for Detection class"""
    
    def test_detection_creation(self):
        """Test detection object creation"""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        det = Detection(
            bbox=bbox,
            class_id=2,
            class_name="car",
            confidence=0.9,
            is_vehicle=True,
        )
        
        assert det.class_name == "car"
        assert det.confidence == 0.9
        assert det.is_vehicle is True
        assert det.is_pedestrian is False
    
    def test_detection_to_dict(self):
        """Test conversion to dictionary"""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        det = Detection(
            bbox=bbox,
            class_id=0,
            class_name="person",
            confidence=0.85,
            is_pedestrian=True,
        )
        
        d = det.to_dict()
        assert d["class_name"] == "person"
        assert d["confidence"] == 0.85
        assert d["is_pedestrian"] is True


class TestDetectionResult:
    """Tests for DetectionResult class"""
    
    def test_update_counts(self):
        """Test count aggregation"""
        result = DetectionResult(frame_id=1, timestamp=0.0)
        
        # Add detections
        result.detections = [
            Detection(
                bbox=BoundingBox(0, 0, 10, 10),
                class_id=2, class_name="car", confidence=0.9,
                is_vehicle=True,
            ),
            Detection(
                bbox=BoundingBox(20, 20, 30, 30),
                class_id=2, class_name="car", confidence=0.85,
                is_vehicle=True,
            ),
            Detection(
                bbox=BoundingBox(40, 40, 50, 50),
                class_id=0, class_name="person", confidence=0.8,
                is_pedestrian=True,
            ),
        ]
        
        result.update_counts()
        
        assert result.vehicle_count == 2
        assert result.pedestrian_count == 1
        assert result.cyclist_count == 0
    
    def test_get_vehicles(self):
        """Test filtering vehicle detections"""
        result = DetectionResult(frame_id=1, timestamp=0.0)
        result.detections = [
            Detection(
                bbox=BoundingBox(0, 0, 10, 10),
                class_id=2, class_name="car", confidence=0.9,
                is_vehicle=True,
            ),
            Detection(
                bbox=BoundingBox(20, 20, 30, 30),
                class_id=0, class_name="person", confidence=0.8,
                is_pedestrian=True,
            ),
        ]
        
        vehicles = result.get_vehicles()
        assert len(vehicles) == 1
        assert vehicles[0].class_name == "car"
    
    def test_get_pedestrians(self):
        """Test filtering pedestrian detections"""
        result = DetectionResult(frame_id=1, timestamp=0.0)
        result.detections = [
            Detection(
                bbox=BoundingBox(0, 0, 10, 10),
                class_id=0, class_name="person", confidence=0.9,
                is_pedestrian=True,
            ),
        ]
        
        pedestrians = result.get_pedestrians()
        assert len(pedestrians) == 1
