"""
Detection result data structure
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class BoundingBox:
    """Bounding box for detected object"""
    x1: float  # Top-left x
    y1: float  # Top-left y
    x2: float  # Bottom-right x
    y2: float  # Bottom-right y
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def width(self) -> float:
        """Get width of bounding box"""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """Get height of bounding box"""
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        """Get area of bounding box"""
        return self.width * self.height
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x1, y1, x2, y2]"""
        return np.array([self.x1, self.y1, self.x2, self.y2])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "BoundingBox":
        """Create from numpy array"""
        return cls(x1=arr[0], y1=arr[1], x2=arr[2], y2=arr[3])


@dataclass
class Detection:
    """Single object detection"""
    bbox: BoundingBox
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None
    
    # Additional attributes for specific detection types
    is_emergency: bool = False
    is_pedestrian: bool = False
    is_cyclist: bool = False
    is_vehicle: bool = False
    
    # Emergency vehicle specific
    has_flashing_lights: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "bbox": [self.bbox.x1, self.bbox.y1, self.bbox.x2, self.bbox.y2],
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "track_id": self.track_id,
            "is_emergency": self.is_emergency,
            "is_pedestrian": self.is_pedestrian,
            "is_cyclist": self.is_cyclist,
            "is_vehicle": self.is_vehicle,
            "has_flashing_lights": self.has_flashing_lights,
        }


@dataclass
class DetectionResult:
    """Complete detection result for a frame"""
    frame_id: int
    timestamp: float
    detections: List[Detection] = field(default_factory=list)
    
    # Aggregated counts
    vehicle_count: int = 0
    pedestrian_count: int = 0
    cyclist_count: int = 0
    emergency_count: int = 0
    
    # Processing info
    inference_time_ms: float = 0.0
    
    def update_counts(self):
        """Update aggregated counts from detections"""
        self.vehicle_count = sum(1 for d in self.detections if d.is_vehicle)
        self.pedestrian_count = sum(1 for d in self.detections if d.is_pedestrian)
        self.cyclist_count = sum(1 for d in self.detections if d.is_cyclist)
        self.emergency_count = sum(1 for d in self.detections if d.is_emergency)
    
    def get_vehicles(self) -> List[Detection]:
        """Get all vehicle detections"""
        return [d for d in self.detections if d.is_vehicle]
    
    def get_pedestrians(self) -> List[Detection]:
        """Get all pedestrian detections"""
        return [d for d in self.detections if d.is_pedestrian]
    
    def get_cyclists(self) -> List[Detection]:
        """Get all cyclist detections"""
        return [d for d in self.detections if d.is_cyclist]
    
    def get_emergency_vehicles(self) -> List[Detection]:
        """Get all emergency vehicle detections"""
        return [d for d in self.detections if d.is_emergency]
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "detections": [d.to_dict() for d in self.detections],
            "vehicle_count": self.vehicle_count,
            "pedestrian_count": self.pedestrian_count,
            "cyclist_count": self.cyclist_count,
            "emergency_count": self.emergency_count,
            "inference_time_ms": self.inference_time_ms,
        }
