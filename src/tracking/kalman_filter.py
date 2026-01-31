"""
Kalman Filter for bounding box tracking
Based on SORT (Simple Online and Realtime Tracking)
"""

import numpy as np
from typing import Optional


class KalmanBoxTracker:
    """
    Kalman Filter for tracking bounding boxes in image space.
    
    The state vector is [x, y, s, r, vx, vy, vs] where:
    - x, y: center position
    - s: scale (area)
    - r: aspect ratio
    - vx, vy, vs: velocities
    """
    
    count = 0
    
    def __init__(self, bbox: np.ndarray):
        """
        Initialize a tracker using initial bounding box.
        
        Args:
            bbox: Initial bounding box [x1, y1, x2, y2]
        """
        # State dimension: [x, y, s, r, vx, vy, vs]
        self.dim_x = 7
        # Measurement dimension: [x, y, s, r]
        self.dim_z = 4
        
        # State vector
        self.x = np.zeros((self.dim_x, 1))
        
        # State covariance matrix
        self.P = np.eye(self.dim_x)
        self.P[4:, 4:] *= 1000.0  # High uncertainty for velocities
        self.P *= 10.0
        
        # State transition matrix
        self.F = np.eye(self.dim_x)
        self.F[0, 4] = 1  # x += vx
        self.F[1, 5] = 1  # y += vy
        self.F[2, 6] = 1  # s += vs
        
        # Measurement matrix
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[:4, :4] = np.eye(4)
        
        # Measurement noise
        self.R = np.eye(self.dim_z)
        self.R[2, 2] *= 10.0
        self.R[3, 3] *= 10.0
        
        # Process noise
        self.Q = np.eye(self.dim_x)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01
        
        # Initialize state from bbox
        self.x[:4] = self._bbox_to_z(bbox).reshape(-1, 1)
        
        # Track management
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
    def _bbox_to_z(self, bbox: np.ndarray) -> np.ndarray:
        """
        Convert bounding box to measurement space [x, y, s, r].
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Measurement vector [x, y, s, r]
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h) if h > 0 else 1.0
        return np.array([x, y, s, r])
    
    def _z_to_bbox(self, z: np.ndarray) -> np.ndarray:
        """
        Convert measurement space to bounding box.
        
        Args:
            z: Measurement vector [x, y, s, r]
            
        Returns:
            Bounding box [x1, y1, x2, y2]
        """
        x, y, s, r = z.flatten()
        w = np.sqrt(s * r) if s > 0 and r > 0 else 0
        h = s / w if w > 0 else 0
        return np.array([
            x - w / 2.0,
            y - h / 2.0,
            x + w / 2.0,
            y + h / 2.0,
        ])
    
    def predict(self) -> np.ndarray:
        """
        Predict next state and return predicted bounding box.
        
        Returns:
            Predicted bounding box [x1, y1, x2, y2]
        """
        # Handle negative area
        if self.x[6] + self.x[2] <= 0:
            self.x[6] *= 0.0
        
        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        return self._z_to_bbox(self.x[:4])
    
    def update(self, bbox: np.ndarray):
        """
        Update state with observed bounding box.
        
        Args:
            bbox: Observed bounding box [x1, y1, x2, y2]
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
        # Measurement
        z = self._bbox_to_z(bbox).reshape(-1, 1)
        
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        
        # Update covariance
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P
    
    def get_state(self) -> np.ndarray:
        """
        Get current bounding box estimate.
        
        Returns:
            Current bounding box [x1, y1, x2, y2]
        """
        return self._z_to_bbox(self.x[:4])
    
    def get_velocity(self) -> np.ndarray:
        """
        Get current velocity estimate.
        
        Returns:
            Velocity [vx, vy]
        """
        return self.x[4:6].flatten()


def iou(bb1: np.ndarray, bb2: np.ndarray) -> float:
    """
    Calculate IoU between two bounding boxes.
    
    Args:
        bb1: First bounding box [x1, y1, x2, y2]
        bb2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    # Intersection
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Union
    area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """
    Calculate IoU matrix between two sets of bounding boxes.
    
    Args:
        bb_test: First set of bounding boxes (N, 4)
        bb_gt: Second set of bounding boxes (M, 4)
        
    Returns:
        IoU matrix (N, M)
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    
    intersection = w * h
    
    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    
    union = area_test + area_gt - intersection
    
    return intersection / np.maximum(union, 1e-10)
