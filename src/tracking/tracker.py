"""
Multi-Object Tracker using SORT algorithm
Maintains persistent IDs for detected objects to prevent double counting
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

from .kalman_filter import KalmanBoxTracker, iou_batch
from .track import Track
from ..perception.detection_result import Detection, DetectionResult


class MultiObjectTracker:
    """
    Multi-Object Tracker using SORT (Simple Online and Realtime Tracking).
    
    Maintains persistent track IDs across frames to enable:
    - Accurate counting without double-counting
    - Waiting time calculation
    - Velocity estimation
    - Lane assignment
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        """
        Initialize the multi-object tracker.
        
        Args:
            max_age: Maximum number of frames to keep alive a track without detections
            min_hits: Minimum number of detections before a track is confirmed
            iou_threshold: Minimum IoU for matching detections to tracks
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        # Active trackers (Kalman filters)
        self.trackers: List[KalmanBoxTracker] = []
        
        # Track metadata
        self.tracks: Dict[int, Track] = {}
        
        # Frame counter
        self.frame_count = 0
        
    def update(
        self,
        detection_result: DetectionResult,
    ) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detection_result: Detection result from object detector
            
        Returns:
            List of confirmed tracks with assigned track IDs
        """
        self.frame_count += 1
        timestamp = detection_result.timestamp
        
        # Convert detections to numpy array
        detections = detection_result.detections
        if len(detections) == 0:
            dets = np.empty((0, 5))
        else:
            dets = np.array([
                [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2, d.confidence]
                for d in detections
            ])
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        # Remove invalid trackers
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        
        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            dets, trks
        )
        
        # Update matched trackers with assigned detections
        for m in matched:
            det_idx, trk_idx = m
            self.trackers[trk_idx].update(dets[det_idx, :4])
            
            # Update track metadata
            track_id = self.trackers[trk_idx].id
            if track_id in self.tracks:
                track = self.tracks[track_id]
                track.update_position(dets[det_idx, :4], timestamp)
                track.confidence = dets[det_idx, 4]
                track.hits = self.trackers[trk_idx].hits
                track.age = self.trackers[trk_idx].age
                track.time_since_update = self.trackers[trk_idx].time_since_update
                
                # Update from detection info
                det = detections[det_idx]
                track.is_vehicle = det.is_vehicle
                track.is_pedestrian = det.is_pedestrian
                track.is_cyclist = det.is_cyclist
                track.is_emergency = det.is_emergency
        
        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            self.trackers.append(trk)
            
            # Create track metadata
            det = detections[i]
            track = Track(
                track_id=trk.id,
                class_name=det.class_name,
                class_id=det.class_id,
                bbox=dets[i, :4],
                confidence=det.confidence,
                is_vehicle=det.is_vehicle,
                is_pedestrian=det.is_pedestrian,
                is_cyclist=det.is_cyclist,
                is_emergency=det.is_emergency,
                first_seen=timestamp,
                last_seen=timestamp,
            )
            track.update_position(dets[i, :4], timestamp)
            self.tracks[trk.id] = track
        
        # Remove dead trackers
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        # Return confirmed tracks
        confirmed_tracks = []
        for trk in self.trackers:
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                track_id = trk.id
                if track_id in self.tracks:
                    track = self.tracks[track_id]
                    track.bbox = trk.get_state()
                    confirmed_tracks.append(track)
        
        # Update track IDs in detections
        for track in confirmed_tracks:
            for det in detections:
                # Match detection to track by bbox overlap
                det_bbox = np.array([det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2])
                if self._bbox_overlap(det_bbox, track.bbox) > 0.5:
                    det.track_id = track.track_id
                    break
        
        return confirmed_tracks
    
    def _associate_detections_to_trackers(
        self,
        detections: np.ndarray,
        trackers: np.ndarray,
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Associate detections to tracked objects using IoU and Hungarian algorithm.
        
        Args:
            detections: Detection boxes (N, 5) - [x1, y1, x2, y2, conf]
            trackers: Tracker boxes (M, 5)
            
        Returns:
            Tuple of (matches, unmatched_detections, unmatched_trackers)
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(detections))), []
        
        if len(detections) == 0:
            return np.empty((0, 2), dtype=int), [], list(range(len(trackers)))
        
        # Calculate IoU matrix
        iou_matrix = iou_batch(detections[:, :4], trackers[:, :4])
        
        if linear_sum_assignment is None:
            raise ImportError("scipy is required. Install with: pip install scipy")
        
        # Hungarian algorithm for optimal assignment
        if min(iou_matrix.shape) > 0:
            # Convert to cost matrix (1 - IoU)
            cost_matrix = 1 - iou_matrix
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            matched_indices = np.array(list(zip(row_indices, col_indices)))
        else:
            matched_indices = np.empty((0, 2), dtype=int)
        
        # Find unmatched detections
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        # Find unmatched trackers
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        
        # Filter out matched with low IoU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, unmatched_detections, unmatched_trackers
    
    def _bbox_overlap(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate overlap ratio between two bboxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        
        return intersection / area1 if area1 > 0 else 0.0
    
    def get_active_tracks(self) -> List[Track]:
        """Get all currently active tracks"""
        active = []
        for trk in self.trackers:
            if trk.time_since_update < self.max_age:
                track_id = trk.id
                if track_id in self.tracks:
                    active.append(self.tracks[track_id])
        return active
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get track by its ID"""
        return self.tracks.get(track_id)
    
    def get_vehicle_tracks(self) -> List[Track]:
        """Get all active vehicle tracks"""
        return [t for t in self.get_active_tracks() if t.is_vehicle]
    
    def get_pedestrian_tracks(self) -> List[Track]:
        """Get all active pedestrian tracks"""
        return [t for t in self.get_active_tracks() if t.is_pedestrian]
    
    def get_emergency_tracks(self) -> List[Track]:
        """Get all active emergency vehicle tracks"""
        return [t for t in self.get_active_tracks() if t.is_emergency]
    
    def reset(self):
        """Reset tracker state"""
        self.trackers = []
        self.tracks = {}
        self.frame_count = 0
        KalmanBoxTracker.count = 0
    
    def get_statistics(self) -> dict:
        """Get tracker statistics"""
        active = self.get_active_tracks()
        return {
            "frame_count": self.frame_count,
            "total_tracks": len(self.tracks),
            "active_tracks": len(active),
            "vehicle_tracks": len([t for t in active if t.is_vehicle]),
            "pedestrian_tracks": len([t for t in active if t.is_pedestrian]),
            "cyclist_tracks": len([t for t in active if t.is_cyclist]),
            "emergency_tracks": len([t for t in active if t.is_emergency]),
        }
