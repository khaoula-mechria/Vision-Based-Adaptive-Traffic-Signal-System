"""
Pedestrian Safety Manager
Enforces hard safety constraints that override traffic optimization
"""

import time
from typing import Dict, List, Optional, Tuple
import logging

from .crosswalk import Crosswalk, CrosswalkState, CrosswalkPhase
from ..tracking.track import Track


logger = logging.getLogger(__name__)


class SafetyViolation:
    """Represents a potential safety violation"""
    
    def __init__(
        self,
        violation_type: str,
        crosswalk_id: int,
        description: str,
        severity: str = "high",
    ):
        self.violation_type = violation_type
        self.crosswalk_id = crosswalk_id
        self.description = description
        self.severity = severity
        self.timestamp = time.time()
    
    def to_dict(self) -> dict:
        return {
            "type": self.violation_type,
            "crosswalk_id": self.crosswalk_id,
            "description": self.description,
            "severity": self.severity,
            "timestamp": self.timestamp,
        }


class PedestrianSafetyManager:
    """
    Manages pedestrian safety constraints for traffic signal control.
    
    SAFETY CONSTRAINTS ARE NON-NEGOTIABLE and override all traffic optimization:
    
    1. Guarantee minimum pedestrian crossing time based on crowd size and width
    2. Prevent phase changes that would endanger pedestrians in crosswalk
    3. Dynamically extend pedestrian green phases during high foot traffic
    4. Pedestrian safety ALWAYS overrides traffic optimization objectives
    """
    
    def __init__(
        self,
        min_time_per_meter: float = 1.5,
        default_crossing_width: float = 10.0,
        safety_margin: float = 2.0,
        clearance_time: float = 5.0,
        high_traffic_threshold: int = 10,
        phase_extension_increment: float = 5.0,
        max_phase_extension: float = 30.0,
    ):
        """
        Initialize the pedestrian safety manager.
        
        Args:
            min_time_per_meter: Minimum crossing time per meter of width
            default_crossing_width: Default crossing width in meters
            safety_margin: Additional safety buffer in seconds
            clearance_time: Clearance time after walk signal
            high_traffic_threshold: Pedestrian count threshold for high traffic
            phase_extension_increment: Time increment for phase extension
            max_phase_extension: Maximum total extension time
        """
        self.min_time_per_meter = min_time_per_meter
        self.default_crossing_width = default_crossing_width
        self.safety_margin = safety_margin
        self.clearance_time = clearance_time
        self.high_traffic_threshold = high_traffic_threshold
        self.phase_extension_increment = phase_extension_increment
        self.max_phase_extension = max_phase_extension
        
        # Crosswalk definitions
        self.crosswalks: Dict[int, Crosswalk] = {}
        
        # Safety state
        self._pending_violations: List[SafetyViolation] = []
        self._phase_extensions: Dict[int, float] = {}  # crosswalk_id -> total extension
        
        # Track mappings
        self._track_crosswalk_map: Dict[int, int] = {}
    
    def add_crosswalk(
        self,
        crosswalk_id: int,
        direction: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width_meters: Optional[float] = None,
    ):
        """
        Add a crosswalk definition.
        
        Args:
            crosswalk_id: Unique crosswalk identifier
            direction: Crosswalk direction
            x1, y1: Top-left corner in pixels
            x2, y2: Bottom-right corner in pixels
            width_meters: Physical crossing width
        """
        crosswalk = Crosswalk(
            crosswalk_id=crosswalk_id,
            direction=direction,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            width_meters=width_meters or self.default_crossing_width,
            min_time_per_meter=self.min_time_per_meter,
            clearance_buffer=self.clearance_time,
        )
        self.crosswalks[crosswalk_id] = crosswalk
    
    def setup_default_crosswalks(
        self,
        frame_width: int,
        frame_height: int,
    ):
        """
        Set up default crosswalks for a standard intersection.
        
        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        """
        center_x = frame_width / 2
        center_y = frame_height / 2
        crosswalk_length = 30  # pixels
        
        # North crosswalk (crosses east-west traffic)
        self.add_crosswalk(
            crosswalk_id=0,
            direction="north",
            x1=center_x - 100,
            y1=center_y - 50 - crosswalk_length,
            x2=center_x + 100,
            y2=center_y - 50,
        )
        
        # South crosswalk
        self.add_crosswalk(
            crosswalk_id=1,
            direction="south",
            x1=center_x - 100,
            y1=center_y + 50,
            x2=center_x + 100,
            y2=center_y + 50 + crosswalk_length,
        )
        
        # East crosswalk (crosses north-south traffic)
        self.add_crosswalk(
            crosswalk_id=2,
            direction="east",
            x1=center_x + 50,
            y1=center_y - 100,
            x2=center_x + 50 + crosswalk_length,
            y2=center_y + 100,
        )
        
        # West crosswalk
        self.add_crosswalk(
            crosswalk_id=3,
            direction="west",
            x1=center_x - 50 - crosswalk_length,
            y1=center_y - 100,
            x2=center_x - 50,
            y2=center_y + 100,
        )
    
    def update(
        self,
        pedestrian_tracks: List[Track],
        timestamp: Optional[float] = None,
    ) -> Dict[int, CrosswalkState]:
        """
        Update safety state with tracked pedestrians.
        
        Args:
            pedestrian_tracks: List of pedestrian tracks
            timestamp: Current timestamp
            
        Returns:
            Dictionary mapping crosswalk_id to CrosswalkState
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Get pedestrian positions
        positions = [
            (track.track_id, *track.get_center())
            for track in pedestrian_tracks
        ]
        
        # Update each crosswalk
        for crosswalk_id, crosswalk in self.crosswalks.items():
            # Filter positions for this crosswalk region
            # (include nearby pedestrians who may be waiting)
            crosswalk.update_pedestrians(positions, timestamp)
            
            # Count total pedestrians in vicinity
            vicinity_count = sum(
                1 for _, x, y in positions
                if self._in_vicinity(crosswalk, x, y)
            )
            crosswalk.set_pedestrian_count(vicinity_count)
            
            # Update remaining time
            crosswalk.update_remaining_time(timestamp)
        
        return {cid: cw.state for cid, cw in self.crosswalks.items()}
    
    def _in_vicinity(
        self,
        crosswalk: Crosswalk,
        x: float,
        y: float,
        margin: float = 50.0,
    ) -> bool:
        """Check if point is in vicinity of crosswalk (including waiting area)"""
        return (
            (crosswalk.x1 - margin <= x <= crosswalk.x2 + margin) and
            (crosswalk.y1 - margin <= y <= crosswalk.y2 + margin)
        )
    
    def can_change_phase(
        self,
        crosswalk_id: int,
        current_time: Optional[float] = None,
    ) -> Tuple[bool, Optional[SafetyViolation]]:
        """
        Check if it's safe to change from pedestrian phase.
        
        THIS IS A HARD SAFETY CONSTRAINT.
        
        Args:
            crosswalk_id: Crosswalk to check
            current_time: Current timestamp
            
        Returns:
            Tuple of (is_safe, violation_if_unsafe)
        """
        if current_time is None:
            current_time = time.time()
        
        if crosswalk_id not in self.crosswalks:
            return True, None
        
        crosswalk = self.crosswalks[crosswalk_id]
        
        # Check if pedestrians are in the crosswalk
        if not crosswalk.check_safe_to_change(current_time):
            violation = SafetyViolation(
                violation_type="pedestrian_in_crosswalk",
                crosswalk_id=crosswalk_id,
                description=f"Cannot change phase: {crosswalk.state.pedestrians_in_crosswalk} pedestrians in crosswalk",
                severity="critical",
            )
            self._pending_violations.append(violation)
            logger.warning(f"SAFETY BLOCK: {violation.description}")
            return False, violation
        
        return True, None
    
    def get_minimum_crossing_time(
        self,
        crosswalk_id: int,
        pedestrian_count: Optional[int] = None,
    ) -> float:
        """
        Get minimum crossing time for a crosswalk.
        
        Args:
            crosswalk_id: Crosswalk identifier
            pedestrian_count: Optional override for pedestrian count
            
        Returns:
            Minimum crossing time in seconds
        """
        if crosswalk_id not in self.crosswalks:
            return self.default_crossing_width * self.min_time_per_meter
        
        crosswalk = self.crosswalks[crosswalk_id]
        return crosswalk.calculate_minimum_crossing_time(pedestrian_count)
    
    def check_phase_extension_needed(
        self,
        crosswalk_id: int,
        current_time: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """
        Check if pedestrian green phase should be extended.
        
        Args:
            crosswalk_id: Crosswalk to check
            current_time: Current timestamp
            
        Returns:
            Tuple of (needs_extension, extension_amount)
        """
        if current_time is None:
            current_time = time.time()
        
        if crosswalk_id not in self.crosswalks:
            return False, 0.0
        
        crosswalk = self.crosswalks[crosswalk_id]
        
        # Check if extension is needed
        needs_extension = crosswalk.check_extension_required(self.high_traffic_threshold)
        
        if not needs_extension:
            return False, 0.0
        
        # Calculate extension amount
        current_extension = self._phase_extensions.get(crosswalk_id, 0.0)
        
        if current_extension >= self.max_phase_extension:
            logger.info(f"Maximum extension reached for crosswalk {crosswalk_id}")
            return False, 0.0
        
        extension = min(
            self.phase_extension_increment,
            self.max_phase_extension - current_extension,
        )
        
        return True, extension
    
    def apply_phase_extension(
        self,
        crosswalk_id: int,
        extension: float,
    ):
        """Apply phase extension to a crosswalk"""
        current = self._phase_extensions.get(crosswalk_id, 0.0)
        self._phase_extensions[crosswalk_id] = current + extension
        logger.info(f"Extended crosswalk {crosswalk_id} by {extension}s (total: {current + extension}s)")
    
    def reset_phase_extension(self, crosswalk_id: int):
        """Reset phase extension counter for a crosswalk"""
        self._phase_extensions[crosswalk_id] = 0.0
    
    def set_crosswalk_phase(
        self,
        crosswalk_id: int,
        phase: CrosswalkPhase,
        timestamp: Optional[float] = None,
    ):
        """
        Set the phase for a crosswalk.
        
        Args:
            crosswalk_id: Crosswalk identifier
            phase: New phase
            timestamp: Phase start time
        """
        if crosswalk_id in self.crosswalks:
            self.crosswalks[crosswalk_id].set_phase(phase, timestamp)
            
            if phase == CrosswalkPhase.WALK:
                self.crosswalks[crosswalk_id].clear_requests()
                self.reset_phase_extension(crosswalk_id)
    
    def get_safety_constraints(
        self,
        current_time: Optional[float] = None,
    ) -> Dict[str, any]:
        """
        Get all current safety constraints.
        
        Returns constraints that MUST be respected by the signal controller.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Dictionary of safety constraints
        """
        if current_time is None:
            current_time = time.time()
        
        constraints = {
            "blocked_phase_changes": [],
            "minimum_times": {},
            "required_extensions": {},
            "pedestrians_in_crosswalks": {},
        }
        
        for crosswalk_id, crosswalk in self.crosswalks.items():
            # Check if phase change is blocked
            is_safe, _ = self.can_change_phase(crosswalk_id, current_time)
            if not is_safe:
                constraints["blocked_phase_changes"].append(crosswalk_id)
            
            # Minimum crossing times
            constraints["minimum_times"][crosswalk_id] = crosswalk.state.minimum_crossing_time
            
            # Required extensions
            needs_ext, ext_amount = self.check_phase_extension_needed(crosswalk_id, current_time)
            if needs_ext:
                constraints["required_extensions"][crosswalk_id] = ext_amount
            
            # Pedestrians in crosswalks
            constraints["pedestrians_in_crosswalks"][crosswalk_id] = crosswalk.state.pedestrians_in_crosswalk
        
        return constraints
    
    def validate_signal_decision(
        self,
        proposed_phase: str,
        conflicting_crosswalks: List[int],
        current_time: Optional[float] = None,
    ) -> Tuple[bool, List[SafetyViolation]]:
        """
        Validate a proposed signal phase change against safety constraints.
        
        This is the final safety check before any phase change.
        
        Args:
            proposed_phase: Proposed signal phase
            conflicting_crosswalks: Crosswalks that conflict with this phase
            current_time: Current timestamp
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        if current_time is None:
            current_time = time.time()
        
        violations = []
        
        for crosswalk_id in conflicting_crosswalks:
            is_safe, violation = self.can_change_phase(crosswalk_id, current_time)
            if not is_safe and violation:
                violations.append(violation)
        
        is_valid = len(violations) == 0
        
        if not is_valid:
            logger.warning(
                f"SAFETY OVERRIDE: Proposed phase '{proposed_phase}' blocked due to "
                f"{len(violations)} safety violations"
            )
        
        return is_valid, violations
    
    def get_pending_violations(self) -> List[SafetyViolation]:
        """Get list of pending safety violations"""
        return self._pending_violations.copy()
    
    def clear_violations(self):
        """Clear pending violations"""
        self._pending_violations.clear()
    
    def get_summary(self) -> dict:
        """Get summary of safety state"""
        return {
            "crosswalks": {cid: cw.to_dict() for cid, cw in self.crosswalks.items()},
            "pending_violations": len(self._pending_violations),
            "phase_extensions": self._phase_extensions.copy(),
            "constraints": self.get_safety_constraints(),
        }
    
    def reset(self):
        """Reset safety manager state"""
        for crosswalk in self.crosswalks.values():
            crosswalk.reset()
        self._pending_violations.clear()
        self._phase_extensions.clear()
        self._track_crosswalk_map.clear()
