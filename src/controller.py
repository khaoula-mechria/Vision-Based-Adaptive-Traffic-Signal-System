"""
Adaptive Traffic Signal Controller
Main controller that integrates all components
"""

import time
from typing import Dict, List, Optional, Any
import logging
import numpy as np

from .perception import ObjectDetector, DetectionResult
from .tracking import MultiObjectTracker, Track
from .queue_estimation import QueueEstimator, LaneState
from .safety import PedestrianSafetyManager, CrosswalkPhase
from .emergency import EmergencyVehicleHandler, EmergencyState
from .rl_agent import TrafficSignalAgent, TrafficSignalEnv
from .utils import ConfigLoader, MetricsCollector

logger = logging.getLogger(__name__)


class SignalPhase:
    """Signal phase definition"""
    NS_GREEN = 0
    EW_GREEN = 1
    NS_LEFT = 2
    EW_LEFT = 3
    PEDESTRIAN = 4
    
    NAMES = ["NS_Green", "EW_Green", "NS_Left", "EW_Left", "Pedestrian"]
    
    @classmethod
    def get_name(cls, phase: int) -> str:
        return cls.NAMES[phase] if 0 <= phase < len(cls.NAMES) else "Unknown"


class AdaptiveTrafficController:
    """
    Vision-Based Adaptive Traffic Signal Controller.
    
    Integrates all system components:
    - Computer vision perception
    - Multi-object tracking
    - Queue and flow estimation
    - Pedestrian safety management
    - Emergency vehicle handling
    - Reinforcement learning optimization
    
    The controller ensures safety constraints override optimization objectives.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        use_rl: bool = True,
        model_path: Optional[str] = None,
    ):
        """
        Initialize the adaptive traffic controller.
        
        Args:
            config_path: Path to configuration file
            use_rl: Whether to use RL agent for optimization
            model_path: Path to trained RL model
        """
        # Load configuration
        self.config = ConfigLoader(config_path)
        
        # Initialize components
        self._init_perception()
        self._init_tracking()
        self._init_queue_estimation()
        self._init_safety()
        self._init_emergency()
        
        # Initialize RL agent if enabled
        self.use_rl = use_rl
        if use_rl:
            self._init_rl_agent(model_path)
        else:
            self.agent = None
        
        # Metrics collection
        self.metrics = MetricsCollector()
        
        # State tracking
        self.current_phase = SignalPhase.NS_GREEN
        self.phase_start_time = time.time()
        self.min_phase_duration = 10.0  # seconds
        self.max_phase_duration = 60.0  # seconds
        
        # Frame counter
        self.frame_count = 0
        
        logger.info("Adaptive Traffic Controller initialized")
    
    def _init_perception(self):
        """Initialize perception components"""
        perception_config = self.config.get_section("perception")
        self.detector = ObjectDetector(
            model_name=perception_config.get("model", "yolov8n"),
            confidence_threshold=perception_config.get("confidence_threshold", 0.5),
            device=perception_config.get("device", "cpu"),
        )
    
    def _init_tracking(self):
        """Initialize tracking components"""
        tracking_config = self.config.get_section("tracking")
        self.tracker = MultiObjectTracker(
            max_age=tracking_config.get("max_age", 30),
            min_hits=tracking_config.get("min_hits", 3),
            iou_threshold=tracking_config.get("iou_threshold", 0.3),
        )
    
    def _init_queue_estimation(self):
        """Initialize queue estimation"""
        queue_config = self.config.get_section("queue_estimation")
        self.queue_estimator = QueueEstimator(
            vehicle_length_meters=queue_config.get("vehicle_length_meters", 4.5),
            meters_per_pixel=queue_config.get("meters_per_pixel", 0.1),
        )
    
    def _init_safety(self):
        """Initialize safety management"""
        safety_config = self.config.get_section("safety")
        self.safety_manager = PedestrianSafetyManager(
            min_time_per_meter=safety_config.get("min_crossing_time_per_meter", 1.5),
            high_traffic_threshold=safety_config.get("high_traffic_pedestrian_count", 10),
        )
    
    def _init_emergency(self):
        """Initialize emergency handling"""
        emergency_config = self.config.get_section("emergency")
        self.emergency_handler = EmergencyVehicleHandler(
            detection_confidence=emergency_config.get("detection_confidence", 0.7),
            green_corridor_duration=emergency_config.get("green_corridor_duration", 15.0),
            cooldown_period=emergency_config.get("cooldown_period", 10.0),
        )
    
    def _init_rl_agent(self, model_path: Optional[str] = None):
        """Initialize RL agent"""
        rl_config = self.config.get_section("rl_agent")
        
        # State dimension based on environment
        state_dim = 28  # 8 queue + 8 wait + 4 ped + 4 ped_wait + 4 other
        action_dim = 5  # Number of phases
        
        self.agent = TrafficSignalAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=rl_config.get("algorithm", "DQN"),
            learning_rate=rl_config.get("learning_rate", 0.0001),
            gamma=rl_config.get("gamma", 0.99),
        )
        
        if model_path:
            self.agent.load(model_path)
            logger.info(f"RL agent loaded from {model_path}")
    
    def setup_intersection(
        self,
        frame_width: int,
        frame_height: int,
        num_lanes: int = 2,
    ):
        """
        Set up intersection configuration.
        
        Args:
            frame_width: Camera frame width
            frame_height: Camera frame height
            num_lanes: Number of lanes per direction
        """
        self.queue_estimator.setup_default_intersection(
            frame_width=frame_width,
            frame_height=frame_height,
            num_lanes_per_direction=num_lanes,
        )
        
        self.safety_manager.setup_default_crosswalks(
            frame_width=frame_width,
            frame_height=frame_height,
        )
        
        logger.info(f"Intersection configured: {frame_width}x{frame_height}, {num_lanes} lanes/direction")
    
    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Process a single camera frame and update controller state.
        
        Args:
            frame: Camera frame (BGR format)
            timestamp: Frame timestamp
            
        Returns:
            Dictionary containing processing results
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.frame_count += 1
        start_time = time.time()
        
        # 1. Object Detection
        detection_result = self.detector.detect(frame, timestamp)
        
        # 2. Multi-Object Tracking
        tracks = self.tracker.update(detection_result)
        
        # 3. Queue and Flow Estimation
        lane_states = self.queue_estimator.update(tracks, timestamp)
        
        # 4. Safety Update (pedestrian tracking)
        pedestrian_tracks = [t for t in tracks if t.is_pedestrian]
        crosswalk_states = self.safety_manager.update(pedestrian_tracks, timestamp)
        
        # 5. Emergency Vehicle Detection
        emergency_state = self.emergency_handler.update(tracks, timestamp)
        
        # Record processing latency
        process_time = (time.time() - start_time) * 1000
        self.metrics.record_latency(process_time, "frame_processing")
        self.metrics.record_latency(detection_result.inference_time_ms, "inference")
        
        # Build state representation for RL agent
        state = self._build_state(lane_states, crosswalk_states, emergency_state)
        
        return {
            "frame_id": self.frame_count,
            "timestamp": timestamp,
            "detections": detection_result.to_dict(),
            "tracks": [t.to_dict() for t in tracks],
            "lane_states": {k: v.to_dict() for k, v in lane_states.items()},
            "crosswalk_states": {k: v.to_dict() for k, v in crosswalk_states.items()},
            "emergency_state": emergency_state.to_dict(),
            "current_phase": SignalPhase.get_name(self.current_phase),
            "rl_state": state,
            "processing_time_ms": process_time,
        }
    
    def _build_state(
        self,
        lane_states: Dict[int, LaneState],
        crosswalk_states: Dict[int, Any],
        emergency_state: EmergencyState,
    ) -> np.ndarray:
        """Build state vector for RL agent"""
        # Queue lengths (8 lanes)
        queue_lengths = np.zeros(8)
        waiting_times = np.zeros(8)
        for i, (lid, state) in enumerate(list(lane_states.items())[:8]):
            queue_lengths[i] = state.queue_length
            waiting_times[i] = state.average_waiting_time
        
        # Pedestrian info (4 crosswalks)
        ped_counts = np.zeros(4)
        ped_waiting = np.zeros(4)
        for i, (cid, state) in enumerate(list(crosswalk_states.items())[:4]):
            ped_counts[i] = state.pedestrian_count
            # Calculate waiting based on phase time if in DONT_WALK
            if state.phase == CrosswalkPhase.DONT_WALK:
                ped_waiting[i] = time.time() - state.phase_start_time
        
        # Other state info
        emergency_detected = float(emergency_state.is_active())
        emergency_dir = float(emergency_state.active_direction if emergency_state.active_direction else -1)
        current_phase = float(self.current_phase)
        phase_time = time.time() - self.phase_start_time
        
        state = np.concatenate([
            queue_lengths,
            waiting_times,
            ped_counts,
            ped_waiting,
            [emergency_detected, emergency_dir, current_phase, phase_time],
        ])
        
        return state.astype(np.float32)
    
    def select_phase(
        self,
        state: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> int:
        """
        Select the next signal phase.
        
        Applies safety constraints and emergency preemption before RL optimization.
        
        Args:
            state: Current state vector
            timestamp: Current timestamp
            
        Returns:
            Selected phase
        """
        if timestamp is None:
            timestamp = time.time()
        
        phase_duration = timestamp - self.phase_start_time
        
        # 1. Check Emergency Preemption (highest priority)
        if self.emergency_handler.is_preemption_active():
            required_phase = self.emergency_handler.get_required_phase()
            if required_phase:
                phase_idx = SignalPhase.NAMES.index(required_phase) if required_phase in SignalPhase.NAMES else None
                if phase_idx is not None:
                    logger.warning(f"EMERGENCY PREEMPTION: Forcing phase {required_phase}")
                    return phase_idx
        
        # 2. Check Safety Constraints
        safety_constraints = self.safety_manager.get_safety_constraints(timestamp)
        
        # If pedestrians in crosswalk, cannot change to conflicting vehicle phase
        blocked_phases = []
        for crosswalk_id in safety_constraints["blocked_phase_changes"]:
            # Map crosswalk to conflicting phases
            if crosswalk_id in [0, 1]:  # NS crosswalks
                blocked_phases.extend([SignalPhase.EW_GREEN, SignalPhase.EW_LEFT])
            else:  # EW crosswalks
                blocked_phases.extend([SignalPhase.NS_GREEN, SignalPhase.NS_LEFT])
        
        # Check phase extension requirements
        for crosswalk_id, extension in safety_constraints["required_extensions"].items():
            if self.current_phase == SignalPhase.PEDESTRIAN:
                self.safety_manager.apply_phase_extension(crosswalk_id, extension)
                logger.info(f"Safety extension applied to crosswalk {crosswalk_id}")
        
        # 3. Check minimum phase duration
        if phase_duration < self.min_phase_duration:
            return self.current_phase
        
        # 4. Use RL agent for phase selection
        if self.use_rl and self.agent:
            proposed_phase = self.agent.select_action(state, training=False)
        else:
            # Simple round-robin fallback
            proposed_phase = (self.current_phase + 1) % 5
        
        # 5. Validate proposed phase against safety constraints
        if proposed_phase in blocked_phases:
            logger.info(f"Phase {SignalPhase.get_name(proposed_phase)} blocked by safety constraints")
            # Find alternative phase
            for alt_phase in range(5):
                if alt_phase not in blocked_phases and alt_phase != self.current_phase:
                    proposed_phase = alt_phase
                    break
            else:
                # Stay in current phase if no alternatives
                return self.current_phase
        
        # 6. Check maximum phase duration
        if phase_duration >= self.max_phase_duration and proposed_phase == self.current_phase:
            # Force phase change
            for alt_phase in range(5):
                if alt_phase not in blocked_phases and alt_phase != self.current_phase:
                    proposed_phase = alt_phase
                    break
        
        return proposed_phase
    
    def apply_phase(self, phase: int, timestamp: Optional[float] = None):
        """
        Apply a signal phase change.
        
        Args:
            phase: Phase to apply
            timestamp: Timestamp of change
        """
        if timestamp is None:
            timestamp = time.time()
        
        if phase != self.current_phase:
            old_phase = self.current_phase
            self.current_phase = phase
            self.phase_start_time = timestamp
            
            logger.info(
                f"Phase change: {SignalPhase.get_name(old_phase)} -> "
                f"{SignalPhase.get_name(phase)}"
            )
            
            # Update crosswalk phases
            if phase == SignalPhase.PEDESTRIAN:
                for cid in self.safety_manager.crosswalks:
                    self.safety_manager.set_crosswalk_phase(cid, CrosswalkPhase.WALK, timestamp)
            else:
                for cid in self.safety_manager.crosswalks:
                    self.safety_manager.set_crosswalk_phase(cid, CrosswalkPhase.DONT_WALK, timestamp)
    
    def step(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute one control step: process frame and update signal.
        
        Args:
            frame: Camera frame
            timestamp: Current timestamp
            
        Returns:
            Step results including selected phase
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Process frame
        result = self.process_frame(frame, timestamp)
        
        # Select phase
        state = result["rl_state"]
        selected_phase = self.select_phase(state, timestamp)
        
        # Apply phase
        self.apply_phase(selected_phase, timestamp)
        
        # Record metrics
        self._record_step_metrics(result, timestamp)
        
        result["selected_phase"] = SignalPhase.get_name(selected_phase)
        result["phase_changed"] = selected_phase != self.current_phase
        
        return result
    
    def _record_step_metrics(self, result: Dict, timestamp: float):
        """Record step metrics"""
        # Queue lengths
        for lane_id, state_dict in result["lane_states"].items():
            self.metrics.record_queue_length(state_dict["queue_length"], int(lane_id))
            self.metrics.record_waiting_time(state_dict["average_waiting_time"], "vehicle", int(lane_id))
        
        # Pedestrian waiting
        for cw_id, state_dict in result["crosswalk_states"].items():
            if state_dict.get("pedestrian_count", 0) > 0:
                # Estimate waiting time
                wait_time = 0  # Would be calculated from actual tracking
                self.metrics.record_waiting_time(wait_time, "pedestrian")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current controller status"""
        return {
            "current_phase": SignalPhase.get_name(self.current_phase),
            "phase_duration": time.time() - self.phase_start_time,
            "frame_count": self.frame_count,
            "emergency_active": self.emergency_handler.is_preemption_active(),
            "queue_summary": self.queue_estimator.get_summary(),
            "safety_summary": self.safety_manager.get_summary(),
            "metrics": self.metrics.get_evaluation_metrics(),
            "tracking_stats": self.tracker.get_statistics(),
        }
    
    def reset(self):
        """Reset controller state"""
        self.tracker.reset()
        self.queue_estimator.reset()
        self.safety_manager.reset()
        self.emergency_handler.reset()
        self.metrics.reset()
        
        self.current_phase = SignalPhase.NS_GREEN
        self.phase_start_time = time.time()
        self.frame_count = 0
        
        logger.info("Controller reset")
