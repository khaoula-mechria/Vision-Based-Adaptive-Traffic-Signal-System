"""
SUMO Traffic Simulation Interface
Connects the RL agent to SUMO for training and validation
"""

import os
import sys
from typing import Dict, List, Optional, Tuple
import logging

# Try to import SUMO libraries
try:
    import traci
    import sumolib
    SUMO_AVAILABLE = True
except ImportError:
    traci = None
    sumolib = None
    SUMO_AVAILABLE = False

logger = logging.getLogger(__name__)


class SUMOInterface:
    """
    Interface for SUMO (Simulation of Urban MObility) traffic simulator.
    
    Provides:
    - Connection management to SUMO
    - Signal phase control
    - Traffic state extraction
    - Metrics collection
    
    Used for:
    - Training RL agents in simulation
    - Validating performance before deployment
    - Comparing against baseline control methods
    """
    
    # Signal phase definitions (must match network file)
    PHASE_MAP = {
        0: "GGrrGGrr",  # NS Green
        1: "rrGGrrGG",  # EW Green
        2: "GrrrGrrr",  # NS Left
        3: "rrrGrrrG",  # EW Left
        4: "rrrrrrrr",  # All Red (pedestrian)
    }
    
    def __init__(
        self,
        network_file: str,
        route_file: str,
        sumo_binary: str = "sumo",
        step_length: float = 1.0,
        gui: bool = False,
        seed: int = 42,
        port: Optional[int] = None,
    ):
        """
        Initialize SUMO interface.
        
        Args:
            network_file: Path to SUMO network file (.net.xml)
            route_file: Path to SUMO route file (.rou.xml)
            sumo_binary: SUMO binary name ('sumo' or 'sumo-gui')
            step_length: Simulation step length in seconds
            gui: Whether to use GUI version
            seed: Random seed for SUMO
            port: TraCI port (auto-assigned if None)
        """
        if not SUMO_AVAILABLE:
            logger.warning("SUMO (traci/sumolib) not available. Install with: pip install traci sumolib")
        
        self.network_file = network_file
        self.route_file = route_file
        self.sumo_binary = "sumo-gui" if gui else "sumo"
        self.step_length = step_length
        self.seed = seed
        self.port = port
        
        # State tracking
        self.is_running = False
        self.current_step = 0
        self.current_phase = 0
        
        # Traffic light ID (assumes single intersection)
        self.tl_id = "center"
        
        # Metrics
        self.total_waiting_time = 0.0
        self.total_departed = 0
        self.total_arrived = 0
        self.emergency_vehicles_passed = 0
    
    def start(self):
        """Start SUMO simulation"""
        if not SUMO_AVAILABLE:
            raise RuntimeError("SUMO is not available")
        
        if self.is_running:
            self.close()
        
        # Build SUMO command
        sumo_cmd = [
            self.sumo_binary,
            "-n", self.network_file,
            "-r", self.route_file,
            "--step-length", str(self.step_length),
            "--seed", str(self.seed),
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--no-warnings", "true",
        ]
        
        if self.port:
            sumo_cmd.extend(["--remote-port", str(self.port)])
        
        # Start SUMO
        try:
            traci.start(sumo_cmd)
            self.is_running = True
            self.current_step = 0
            logger.info("SUMO simulation started")
        except Exception as e:
            logger.error(f"Failed to start SUMO: {e}")
            raise
    
    def step(self, phase: Optional[int] = None) -> Dict:
        """
        Advance simulation by one step.
        
        Args:
            phase: Signal phase to set (None to keep current)
            
        Returns:
            Dictionary with current traffic state
        """
        if not self.is_running:
            return self._get_empty_state()
        
        # Set signal phase if specified
        if phase is not None:
            self.set_phase(phase)
        
        # Advance simulation
        traci.simulationStep()
        self.current_step += 1
        
        # Collect state
        state = self._collect_state()
        
        # Update metrics
        self._update_metrics(state)
        
        return state
    
    def set_phase(self, phase: int):
        """
        Set traffic light phase.
        
        Args:
            phase: Phase index (0-4)
        """
        if not self.is_running:
            return
        
        phase = int(phase)
        if phase < 0 or phase >= len(self.PHASE_MAP):
            logger.warning(f"Invalid phase {phase}, using 0")
            phase = 0
        
        try:
            # Set the signal state directly
            state = self.PHASE_MAP[phase]
            traci.trafficlight.setRedYellowGreenState(self.tl_id, state)
            self.current_phase = phase
        except Exception as e:
            logger.error(f"Failed to set phase: {e}")
    
    def _collect_state(self) -> Dict:
        """Collect current traffic state from SUMO"""
        state = {
            "simulation_time": traci.simulation.getTime(),
            "queue_lengths": [],
            "waiting_times": [],
            "pedestrian_counts": [],
            "pedestrian_waiting": [],
            "emergency_detected": False,
            "emergency_direction": -1,
            "vehicles_passed": 0,
        }
        
        # Get lane data
        try:
            # Get all lanes
            lane_ids = traci.lane.getIDList()
            
            for lane_id in lane_ids:
                if "in" in lane_id:  # Incoming lanes only
                    # Queue length (halting vehicles)
                    halting = traci.lane.getLastStepHaltingNumber(lane_id)
                    state["queue_lengths"].append(halting)
                    
                    # Waiting time
                    waiting = traci.lane.getWaitingTime(lane_id)
                    state["waiting_times"].append(waiting)
            
            # Pad to expected size if needed
            while len(state["queue_lengths"]) < 8:
                state["queue_lengths"].append(0)
                state["waiting_times"].append(0.0)
            
            # Get vehicle data for emergency detection
            vehicle_ids = traci.vehicle.getIDList()
            for veh_id in vehicle_ids:
                vtype = traci.vehicle.getTypeID(veh_id)
                if vtype in ["ambulance", "fire_truck", "police"]:
                    state["emergency_detected"] = True
                    # Determine direction from route
                    route = traci.vehicle.getRoute(veh_id)
                    if route:
                        if "north" in route[0]:
                            state["emergency_direction"] = 0
                        elif "south" in route[0]:
                            state["emergency_direction"] = 1
                        elif "east" in route[0]:
                            state["emergency_direction"] = 2
                        elif "west" in route[0]:
                            state["emergency_direction"] = 3
            
            # Vehicles that arrived this step
            state["vehicles_passed"] = traci.simulation.getArrivedNumber()
            
            # Pedestrian data (if available)
            try:
                person_ids = traci.person.getIDList()
                state["pedestrian_counts"] = [len(person_ids) // 4] * 4  # Distribute
                state["pedestrian_waiting"] = [0.0] * 4
                
                for person_id in person_ids:
                    waiting = traci.person.getWaitingTime(person_id)
                    # Assign to crosswalk based on position
                    pos = traci.person.getPosition(person_id)
                    crosswalk = self._position_to_crosswalk(pos)
                    if 0 <= crosswalk < 4:
                        state["pedestrian_waiting"][crosswalk] += waiting
            except Exception:
                state["pedestrian_counts"] = [0, 0, 0, 0]
                state["pedestrian_waiting"] = [0.0, 0.0, 0.0, 0.0]
        
        except Exception as e:
            logger.error(f"Error collecting state: {e}")
        
        return state
    
    def _position_to_crosswalk(self, pos: Tuple[float, float]) -> int:
        """Map position to crosswalk index"""
        x, y = pos
        if y > 10:
            return 0  # North
        elif y < -10:
            return 1  # South
        elif x > 10:
            return 2  # East
        elif x < -10:
            return 3  # West
        return -1
    
    def _update_metrics(self, state: Dict):
        """Update running metrics"""
        self.total_waiting_time += sum(state.get("waiting_times", []))
        self.total_arrived += state.get("vehicles_passed", 0)
        
        if state.get("emergency_detected"):
            # Track emergency vehicles
            pass
    
    def _get_empty_state(self) -> Dict:
        """Return empty state dictionary"""
        return {
            "simulation_time": 0,
            "queue_lengths": [0] * 8,
            "waiting_times": [0.0] * 8,
            "pedestrian_counts": [0] * 4,
            "pedestrian_waiting": [0.0] * 4,
            "emergency_detected": False,
            "emergency_direction": -1,
            "vehicles_passed": 0,
        }
    
    def reset(self) -> Dict:
        """Reset simulation"""
        if self.is_running:
            self.close()
        
        # Reset metrics
        self.total_waiting_time = 0.0
        self.total_departed = 0
        self.total_arrived = 0
        self.emergency_vehicles_passed = 0
        
        # Restart
        self.start()
        
        return self._collect_state() if self.is_running else self._get_empty_state()
    
    def close(self):
        """Close SUMO connection"""
        if self.is_running and SUMO_AVAILABLE:
            try:
                traci.close()
                self.is_running = False
                logger.info("SUMO simulation closed")
            except Exception as e:
                logger.error(f"Error closing SUMO: {e}")
    
    def get_metrics(self) -> Dict:
        """Get collected metrics"""
        return {
            "current_step": self.current_step,
            "simulation_time": self.current_step * self.step_length,
            "total_waiting_time": self.total_waiting_time,
            "total_departed": self.total_departed,
            "total_arrived": self.total_arrived,
            "average_waiting_time": self.total_waiting_time / max(1, self.current_step),
            "throughput": self.total_arrived / max(1, self.current_step * self.step_length) * 3600,
        }
    
    def run_fixed_time_baseline(
        self,
        duration: int = 3600,
        ns_green: int = 30,
        ew_green: int = 30,
        yellow: int = 5,
    ) -> Dict:
        """
        Run simulation with fixed-time signal control.
        
        Args:
            duration: Simulation duration in seconds
            ns_green: North-South green duration
            ew_green: East-West green duration
            yellow: Yellow phase duration
            
        Returns:
            Performance metrics
        """
        self.reset()
        
        cycle_length = ns_green + yellow + ew_green + yellow
        metrics_history = []
        
        for step in range(int(duration / self.step_length)):
            time_in_cycle = (step * self.step_length) % cycle_length
            
            # Determine phase
            if time_in_cycle < ns_green:
                phase = 0  # NS Green
            elif time_in_cycle < ns_green + yellow:
                phase = 0  # Yellow (simplified)
            elif time_in_cycle < ns_green + yellow + ew_green:
                phase = 1  # EW Green
            else:
                phase = 1  # Yellow (simplified)
            
            state = self.step(phase)
            metrics_history.append(state)
        
        final_metrics = self.get_metrics()
        final_metrics["control_type"] = "fixed_time"
        
        self.close()
        return final_metrics
    
    def run_actuated_baseline(
        self,
        duration: int = 3600,
        min_green: int = 10,
        max_green: int = 60,
        extension: int = 3,
    ) -> Dict:
        """
        Run simulation with actuated signal control.
        
        Extends green phase based on vehicle presence.
        
        Args:
            duration: Simulation duration in seconds
            min_green: Minimum green duration
            max_green: Maximum green duration
            extension: Extension per vehicle detection
            
        Returns:
            Performance metrics
        """
        self.reset()
        
        current_phase = 0
        phase_time = 0
        
        for step in range(int(duration / self.step_length)):
            state = self.step(current_phase)
            phase_time += self.step_length
            
            # Get queue for current direction
            if current_phase == 0:  # NS
                queue = sum(state["queue_lengths"][:4])
            else:  # EW
                queue = sum(state["queue_lengths"][4:])
            
            # Check phase change
            should_extend = queue > 0 and phase_time < max_green
            should_change = phase_time >= min_green and (not should_extend or phase_time >= max_green)
            
            if should_change:
                current_phase = 1 - current_phase  # Toggle
                phase_time = 0
        
        final_metrics = self.get_metrics()
        final_metrics["control_type"] = "actuated"
        
        self.close()
        return final_metrics
    
    @staticmethod
    def check_sumo_installation() -> bool:
        """Check if SUMO is properly installed"""
        if not SUMO_AVAILABLE:
            return False
        
        try:
            sumo_home = os.environ.get("SUMO_HOME")
            return sumo_home is not None and os.path.exists(sumo_home)
        except Exception:
            return False
