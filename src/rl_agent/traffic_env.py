"""
Traffic Signal Environment for Reinforcement Learning
Compatible with Gymnasium/OpenAI Gym interface
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None

from .reward import RewardCalculator, RewardWeights


class TrafficSignalEnv:
    """
    Gymnasium-compatible environment for traffic signal control.
    
    The agent learns to optimize signal timing by observing:
    - Lane-level traffic states
    - Pedestrian density
    - Emergency vehicle presence
    - Current signal phase
    
    Actions are signal phase selections with duration.
    """
    
    # Signal phases
    PHASES = [
        "NS_Green",      # North-South green
        "EW_Green",      # East-West green
        "NS_Left",       # North-South left turn
        "EW_Left",       # East-West left turn
        "Pedestrian",    # All-way pedestrian
    ]
    
    def __init__(
        self,
        num_lanes: int = 8,
        num_crosswalks: int = 4,
        min_phase_duration: int = 10,
        max_phase_duration: int = 60,
        simulation_step: float = 1.0,
        max_steps: int = 500,
        reward_weights: Optional[RewardWeights] = None,
    ):
        """
        Initialize the traffic signal environment.
        
        Args:
            num_lanes: Number of traffic lanes
            num_crosswalks: Number of pedestrian crosswalks
            min_phase_duration: Minimum phase duration in seconds
            max_phase_duration: Maximum phase duration in seconds
            simulation_step: Simulation time step in seconds
            max_steps: Maximum steps per episode
            reward_weights: Custom reward weights
        """
        self.num_lanes = num_lanes
        self.num_crosswalks = num_crosswalks
        self.min_phase_duration = min_phase_duration
        self.max_phase_duration = max_phase_duration
        self.simulation_step = simulation_step
        self.max_steps = max_steps
        
        # Reward calculator
        self.reward_calculator = RewardCalculator(weights=reward_weights)
        
        # State variables
        self.current_step = 0
        self.current_phase = 0
        self.phase_time = 0.0
        self.total_time = 0.0
        
        # Traffic state
        self.queue_lengths = np.zeros(num_lanes)
        self.waiting_times = np.zeros(num_lanes)
        self.pedestrian_counts = np.zeros(num_crosswalks)
        self.pedestrian_waiting = np.zeros(num_crosswalks)
        self.emergency_detected = False
        self.emergency_direction = -1
        
        # Statistics
        self.vehicles_passed = 0
        self.total_vehicle_wait = 0.0
        self.total_pedestrian_wait = 0.0
        
        # Define spaces if gymnasium is available
        if spaces is not None:
            self._define_spaces()
        
        # External connections (SUMO, perception, etc.)
        self._simulation = None
        self._perception = None
    
    def _define_spaces(self):
        """Define observation and action spaces"""
        # Observation space
        # [queue_lengths (8), waiting_times (8), ped_counts (4), ped_waiting (4),
        #  emergency (1), emergency_dir (1), current_phase (1), phase_time (1)]
        obs_dim = self.num_lanes * 2 + self.num_crosswalks * 2 + 4
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # Action space: phase selection (discrete)
        # Each phase can be selected; duration is determined by policy or fixed
        self.action_space = spaces.Discrete(len(self.PHASES))
    
    def connect_simulation(self, simulation):
        """Connect to external simulation (e.g., SUMO)"""
        self._simulation = simulation
    
    def connect_perception(self, perception):
        """Connect to perception system for real-world deployment"""
        self._perception = perception
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset state
        self.current_step = 0
        self.current_phase = 0
        self.phase_time = 0.0
        self.total_time = 0.0
        
        # Initialize traffic state with some randomness
        self.queue_lengths = np.random.randint(0, 10, self.num_lanes).astype(np.float32)
        self.waiting_times = np.random.uniform(0, 30, self.num_lanes).astype(np.float32)
        self.pedestrian_counts = np.random.randint(0, 5, self.num_crosswalks).astype(np.float32)
        self.pedestrian_waiting = np.random.uniform(0, 20, self.num_crosswalks).astype(np.float32)
        self.emergency_detected = False
        self.emergency_direction = -1
        
        # Reset statistics
        self.vehicles_passed = 0
        self.total_vehicle_wait = 0.0
        self.total_pedestrian_wait = 0.0
        
        # Reset simulation if connected
        if self._simulation:
            self._simulation.reset()
        
        # Reset reward normalization
        self.reward_calculator.reset_normalization()
        
        return self._get_observation(), self._get_info()
    
    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Phase to activate (0-4)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Store previous state
        prev_state = self._get_state_dict()
        
        # Validate action
        action = int(action)
        if action < 0 or action >= len(self.PHASES):
            action = 0
        
        # Check for phase change
        phase_changed = (action != self.current_phase)
        
        # Apply action
        if phase_changed:
            self.current_phase = action
            self.phase_time = 0.0
        
        # Simulate traffic for one step
        self._simulate_step()
        
        # Update time
        self.phase_time += self.simulation_step
        self.total_time += self.simulation_step
        self.current_step += 1
        
        # Get new state
        next_state = self._get_state_dict()
        
        # Calculate info
        info = self._get_info()
        info["phase_changed"] = phase_changed
        info["smooth_transition"] = not phase_changed or self.phase_time >= self.min_phase_duration
        
        # Check safety violations
        info["safety_violation"] = self._check_safety_violation()
        
        # Check emergency delay
        info["emergency_delayed"] = self.emergency_detected and action not in self._get_emergency_phases()
        
        # Calculate reward
        reward = self.reward_calculator.calculate(prev_state, action, next_state, info)
        
        # Check termination
        terminated = False  # Environment doesn't naturally terminate
        truncated = self.current_step >= self.max_steps
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _simulate_step(self):
        """Simulate one time step of traffic"""
        if self._simulation:
            # Use external simulation
            state = self._simulation.step(self.current_phase)
            self._update_from_simulation(state)
        else:
            # Use internal simulation model
            self._internal_simulate()
    
    def _internal_simulate(self):
        """Internal traffic simulation (simplified model)"""
        phase = self.PHASES[self.current_phase]
        
        # Determine which lanes get green
        green_lanes = self._get_green_lanes(phase)
        
        for lane in range(self.num_lanes):
            if lane in green_lanes:
                # Green light: vehicles depart
                departures = min(self.queue_lengths[lane], np.random.poisson(3))
                self.queue_lengths[lane] -= departures
                self.vehicles_passed += departures
                self.waiting_times[lane] = max(0, self.waiting_times[lane] - self.simulation_step)
            else:
                # Red light: vehicles arrive and wait
                arrivals = np.random.poisson(2)
                self.queue_lengths[lane] += arrivals
                self.waiting_times[lane] += self.simulation_step
        
        # Pedestrian simulation
        for cw in range(self.num_crosswalks):
            if phase == "Pedestrian" or self._crosswalk_has_green(cw, phase):
                # Pedestrians can cross
                crossings = min(self.pedestrian_counts[cw], np.random.poisson(2))
                self.pedestrian_counts[cw] -= crossings
                self.pedestrian_waiting[cw] = max(0, self.pedestrian_waiting[cw] - self.simulation_step)
            else:
                # Pedestrians arrive and wait
                arrivals = np.random.poisson(1)
                self.pedestrian_counts[cw] += arrivals
                self.pedestrian_waiting[cw] += self.simulation_step
        
        # Emergency vehicle simulation (rare)
        if not self.emergency_detected and np.random.random() < 0.01:
            self.emergency_detected = True
            self.emergency_direction = np.random.randint(0, 4)
        elif self.emergency_detected and np.random.random() < 0.1:
            self.emergency_detected = False
            self.emergency_direction = -1
        
        # Update totals
        self.total_vehicle_wait += np.sum(self.waiting_times)
        self.total_pedestrian_wait += np.sum(self.pedestrian_waiting)
    
    def _get_green_lanes(self, phase: str) -> List[int]:
        """Get lanes that have green light for given phase"""
        # Simplified: 0-3 are NS lanes, 4-7 are EW lanes
        if phase == "NS_Green":
            return [0, 1, 2, 3]
        elif phase == "EW_Green":
            return [4, 5, 6, 7]
        elif phase == "NS_Left":
            return [0, 2]  # Left turn lanes
        elif phase == "EW_Left":
            return [4, 6]
        elif phase == "Pedestrian":
            return []  # All vehicles stop
        return []
    
    def _crosswalk_has_green(self, crosswalk: int, phase: str) -> bool:
        """Check if crosswalk can have pedestrians cross during phase"""
        # 0-1 are NS crosswalks (cross EW traffic)
        # 2-3 are EW crosswalks (cross NS traffic)
        if crosswalk in [0, 1]:
            return phase in ["NS_Green", "NS_Left"]
        else:
            return phase in ["EW_Green", "EW_Left"]
    
    def _get_emergency_phases(self) -> List[int]:
        """Get phases that allow emergency passage"""
        if self.emergency_direction in [0, 1]:  # NS
            return [0, 2]  # NS_Green, NS_Left
        elif self.emergency_direction in [2, 3]:  # EW
            return [1, 3]  # EW_Green, EW_Left
        return list(range(len(self.PHASES)))
    
    def _check_safety_violation(self) -> bool:
        """Check for safety violations"""
        # Check if phase changed while pedestrians in crosswalk
        phase = self.PHASES[self.current_phase]
        if phase != "Pedestrian":
            # Check relevant crosswalks
            if phase in ["NS_Green", "NS_Left"]:
                # Check EW crosswalks
                if np.sum(self.pedestrian_counts[2:4]) > 0 and self.pedestrian_waiting[2:4].max() < 5:
                    return True  # Pedestrians may be crossing
            elif phase in ["EW_Green", "EW_Left"]:
                # Check NS crosswalks
                if np.sum(self.pedestrian_counts[0:2]) > 0 and self.pedestrian_waiting[0:2].max() < 5:
                    return True
        return False
    
    def _update_from_simulation(self, state: Dict):
        """Update environment state from external simulation"""
        self.queue_lengths = np.array(state.get("queue_lengths", self.queue_lengths))
        self.waiting_times = np.array(state.get("waiting_times", self.waiting_times))
        self.pedestrian_counts = np.array(state.get("pedestrian_counts", self.pedestrian_counts))
        self.pedestrian_waiting = np.array(state.get("pedestrian_waiting", self.pedestrian_waiting))
        self.emergency_detected = state.get("emergency_detected", self.emergency_detected)
        self.emergency_direction = state.get("emergency_direction", self.emergency_direction)
        self.vehicles_passed += state.get("vehicles_passed", 0)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector"""
        obs = np.concatenate([
            self.queue_lengths,
            self.waiting_times,
            self.pedestrian_counts,
            self.pedestrian_waiting,
            [float(self.emergency_detected)],
            [float(self.emergency_direction)],
            [float(self.current_phase)],
            [self.phase_time],
        ])
        return obs.astype(np.float32)
    
    def _get_state_dict(self) -> Dict:
        """Get current state as dictionary"""
        return {
            "queue_lengths": self.queue_lengths.copy(),
            "waiting_times": self.waiting_times.copy(),
            "pedestrian_counts": self.pedestrian_counts.copy(),
            "pedestrian_waiting_times": self.pedestrian_waiting.copy(),
            "emergency_detected": self.emergency_detected,
            "emergency_direction": self.emergency_direction,
            "current_phase": self.current_phase,
            "phase_time": self.phase_time,
            "vehicles_passed": self.vehicles_passed,
        }
    
    def _get_info(self) -> Dict:
        """Get additional information"""
        return {
            "step": self.current_step,
            "total_time": self.total_time,
            "current_phase": self.PHASES[self.current_phase],
            "phase_time": self.phase_time,
            "total_queue": np.sum(self.queue_lengths),
            "avg_waiting_time": np.mean(self.waiting_times),
            "max_waiting_time": np.max(self.waiting_times),
            "total_pedestrians": np.sum(self.pedestrian_counts),
            "vehicles_passed": self.vehicles_passed,
            "emergency_active": self.emergency_detected,
        }
    
    def render(self, mode: str = "human"):
        """Render the environment (placeholder)"""
        info = self._get_info()
        print(f"Step {info['step']}: Phase={info['current_phase']}, "
              f"Queue={info['total_queue']:.0f}, "
              f"AvgWait={info['avg_waiting_time']:.1f}s, "
              f"Passed={info['vehicles_passed']}")
    
    def close(self):
        """Clean up resources"""
        if self._simulation:
            self._simulation.close()


# Register with gymnasium if available
if gym is not None:
    try:
        gym.register(
            id="TrafficSignal-v0",
            entry_point="src.rl_agent.traffic_env:TrafficSignalEnv",
        )
    except Exception:
        pass  # Already registered or registration failed
