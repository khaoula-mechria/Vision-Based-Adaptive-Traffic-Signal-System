"""
Reward Calculator for Traffic Signal RL Agent
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class RewardWeights:
    """Weights for different reward components"""
    vehicle_waiting_time: float = -1.0  # Penalty for vehicle waiting
    pedestrian_waiting_time: float = -1.5  # Higher penalty for pedestrian waiting
    queue_length: float = -0.5  # Penalty for queue buildup
    throughput: float = 1.0  # Reward for vehicles passing
    safety_violation: float = -100.0  # Large penalty for safety violations
    emergency_delay: float = -50.0  # Penalty for delaying emergency vehicles
    smooth_transition: float = 0.1  # Small reward for smooth transitions


class RewardCalculator:
    """
    Calculates reward for the traffic signal RL agent.
    
    The reward function balances:
    - Reduced vehicle waiting time
    - Reduced pedestrian delay
    - Congestion avoidance
    - Safety compliance (highest priority)
    - Emergency vehicle clearance
    """
    
    def __init__(
        self,
        weights: Optional[RewardWeights] = None,
        normalize: bool = True,
    ):
        """
        Initialize reward calculator.
        
        Args:
            weights: Reward component weights
            normalize: Whether to normalize reward
        """
        self.weights = weights or RewardWeights()
        self.normalize = normalize
        
        # Running statistics for normalization
        self._reward_mean = 0.0
        self._reward_std = 1.0
        self._reward_count = 0
    
    def calculate(
        self,
        state: Dict,
        action: int,
        next_state: Dict,
        info: Dict,
    ) -> float:
        """
        Calculate total reward.
        
        Args:
            state: Current state before action
            action: Action taken
            next_state: State after action
            info: Additional information
            
        Returns:
            Total reward value
        """
        reward = 0.0
        
        # Vehicle waiting time penalty
        vehicle_wait = self._calculate_vehicle_waiting_penalty(state, next_state)
        reward += self.weights.vehicle_waiting_time * vehicle_wait
        
        # Pedestrian waiting time penalty
        ped_wait = self._calculate_pedestrian_waiting_penalty(state, next_state)
        reward += self.weights.pedestrian_waiting_time * ped_wait
        
        # Queue length penalty
        queue_penalty = self._calculate_queue_penalty(next_state)
        reward += self.weights.queue_length * queue_penalty
        
        # Throughput reward
        throughput = self._calculate_throughput_reward(state, next_state)
        reward += self.weights.throughput * throughput
        
        # Safety violation penalty
        if info.get("safety_violation", False):
            reward += self.weights.safety_violation
        
        # Emergency delay penalty
        if info.get("emergency_delayed", False):
            emergency_delay_time = info.get("emergency_delay_time", 0)
            reward += self.weights.emergency_delay * (1 + emergency_delay_time / 10.0)
        
        # Smooth transition bonus
        if info.get("smooth_transition", True):
            reward += self.weights.smooth_transition
        
        # Normalize if enabled
        if self.normalize:
            reward = self._normalize_reward(reward)
        
        return reward
    
    def _calculate_vehicle_waiting_penalty(
        self,
        state: Dict,
        next_state: Dict,
    ) -> float:
        """Calculate penalty based on vehicle waiting times"""
        if "waiting_times" not in next_state:
            return 0.0
        
        waiting_times = next_state["waiting_times"]
        if not isinstance(waiting_times, (list, np.ndarray)):
            waiting_times = [waiting_times]
        
        # Penalty increases with waiting time
        if len(waiting_times) == 0:
            return 0.0
        
        avg_wait = np.mean(waiting_times)
        max_wait = np.max(waiting_times)
        
        # Combined penalty with emphasis on maximum wait
        return avg_wait + 0.5 * max_wait
    
    def _calculate_pedestrian_waiting_penalty(
        self,
        state: Dict,
        next_state: Dict,
    ) -> float:
        """Calculate penalty based on pedestrian waiting times"""
        if "pedestrian_waiting_times" not in next_state:
            return 0.0
        
        waiting_times = next_state["pedestrian_waiting_times"]
        if not isinstance(waiting_times, (list, np.ndarray)):
            waiting_times = [waiting_times]
        
        if len(waiting_times) == 0:
            return 0.0
        
        # Pedestrians are more sensitive to waiting
        avg_wait = np.mean(waiting_times)
        return avg_wait * 1.2  # 20% higher sensitivity
    
    def _calculate_queue_penalty(self, next_state: Dict) -> float:
        """Calculate penalty based on queue lengths"""
        if "queue_lengths" not in next_state:
            return 0.0
        
        queue_lengths = next_state["queue_lengths"]
        if not isinstance(queue_lengths, (list, np.ndarray)):
            queue_lengths = [queue_lengths]
        
        if len(queue_lengths) == 0:
            return 0.0
        
        # Penalty for total queue and imbalance
        total_queue = np.sum(queue_lengths)
        queue_std = np.std(queue_lengths)  # Imbalance penalty
        
        return total_queue + 2 * queue_std
    
    def _calculate_throughput_reward(
        self,
        state: Dict,
        next_state: Dict,
    ) -> float:
        """Calculate reward based on vehicle throughput"""
        vehicles_passed = next_state.get("vehicles_passed", 0)
        return float(vehicles_passed)
    
    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward using running statistics"""
        # Update running statistics
        self._reward_count += 1
        delta = reward - self._reward_mean
        self._reward_mean += delta / self._reward_count
        delta2 = reward - self._reward_mean
        self._reward_std = np.sqrt(
            (self._reward_std**2 * (self._reward_count - 1) + delta * delta2) 
            / self._reward_count
        )
        
        # Normalize
        if self._reward_std > 1e-6:
            return (reward - self._reward_mean) / self._reward_std
        return reward - self._reward_mean
    
    def get_reward_breakdown(
        self,
        state: Dict,
        action: int,
        next_state: Dict,
        info: Dict,
    ) -> Dict[str, float]:
        """
        Get detailed breakdown of reward components.
        
        Useful for debugging and understanding agent behavior.
        """
        breakdown = {}
        
        vehicle_wait = self._calculate_vehicle_waiting_penalty(state, next_state)
        breakdown["vehicle_waiting"] = self.weights.vehicle_waiting_time * vehicle_wait
        
        ped_wait = self._calculate_pedestrian_waiting_penalty(state, next_state)
        breakdown["pedestrian_waiting"] = self.weights.pedestrian_waiting_time * ped_wait
        
        queue_penalty = self._calculate_queue_penalty(next_state)
        breakdown["queue_length"] = self.weights.queue_length * queue_penalty
        
        throughput = self._calculate_throughput_reward(state, next_state)
        breakdown["throughput"] = self.weights.throughput * throughput
        
        if info.get("safety_violation", False):
            breakdown["safety_violation"] = self.weights.safety_violation
        
        if info.get("emergency_delayed", False):
            breakdown["emergency_delay"] = self.weights.emergency_delay
        
        breakdown["total"] = sum(breakdown.values())
        
        return breakdown
    
    def reset_normalization(self):
        """Reset normalization statistics"""
        self._reward_mean = 0.0
        self._reward_std = 1.0
        self._reward_count = 0
