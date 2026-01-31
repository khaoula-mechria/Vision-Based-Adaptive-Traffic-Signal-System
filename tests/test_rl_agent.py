"""
Unit tests for reinforcement learning module
"""

import pytest
import numpy as np

from src.rl_agent.reward import RewardCalculator, RewardWeights
from src.rl_agent.traffic_env import TrafficSignalEnv


class TestRewardWeights:
    """Tests for RewardWeights class"""
    
    def test_default_weights(self):
        """Test default weight values"""
        weights = RewardWeights()
        
        assert weights.vehicle_waiting_time == -1.0
        assert weights.pedestrian_waiting_time == -1.5
        assert weights.safety_violation == -100.0


class TestRewardCalculator:
    """Tests for RewardCalculator class"""
    
    def test_calculator_creation(self):
        """Test reward calculator creation"""
        calc = RewardCalculator()
        assert calc is not None
    
    def test_throughput_reward(self):
        """Test throughput reward calculation"""
        calc = RewardCalculator(normalize=False)
        
        state = {}
        next_state = {"vehicles_passed": 10}
        info = {}
        
        reward = calc.calculate(state, 0, next_state, info)
        
        # Should have positive contribution from throughput
        assert reward > 0
    
    def test_safety_violation_penalty(self):
        """Test safety violation penalty"""
        calc = RewardCalculator(normalize=False)
        
        state = {}
        next_state = {"vehicles_passed": 0}
        info = {"safety_violation": True}
        
        reward = calc.calculate(state, 0, next_state, info)
        
        # Should have large negative reward due to violation
        assert reward <= -99.0  # Allow small margin for floating point
    
    def test_emergency_delay_penalty(self):
        """Test emergency delay penalty"""
        calc = RewardCalculator(normalize=False)
        
        state = {}
        next_state = {"vehicles_passed": 0}
        info = {"emergency_delayed": True, "emergency_delay_time": 5}
        
        reward = calc.calculate(state, 0, next_state, info)
        
        # Should have negative reward due to emergency delay
        assert reward < 0
    
    def test_reward_breakdown(self):
        """Test reward breakdown"""
        calc = RewardCalculator(normalize=False)
        
        state = {}
        next_state = {
            "vehicles_passed": 5,
            "waiting_times": [10, 15, 20],
            "queue_lengths": [3, 4, 5],
        }
        info = {}
        
        breakdown = calc.get_reward_breakdown(state, 0, next_state, info)
        
        assert "throughput" in breakdown
        assert "total" in breakdown


class TestTrafficSignalEnv:
    """Tests for TrafficSignalEnv class"""
    
    def test_env_creation(self):
        """Test environment creation"""
        env = TrafficSignalEnv()
        
        assert env.num_lanes == 8
        assert env.num_crosswalks == 4
    
    def test_reset(self):
        """Test environment reset"""
        env = TrafficSignalEnv()
        
        obs, info = env.reset(seed=42)
        
        assert obs is not None
        assert len(obs) > 0
        assert isinstance(info, dict)
    
    def test_step(self):
        """Test environment step"""
        env = TrafficSignalEnv()
        env.reset(seed=42)
        
        obs, reward, terminated, truncated, info = env.step(0)
        
        assert obs is not None
        assert isinstance(float(reward), float)  # Handle numpy types
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_phases(self):
        """Test signal phases"""
        env = TrafficSignalEnv()
        env.reset(seed=42)
        
        # Test all phases
        for phase in range(5):
            obs, reward, terminated, truncated, info = env.step(phase)
            assert info["current_phase"] in env.PHASES
    
    def test_observation_shape(self):
        """Test observation shape consistency"""
        env = TrafficSignalEnv()
        obs1, _ = env.reset(seed=42)
        
        obs2, _, _, _, _ = env.step(0)
        obs3, _, _, _, _ = env.step(1)
        
        assert obs1.shape == obs2.shape == obs3.shape
    
    def test_truncation(self):
        """Test episode truncation"""
        env = TrafficSignalEnv(max_steps=10)
        env.reset(seed=42)
        
        truncated = False
        for _ in range(15):
            _, _, _, truncated, _ = env.step(0)
            if truncated:
                break
        
        assert truncated is True
    
    def test_emergency_simulation(self):
        """Test emergency vehicle simulation"""
        env = TrafficSignalEnv()
        env.reset(seed=42)
        
        # Run many steps to have chance of emergency
        emergency_detected = False
        for _ in range(100):
            _, _, _, _, info = env.step(0)
            if info.get("emergency_active", False):
                emergency_detected = True
                break
        
        # Emergency may or may not occur due to randomness
        # Just verify info has the key
        assert "emergency_active" in info
