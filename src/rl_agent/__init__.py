"""Reinforcement learning: environment, observation encoding, reward, DQN agent."""

from .observation import ObservationEncoder
from .reward import IntervalStats, RewardCalculator, RewardWeights
from .traffic_env import TrafficSignalEnv

__all__ = ["TrafficSignalEnv", "ObservationEncoder", "RewardCalculator", "RewardWeights", "IntervalStats",
           "TrafficSignalAgent", "DQNConfig"]


def __getattr__(name):  # torch is imported only when the agent is actually used
    if name in ("TrafficSignalAgent", "DQNConfig"):
        from . import agent

        return getattr(agent, name)
    raise AttributeError(name)
