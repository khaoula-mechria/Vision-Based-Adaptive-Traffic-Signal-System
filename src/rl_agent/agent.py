"""
Traffic Signal RL Agent
Implements DQN and PPO agents for signal optimization
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import logging

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from collections import deque
    import random
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    optim = None
    TORCH_AVAILABLE = False

# Import deque separately since it's used without torch
from collections import deque
import random

logger = logging.getLogger(__name__)


class DQNNetwork:
    """Deep Q-Network architecture (requires PyTorch)"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQNNetwork")
        
        # Initialize as nn.Module
        import torch.nn as nn_module
        
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn_module.Linear(prev_dim, hidden_dim))
            layers.append(nn_module.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn_module.Linear(prev_dim, action_dim))
        
        self.network = nn_module.Sequential(*layers)
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def forward(self, x):
        return self.network(x)
    
    def __call__(self, x):
        return self.forward(x)
    
    def to(self, device):
        self.network = self.network.to(device)
        return self
    
    def state_dict(self):
        return self.network.state_dict()
    
    def load_state_dict(self, state_dict):
        self.network.load_state_dict(state_dict)
    
    def parameters(self):
        return self.network.parameters()


class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )
    
    def __len__(self):
        return len(self.buffer)


class TrafficSignalAgent:
    """
    Reinforcement Learning agent for traffic signal control.
    
    Supports:
    - DQN (Deep Q-Network)
    - Integration with stable-baselines3 for PPO/A2C
    
    The agent:
    - Observes lane-level traffic states and pedestrian density
    - Selects signal phase durations and transitions
    - Optimizes for reduced waiting time while maintaining safety
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        algorithm: str = "DQN",
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_size: int = 100000,
        target_update_freq: int = 1000,
        device: str = "auto",
    ):
        """
        Initialize the RL agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            algorithm: RL algorithm ('DQN', 'PPO', 'A2C')
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            batch_size: Training batch size
            buffer_size: Replay buffer size
            target_update_freq: Target network update frequency
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else "cpu"
        else:
            self.device = torch.device(device) if TORCH_AVAILABLE else device
        
        # Initialize based on algorithm
        if algorithm == "DQN" and TORCH_AVAILABLE:
            self._init_dqn(buffer_size)
        elif algorithm in ["PPO", "A2C"]:
            self._init_sb3(algorithm)
        else:
            logger.warning(f"Algorithm {algorithm} not available, using random policy")
            self.model = None
        
        # Training statistics
        self.training_steps = 0
        self.episodes = 0
        self.total_reward = 0.0
        self.episode_rewards: List[float] = []
    
    def _init_dqn(self, buffer_size: int):
        """Initialize DQN components"""
        # Q-networks
        self.q_network = DQNNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DQNNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        self.model = self.q_network
    
    def _init_sb3(self, algorithm: str):
        """Initialize stable-baselines3 agent"""
        try:
            from stable_baselines3 import PPO, A2C
            from stable_baselines3.common.vec_env import DummyVecEnv
            
            # Model will be created when training starts
            self.sb3_class = PPO if algorithm == "PPO" else A2C
            self.model = None
            logger.info(f"Stable-baselines3 {algorithm} initialized")
        except ImportError:
            logger.warning("stable-baselines3 not available")
            self.model = None
    
    def select_action(
        self,
        state: np.ndarray,
        training: bool = True,
    ) -> int:
        """
        Select an action given current state.
        
        Args:
            state: Current observation
            training: Whether in training mode (enables exploration)
            
        Returns:
            Selected action
        """
        if self.algorithm == "DQN" and TORCH_AVAILABLE and self.model is not None:
            return self._select_action_dqn(state, training)
        elif self.model is not None and hasattr(self.model, 'predict'):
            # Stable-baselines3
            action, _ = self.model.predict(state, deterministic=not training)
            return int(action)
        else:
            # Random policy
            return np.random.randint(0, self.action_dim)
    
    def _select_action_dqn(self, state: np.ndarray, training: bool) -> int:
        """Select action using DQN with epsilon-greedy"""
        # Epsilon-greedy exploration
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        # Greedy action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store experience in replay buffer (DQN only)"""
        if self.algorithm == "DQN" and hasattr(self, 'replay_buffer'):
            self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if self.algorithm == "DQN":
            return self._train_step_dqn()
        return None
    
    def _train_step_dqn(self) -> Optional[float]:
        """DQN training step"""
        if not TORCH_AVAILABLE or len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + self.gamma * next_q * (1 - dones.unsqueeze(1))
        
        # Loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def train(
        self,
        env,
        episodes: int = 1000,
        log_interval: int = 100,
        save_path: Optional[str] = None,
    ) -> List[float]:
        """
        Train the agent.
        
        Args:
            env: Training environment
            episodes: Number of training episodes
            log_interval: Logging interval
            save_path: Path to save model
            
        Returns:
            List of episode rewards
        """
        logger.info(f"Starting training for {episodes} episodes")
        
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            done = False
            
            while not done:
                # Select and execute action
                action = self.select_action(state, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Store experience and train
                self.store_experience(state, action, reward, next_state, done)
                loss = self.train_step()
                
                # Update state
                state = next_state
                episode_reward += reward
            
            # Episode complete
            self.episodes += 1
            self.episode_rewards.append(episode_reward)
            
            # Logging
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-log_interval:])
                logger.info(
                    f"Episode {episode + 1}/{episodes}, "
                    f"Avg Reward: {avg_reward:.2f}, "
                    f"Epsilon: {self.epsilon:.3f}"
                )
                
                # Save checkpoint
                if save_path:
                    self.save(f"{save_path}_ep{episode + 1}.pt")
        
        # Final save
        if save_path:
            self.save(f"{save_path}_final.pt")
        
        return self.episode_rewards
    
    def evaluate(
        self,
        env,
        episodes: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate the agent.
        
        Args:
            env: Evaluation environment
            episodes: Number of evaluation episodes
            
        Returns:
            Dictionary of evaluation metrics
        """
        rewards = []
        waiting_times = []
        queue_lengths = []
        throughputs = []
        
        for _ in range(episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            done = False
            
            while not done:
                action = self.select_action(state, training=False)
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                waiting_times.append(info.get("avg_waiting_time", 0))
                queue_lengths.append(info.get("total_queue", 0))
                throughputs.append(info.get("vehicles_passed", 0))
            
            rewards.append(episode_reward)
        
        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_waiting_time": np.mean(waiting_times),
            "mean_queue_length": np.mean(queue_lengths),
            "total_throughput": np.sum(throughputs),
        }
    
    def save(self, path: str):
        """Save model to file"""
        if self.algorithm == "DQN" and TORCH_AVAILABLE and self.model is not None:
            torch.save({
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_steps": self.training_steps,
                "episodes": self.episodes,
            }, path)
            logger.info(f"Model saved to {path}")
        elif self.model is not None and hasattr(self.model, 'save'):
            self.model.save(path)
    
    def load(self, path: str):
        """Load model from file"""
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return
        
        if self.algorithm == "DQN" and TORCH_AVAILABLE:
            checkpoint = torch.load(path, map_location=self.device)
            self.q_network.load_state_dict(checkpoint["q_network"])
            self.target_network.load_state_dict(checkpoint["target_network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
            self.training_steps = checkpoint.get("training_steps", 0)
            self.episodes = checkpoint.get("episodes", 0)
            logger.info(f"Model loaded from {path}")
        elif hasattr(self, 'sb3_class'):
            self.model = self.sb3_class.load(path)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            "algorithm": self.algorithm,
            "training_steps": self.training_steps,
            "episodes": self.episodes,
            "epsilon": self.epsilon,
            "mean_reward": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            "buffer_size": len(self.replay_buffer) if hasattr(self, 'replay_buffer') else 0,
        }
