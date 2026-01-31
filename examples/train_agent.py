#!/usr/bin/env python
"""
Example: Training the RL Agent for Traffic Signal Control

This example demonstrates how to train the reinforcement learning agent
using the internal traffic simulation environment.
"""

import sys
sys.path.insert(0, '.')

from src.rl_agent import TrafficSignalAgent, TrafficSignalEnv


def main():
    print("=" * 60)
    print("Vision-Based Adaptive Traffic Signal System")
    print("RL Agent Training Example")
    print("=" * 60)
    
    # Create environment
    print("\n1. Creating traffic environment...")
    env = TrafficSignalEnv(
        num_lanes=8,
        num_crosswalks=4,
        max_steps=200,  # Short episodes for demo
    )
    
    print(f"   - Number of lanes: {env.num_lanes}")
    print(f"   - Number of crosswalks: {env.num_crosswalks}")
    print(f"   - Signal phases: {env.PHASES}")
    
    # Create agent
    print("\n2. Creating RL agent...")
    agent = TrafficSignalAgent(
        state_dim=28,  # Observation dimension
        action_dim=5,  # Number of signal phases
        algorithm="DQN",
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.99,
    )
    
    print(f"   - Algorithm: {agent.algorithm}")
    print(f"   - State dimension: {agent.state_dim}")
    print(f"   - Action dimension: {agent.action_dim}")
    
    # Quick training demo
    print("\n3. Running quick training demo (5 episodes)...")
    
    for episode in range(5):
        state, _ = env.reset(seed=42 + episode)
        episode_reward = 0.0
        done = False
        step = 0
        
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.store_experience(state, action, reward, next_state, done)
            agent.train_step()
            
            state = next_state
            episode_reward += reward
            step += 1
        
        print(f"   Episode {episode + 1}: Steps={step}, Reward={episode_reward:.2f}, "
              f"Epsilon={agent.epsilon:.3f}")
    
    # Evaluation
    print("\n4. Evaluating trained agent...")
    metrics = agent.evaluate(env, episodes=3)
    
    print(f"   - Mean reward: {metrics['mean_reward']:.2f}")
    print(f"   - Mean waiting time: {metrics['mean_waiting_time']:.2f}s")
    print(f"   - Mean queue length: {metrics['mean_queue_length']:.2f}")
    print(f"   - Total throughput: {metrics['total_throughput']:.0f} vehicles")
    
    print("\n" + "=" * 60)
    print("Training demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
