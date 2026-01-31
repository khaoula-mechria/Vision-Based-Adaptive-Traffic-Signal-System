# Vision-Based Adaptive Traffic Signal System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A real-time, computer vision-driven traffic signal control system that replaces traditional fixed-timer traffic lights with an adaptive, learning-based approach. The system perceives actual road conditions, estimates lane-level congestion, prioritizes pedestrian safety, and autonomously optimizes signal timing using reinforcement learning.

## Features

### 1. Vision-Based Perception (Core Layer)
- **Object Detection**: YOLOv8-based detection of vehicles (cars, buses, trucks, motorcycles), pedestrians, cyclists, and emergency vehicles
- **Multi-Object Tracking**: SORT algorithm with Kalman filtering for persistent ID tracking
- **Robust Operation**: Designed for varying lighting and weather conditions

### 2. Lane-Level Queue & Flow Estimation
- Queue length estimation per lane using spatial position and object density
- Vehicle arrival rate and lane occupancy measurement
- Average and maximum waiting time calculation
- Real-time congestion pattern detection and lane imbalance identification

### 3. Pedestrian-First Safety Constraints (Non-Negotiable)
- **Minimum Crossing Time**: Guaranteed based on crowd size and crossing width
- **Phase Change Prevention**: Protects pedestrians already in crosswalk
- **Dynamic Extension**: Automatic green phase extension during high foot traffic
- **Safety Override**: All safety constraints override traffic optimization objectives

### 4. Emergency Vehicle Priority Handling
- Visual detection of emergency vehicles (ambulance, fire truck, police)
- Flashing light detection for confirmation
- Signal preemption for green corridor creation
- Smooth traffic flow restoration after emergency passage

### 5. Reinforcement Learning Signal Optimization
- DQN (Deep Q-Network) agent for phase selection
- State features: queue lengths, waiting times, pedestrian density, emergency status
- Multi-objective reward function balancing:
  - Reduced vehicle waiting time
  - Reduced pedestrian delay
  - Congestion avoidance
  - Safety compliance

### 6. SUMO Simulation Integration
- Training environment using SUMO traffic simulator
- Baseline comparison with fixed-time and actuated signals
- Comprehensive metrics collection

### 7. Edge Deployment Support
- ONNX Runtime optimization for edge devices
- Support for NVIDIA Jetson (Nano, Xavier, Orin), Google Coral
- Low latency inference with configurable FPS targets

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA (optional, for GPU acceleration)
- SUMO (optional, for simulation)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install SUMO (Optional)

For simulation and training:

```bash
# Ubuntu/Debian
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc

# Set environment variable
export SUMO_HOME=/usr/share/sumo
```

## Project Structure

```
Vision-Based-Adaptive-Traffic-Signal-System/
├── config/
│   └── default_config.yaml       # System configuration
├── src/
│   ├── perception/               # Object detection module
│   │   ├── object_detector.py    # YOLOv8-based detector
│   │   └── detection_result.py   # Detection data structures
│   ├── tracking/                 # Multi-object tracking
│   │   ├── tracker.py            # SORT tracker implementation
│   │   ├── kalman_filter.py      # Kalman filter for tracking
│   │   └── track.py              # Track data structure
│   ├── queue_estimation/         # Queue and flow estimation
│   │   ├── queue_estimator.py    # Lane-level queue estimation
│   │   └── lane.py               # Lane data structure
│   ├── safety/                   # Pedestrian safety
│   │   ├── safety_manager.py     # Safety constraint enforcement
│   │   └── crosswalk.py          # Crosswalk state management
│   ├── emergency/                # Emergency vehicle handling
│   │   ├── emergency_handler.py  # Preemption logic
│   │   └── emergency_state.py    # Emergency state tracking
│   ├── rl_agent/                 # Reinforcement learning
│   │   ├── agent.py              # DQN/PPO agent
│   │   ├── traffic_env.py        # Gymnasium environment
│   │   └── reward.py             # Reward calculation
│   ├── simulation/               # SUMO integration
│   │   ├── sumo_interface.py     # SUMO TraCI interface
│   │   └── traffic_generator.py  # Traffic demand generation
│   ├── edge/                     # Edge deployment
│   │   ├── edge_inference.py     # Optimized inference
│   │   └── hardware_config.py    # Hardware configurations
│   ├── utils/                    # Utilities
│   │   ├── config_loader.py      # Configuration loading
│   │   └── metrics.py            # Metrics collection
│   └── controller.py             # Main traffic controller
├── tests/                        # Unit tests
├── requirements.txt              # Python dependencies
└── README.md
```

## Usage

### Basic Usage

```python
from src.controller import AdaptiveTrafficController
import cv2

# Initialize controller
controller = AdaptiveTrafficController(
    config_path="config/default_config.yaml",
    use_rl=True,
)

# Setup intersection (640x480 camera, 2 lanes per direction)
controller.setup_intersection(640, 480, num_lanes=2)

# Process video stream
cap = cv2.VideoCapture("intersection_video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame and get signal decision
    result = controller.step(frame)
    
    print(f"Phase: {result['selected_phase']}")
    print(f"Queue: {result['lane_states']}")

cap.release()
```

### Training the RL Agent

```python
from src.rl_agent import TrafficSignalAgent, TrafficSignalEnv

# Create environment
env = TrafficSignalEnv(
    num_lanes=8,
    num_crosswalks=4,
    max_steps=500,
)

# Create agent
agent = TrafficSignalAgent(
    state_dim=28,
    action_dim=5,
    algorithm="DQN",
)

# Train
episode_rewards = agent.train(
    env=env,
    episodes=1000,
    save_path="models/traffic_agent",
)

# Evaluate
metrics = agent.evaluate(env, episodes=10)
print(f"Average reward: {metrics['mean_reward']}")
```

### SUMO Simulation

```python
from src.simulation import SUMOInterface, TrafficGenerator

# Generate traffic demand
generator = TrafficGenerator(seed=42)
generator.generate_route_file(
    output_path="routes.rou.xml",
    duration=3600,
    density="medium",
)

# Run simulation with RL agent
sumo = SUMOInterface(
    network_file="intersection.net.xml",
    route_file="routes.rou.xml",
)

sumo.start()

for step in range(3600):
    state = sumo.step(phase=current_phase)
    # Use agent to select next phase
    next_phase = agent.select_action(state)
    current_phase = next_phase

metrics = sumo.get_metrics()
print(f"Throughput: {metrics['throughput']} vph")
sumo.close()
```

### Edge Deployment

```python
from src.edge import EdgeInference, HardwareConfig

# Configure for Jetson Nano
config = HardwareConfig.for_jetson_nano()

# Initialize inference engine
inference = EdgeInference(
    model_path="models/yolov8n.onnx",
    config=config,
)

# Run detection
detections = inference.detect(frame, confidence_threshold=0.5)

# Check performance
print(inference.get_latency_stats())
print(f"Meets FPS target: {inference.meets_fps_target()}")
```

## Configuration

The system is configured via YAML files. See `config/default_config.yaml` for all options:

```yaml
# Key configuration sections
perception:
  model: "yolov8n"
  confidence_threshold: 0.5
  device: "cuda"

safety:
  min_crossing_time_per_meter: 1.5
  high_traffic_pedestrian_count: 10

rl_agent:
  algorithm: "DQN"
  learning_rate: 0.0001
  gamma: 0.99
```

Environment variables can override configuration:
```bash
export TRAFFIC_PERCEPTION_DEVICE=cpu
export TRAFFIC_SAFETY_HIGH_TRAFFIC_PEDESTRIAN_COUNT=15
```

## Evaluation Metrics

The system tracks the following metrics:

| Metric | Description |
|--------|-------------|
| Average Vehicle Waiting Time | Mean wait time across all vehicles |
| Maximum Vehicle Waiting Time | Longest wait time observed |
| Average Pedestrian Waiting Time | Mean wait time for pedestrians |
| Queue Length per Lane | Number of vehicles queued per lane |
| Intersection Throughput | Vehicles per hour passing through |
| Emergency Clearance Time | Time to clear emergency vehicles |
| Inference Latency | Model inference time in milliseconds |
| Pedestrian Safety Score | Compliance with safety constraints |

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test module
pytest tests/test_safety.py -v
```

## Safety Guarantees

The system enforces several non-negotiable safety constraints:

1. **Pedestrian Protection**: Phase changes are blocked if pedestrians are detected in the crosswalk
2. **Minimum Crossing Time**: Calculated based on crosswalk width and crowd size
3. **Emergency Priority**: Emergency vehicles receive immediate green corridor
4. **Safety Override**: All optimization objectives are overridden by safety constraints

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Camera Input (Video Stream)                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Perception Layer                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Object Detection │  │  Multi-Object   │  │   Emergency     │  │
│  │    (YOLOv8)     │──│    Tracking     │──│   Detection     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Analysis Layer                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Queue Estimation │  │ Pedestrian      │  │ Emergency       │  │
│  │   (Per Lane)    │  │ Safety Manager  │  │ Handler         │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Decision Layer                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Safety Constraints (HARD)                   │    │
│  │   - Pedestrian protection (override all else)           │    │
│  │   - Emergency vehicle priority                          │    │
│  │   - Minimum phase durations                             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │         RL Agent (DQN) - Phase Optimization             │    │
│  │   - Minimize waiting time                               │    │
│  │   - Balance throughput                                  │    │
│  │   - Adapt to traffic patterns                          │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Signal Control Output                         │
│           (Phase selection with timing)                          │
└─────────────────────────────────────────────────────────────────┘
```

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [SUMO](https://www.eclipse.org/sumo/) for traffic simulation
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for RL algorithms