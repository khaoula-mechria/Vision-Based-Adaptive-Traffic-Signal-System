"""Exception types shared across the traffic signal system."""


class TrafficSystemError(Exception):
    """Base class for all errors raised by this package."""


class ConfigError(TrafficSystemError, ValueError):
    """Configuration is missing, malformed or fails validation."""


class IntersectionModelError(ConfigError):
    """The intersection / phase definition is internally inconsistent (e.g. a phase
    gives green to two conflicting movements)."""


class DetectionError(TrafficSystemError):
    """Perception failed (bad frame, model failure, camera interruption)."""


class ModelNotFoundError(TrafficSystemError, FileNotFoundError):
    """A required model file (YOLO weights, ONNX file, RL checkpoint) does not exist."""


class SimulationError(TrafficSystemError):
    """The traffic simulator could not be started, crashed, or is unavailable."""
