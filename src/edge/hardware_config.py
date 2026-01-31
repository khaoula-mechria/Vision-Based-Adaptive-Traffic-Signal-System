"""
Hardware configuration for edge deployment
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DeviceType(Enum):
    """Supported edge device types"""
    CPU = "cpu"
    NVIDIA_JETSON_NANO = "jetson_nano"
    NVIDIA_JETSON_XAVIER = "jetson_xavier"
    NVIDIA_JETSON_ORIN = "jetson_orin"
    GOOGLE_CORAL = "coral"
    INTEL_NCS = "ncs"  # Neural Compute Stick


@dataclass
class HardwareConfig:
    """
    Hardware configuration for edge deployment.
    
    Defines constraints and optimizations for specific hardware targets.
    """
    device_type: DeviceType = DeviceType.CPU
    
    # Processing constraints
    target_fps: int = 15
    max_latency_ms: int = 100
    
    # Memory constraints
    max_memory_mb: int = 2048
    
    # Model optimization settings
    use_tensorrt: bool = False
    use_onnx: bool = True
    quantization: str = "fp32"  # 'fp32', 'fp16', 'int8'
    
    # Input constraints
    max_input_width: int = 640
    max_input_height: int = 480
    
    # Power settings (for battery/thermal management)
    power_mode: str = "balanced"  # 'low', 'balanced', 'high'
    
    @classmethod
    def for_jetson_nano(cls) -> "HardwareConfig":
        """Get configuration optimized for Jetson Nano"""
        return cls(
            device_type=DeviceType.NVIDIA_JETSON_NANO,
            target_fps=15,
            max_latency_ms=100,
            max_memory_mb=4096,
            use_tensorrt=True,
            use_onnx=False,
            quantization="fp16",
            max_input_width=416,
            max_input_height=416,
            power_mode="balanced",
        )
    
    @classmethod
    def for_jetson_xavier(cls) -> "HardwareConfig":
        """Get configuration optimized for Jetson Xavier"""
        return cls(
            device_type=DeviceType.NVIDIA_JETSON_XAVIER,
            target_fps=30,
            max_latency_ms=50,
            max_memory_mb=8192,
            use_tensorrt=True,
            use_onnx=False,
            quantization="fp16",
            max_input_width=640,
            max_input_height=640,
            power_mode="high",
        )
    
    @classmethod
    def for_jetson_orin(cls) -> "HardwareConfig":
        """Get configuration optimized for Jetson Orin"""
        return cls(
            device_type=DeviceType.NVIDIA_JETSON_ORIN,
            target_fps=60,
            max_latency_ms=30,
            max_memory_mb=16384,
            use_tensorrt=True,
            use_onnx=False,
            quantization="fp16",
            max_input_width=1280,
            max_input_height=720,
            power_mode="high",
        )
    
    @classmethod
    def for_coral(cls) -> "HardwareConfig":
        """Get configuration optimized for Google Coral"""
        return cls(
            device_type=DeviceType.GOOGLE_CORAL,
            target_fps=30,
            max_latency_ms=50,
            max_memory_mb=1024,
            use_tensorrt=False,
            use_onnx=False,  # Uses TFLite
            quantization="int8",
            max_input_width=300,
            max_input_height=300,
            power_mode="low",
        )
    
    @classmethod
    def for_cpu(cls) -> "HardwareConfig":
        """Get configuration for CPU-only deployment"""
        return cls(
            device_type=DeviceType.CPU,
            target_fps=10,
            max_latency_ms=150,
            max_memory_mb=4096,
            use_tensorrt=False,
            use_onnx=True,
            quantization="fp32",
            max_input_width=416,
            max_input_height=416,
            power_mode="balanced",
        )
    
    def get_model_input_size(self) -> tuple:
        """Get recommended model input size"""
        return (self.max_input_width, self.max_input_height)
    
    def get_batch_size(self) -> int:
        """Get recommended batch size"""
        if self.device_type in [DeviceType.NVIDIA_JETSON_ORIN, DeviceType.NVIDIA_JETSON_XAVIER]:
            return 4
        elif self.device_type == DeviceType.NVIDIA_JETSON_NANO:
            return 2
        return 1
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "device_type": self.device_type.value,
            "target_fps": self.target_fps,
            "max_latency_ms": self.max_latency_ms,
            "max_memory_mb": self.max_memory_mb,
            "use_tensorrt": self.use_tensorrt,
            "use_onnx": self.use_onnx,
            "quantization": self.quantization,
            "input_size": self.get_model_input_size(),
            "batch_size": self.get_batch_size(),
        }
