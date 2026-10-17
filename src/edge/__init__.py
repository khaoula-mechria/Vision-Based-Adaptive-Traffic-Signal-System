"""ONNX Runtime edge inference and benchmarking."""

from .edge_inference import OnnxYoloDetector, select_providers

__all__ = ["OnnxYoloDetector", "select_providers"]
