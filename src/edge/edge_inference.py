"""
Edge Inference Module
Optimized inference for edge devices (Jetson, Coral, etc.)
"""

import time
from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False

from .hardware_config import HardwareConfig, DeviceType

logger = logging.getLogger(__name__)


class EdgeInference:
    """
    Optimized inference engine for edge deployment.
    
    Features:
    - ONNX Runtime optimization
    - TensorRT acceleration (NVIDIA devices)
    - Dynamic batching
    - Latency monitoring
    - Memory management
    
    Designed for real-world deployment on:
    - NVIDIA Jetson (Nano, Xavier, Orin)
    - Google Coral
    - CPU-only systems
    """
    
    def __init__(
        self,
        model_path: str,
        config: Optional[HardwareConfig] = None,
    ):
        """
        Initialize edge inference engine.
        
        Args:
            model_path: Path to model file (ONNX, TensorRT, or TFLite)
            config: Hardware configuration
        """
        self.model_path = model_path
        self.config = config or HardwareConfig.for_cpu()
        
        self.session = None
        self.input_name = None
        self.output_names = None
        
        # Performance monitoring
        self.inference_times: List[float] = []
        self.frame_count = 0
        
    def load_model(self):
        """Load and optimize model for inference"""
        if self.config.use_onnx and ONNX_AVAILABLE:
            self._load_onnx_model()
        elif self.config.use_tensorrt:
            self._load_tensorrt_model()
        else:
            logger.warning("No supported inference runtime available")
    
    def _load_onnx_model(self):
        """Load ONNX model with optimizations"""
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime not available")
        
        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Set number of threads based on hardware
        if self.config.device_type == DeviceType.CPU:
            import multiprocessing
            sess_options.intra_op_num_threads = multiprocessing.cpu_count()
        
        # Select execution provider
        providers = self._get_execution_providers()
        
        try:
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers,
            )
            
            # Get input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            
            logger.info(f"ONNX model loaded with providers: {providers}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def _get_execution_providers(self) -> List[str]:
        """Get appropriate execution providers for hardware"""
        providers = []
        
        if self.config.device_type in [
            DeviceType.NVIDIA_JETSON_NANO,
            DeviceType.NVIDIA_JETSON_XAVIER,
            DeviceType.NVIDIA_JETSON_ORIN,
        ]:
            # Try CUDA/TensorRT
            if 'TensorrtExecutionProvider' in ort.get_available_providers():
                providers.append('TensorrtExecutionProvider')
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append('CUDAExecutionProvider')
        
        # Always add CPU as fallback
        providers.append('CPUExecutionProvider')
        
        return providers
    
    def _load_tensorrt_model(self):
        """Load TensorRT optimized model"""
        # TensorRT loading would be implemented here
        # This requires trt Python bindings
        logger.warning("TensorRT loading not fully implemented")
        self._load_onnx_model()  # Fallback to ONNX
    
    def preprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (BGR, HWC format)
            
        Returns:
            Preprocessed tensor
        """
        target_size = self.config.get_model_input_size()
        
        # Resize
        import cv2
        resized = cv2.resize(image, target_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Convert to CHW format
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def infer(
        self,
        input_tensor: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Run inference on preprocessed input.
        
        Args:
            input_tensor: Preprocessed input tensor
            
        Returns:
            List of output tensors
        """
        if self.session is None:
            self.load_model()
        
        start_time = time.time()
        
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_tensor},
        )
        
        # Record timing
        inference_time = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time)
        self.frame_count += 1
        
        # Keep only recent times for monitoring
        if len(self.inference_times) > 100:
            self.inference_times = self.inference_times[-100:]
        
        return outputs
    
    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Run complete detection pipeline.
        
        Args:
            image: Input image (BGR format)
            confidence_threshold: Minimum detection confidence
            
        Returns:
            List of detections with bboxes and classes
        """
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Infer
        outputs = self.infer(input_tensor)
        
        # Postprocess (YOLO format assumed)
        detections = self._postprocess_yolo(
            outputs,
            image.shape[:2],
            confidence_threshold,
        )
        
        return detections
    
    def _postprocess_yolo(
        self,
        outputs: List[np.ndarray],
        original_size: Tuple[int, int],
        confidence_threshold: float,
    ) -> List[Dict]:
        """Postprocess YOLO outputs"""
        detections = []
        
        # YOLO output format: [batch, num_detections, 5 + num_classes]
        # [x_center, y_center, width, height, confidence, class_probs...]
        
        if len(outputs) == 0:
            return detections
        
        predictions = outputs[0]
        
        if len(predictions.shape) == 3:
            predictions = predictions[0]  # Remove batch dimension
        
        input_size = self.config.get_model_input_size()
        scale_x = original_size[1] / input_size[0]
        scale_y = original_size[0] / input_size[1]
        
        for pred in predictions:
            confidence = pred[4]
            if confidence < confidence_threshold:
                continue
            
            # Get class with highest probability
            class_probs = pred[5:]
            class_id = np.argmax(class_probs)
            class_conf = class_probs[class_id]
            
            if confidence * class_conf < confidence_threshold:
                continue
            
            # Convert to corner format and scale
            x_center, y_center, width, height = pred[:4]
            x1 = (x_center - width / 2) * scale_x
            y1 = (y_center - height / 2) * scale_y
            x2 = (x_center + width / 2) * scale_x
            y2 = (y_center + height / 2) * scale_y
            
            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(confidence * class_conf),
                "class_id": int(class_id),
            })
        
        return detections
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get inference latency statistics"""
        if not self.inference_times:
            return {
                "mean_ms": 0.0,
                "std_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "fps": 0.0,
            }
        
        times = np.array(self.inference_times)
        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "fps": 1000.0 / np.mean(times) if np.mean(times) > 0 else 0.0,
        }
    
    def meets_latency_target(self) -> bool:
        """Check if inference meets latency target"""
        if not self.inference_times:
            return True
        
        avg_latency = np.mean(self.inference_times)
        return avg_latency <= self.config.max_latency_ms
    
    def meets_fps_target(self) -> bool:
        """Check if inference meets FPS target"""
        stats = self.get_latency_stats()
        return stats["fps"] >= self.config.target_fps
    
    def get_summary(self) -> Dict:
        """Get inference engine summary"""
        latency_stats = self.get_latency_stats()
        
        return {
            "model_path": self.model_path,
            "hardware": self.config.to_dict(),
            "frames_processed": self.frame_count,
            "latency": latency_stats,
            "meets_latency_target": self.meets_latency_target(),
            "meets_fps_target": self.meets_fps_target(),
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.inference_times.clear()
        self.frame_count = 0


def optimize_model_for_edge(
    input_model_path: str,
    output_model_path: str,
    config: HardwareConfig,
) -> str:
    """
    Optimize model for edge deployment.
    
    Args:
        input_model_path: Path to input model
        output_model_path: Path for optimized model
        config: Target hardware configuration
        
    Returns:
        Path to optimized model
    """
    if not ONNX_AVAILABLE:
        logger.warning("ONNX not available, skipping optimization")
        return input_model_path
    
    try:
        import onnx
        from onnxruntime.transformers import optimizer
        
        # Load model
        model = onnx.load(input_model_path)
        
        # Optimize
        optimized = optimizer.optimize_model(
            input_model_path,
            model_type='bert',  # Generic optimization
            num_heads=0,
            hidden_size=0,
        )
        
        # Apply quantization if requested
        if config.quantization == "fp16":
            optimized.convert_float_to_float16()
        elif config.quantization == "int8":
            # INT8 quantization requires calibration data
            logger.warning("INT8 quantization requires calibration, skipping")
        
        # Save optimized model
        optimized.save_model_to_file(output_model_path)
        
        logger.info(f"Model optimized and saved to {output_model_path}")
        return output_model_path
        
    except Exception as e:
        logger.error(f"Model optimization failed: {e}")
        return input_model_path
