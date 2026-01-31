"""
Object Detection Module using YOLOv8
Detects vehicles, pedestrians, cyclists, and emergency vehicles
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from .detection_result import BoundingBox, Detection, DetectionResult


class ObjectDetector:
    """
    Real-time object detector for traffic scene analysis.
    Uses YOLOv8 for detection of vehicles, pedestrians, cyclists, and emergency vehicles.
    """
    
    # COCO class mapping for relevant traffic objects
    COCO_CLASSES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }
    
    # Custom classes for emergency vehicles (would be trained separately)
    EMERGENCY_CLASSES = {
        "ambulance": 100,
        "fire_truck": 101,
        "police_car": 102,
    }
    
    # Category mapping
    VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}
    PEDESTRIAN_CLASSES = {"person"}
    CYCLIST_CLASSES = {"bicycle"}
    
    def __init__(
        self,
        model_name: str = "yolov8n",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        device: str = "cpu",
        input_size: Tuple[int, int] = (640, 640),
    ):
        """
        Initialize the object detector.
        
        Args:
            model_name: YOLOv8 model variant (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
            device: Device to run inference on ('cuda' or 'cpu')
            input_size: Input image size (width, height)
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self.input_size = input_size
        
        self.model = None
        self.frame_count = 0
        
        # Emergency vehicle detection history for flashing light detection
        self._light_history: Dict[int, List[float]] = {}
        
    def load_model(self):
        """Load the YOLO model"""
        if YOLO is None:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")
        
        self.model = YOLO(f"{self.model_name}.pt")
        self.model.to(self.device)
        
    def detect(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> DetectionResult:
        """
        Perform object detection on a single frame.
        
        Args:
            frame: Input image as numpy array (BGR format)
            timestamp: Optional timestamp for the frame
            
        Returns:
            DetectionResult containing all detections
        """
        if self.model is None:
            self.load_model()
        
        if timestamp is None:
            timestamp = time.time()
        
        self.frame_count += 1
        start_time = time.time()
        
        # Run inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            imgsz=self.input_size,
            verbose=False,
        )
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for i in range(len(boxes)):
                # Get bounding box
                xyxy = boxes.xyxy[i].cpu().numpy()
                bbox = BoundingBox(
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                )
                
                # Get class and confidence
                class_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())
                
                # Get class name
                class_name = self.COCO_CLASSES.get(class_id, f"class_{class_id}")
                
                # Categorize detection
                is_vehicle = class_name in self.VEHICLE_CLASSES
                is_pedestrian = class_name in self.PEDESTRIAN_CLASSES
                is_cyclist = class_name in self.CYCLIST_CLASSES
                is_emergency = self._check_emergency_vehicle(frame, bbox, class_name)
                
                detection = Detection(
                    bbox=bbox,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    is_vehicle=is_vehicle,
                    is_pedestrian=is_pedestrian,
                    is_cyclist=is_cyclist,
                    is_emergency=is_emergency,
                )
                
                detections.append(detection)
        
        # Create result
        result = DetectionResult(
            frame_id=self.frame_count,
            timestamp=timestamp,
            detections=detections,
            inference_time_ms=inference_time,
        )
        result.update_counts()
        
        return result
    
    def _check_emergency_vehicle(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        class_name: str,
    ) -> bool:
        """
        Check if a detected vehicle is an emergency vehicle.
        Uses color analysis to detect emergency vehicle colors and flashing lights.
        
        Args:
            frame: Input frame
            bbox: Bounding box of the detection
            class_name: Detected class name
            
        Returns:
            True if emergency vehicle detected
        """
        if class_name not in self.VEHICLE_CLASSES:
            return False
        
        if cv2 is None:
            return False
            
        # Extract region of interest
        x1, y1 = int(max(0, bbox.x1)), int(max(0, bbox.y1))
        x2, y2 = int(min(frame.shape[1], bbox.x2)), int(min(frame.shape[0], bbox.y2))
        
        if x2 <= x1 or y2 <= y1:
            return False
            
        roi = frame[y1:y2, x1:x2]
        
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Check for emergency vehicle colors (red, blue)
        # Red color range
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Blue color range
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        
        # Create masks
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Calculate percentage of emergency colors
        total_pixels = roi.shape[0] * roi.shape[1]
        red_ratio = (np.sum(mask_red1) + np.sum(mask_red2)) / (total_pixels * 255)
        blue_ratio = np.sum(mask_blue) / (total_pixels * 255)
        
        # Emergency vehicle heuristic: significant red or blue coloring
        # This is a simplified approach - production would use a dedicated classifier
        if red_ratio > 0.1 or blue_ratio > 0.1:
            return True
        
        return False
    
    def detect_flashing_lights(
        self,
        frame: np.ndarray,
        detection: Detection,
        history_frames: int = 10,
    ) -> bool:
        """
        Detect flashing lights on a potential emergency vehicle.
        
        Args:
            frame: Current frame
            detection: Detection to analyze
            history_frames: Number of frames to keep in history
            
        Returns:
            True if flashing lights detected
        """
        if detection.track_id is None:
            return False
        
        if cv2 is None:
            return False
            
        # Extract top region of vehicle (where lights typically are)
        bbox = detection.bbox
        x1, y1 = int(max(0, bbox.x1)), int(max(0, bbox.y1))
        x2, y2 = int(min(frame.shape[1], bbox.x2)), int(min(frame.shape[0], bbox.y1 + bbox.height * 0.3))
        
        if x2 <= x1 or y2 <= y1:
            return False
            
        roi = frame[y1:y2, x1:x2]
        
        # Calculate average brightness
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # Update history
        track_id = detection.track_id
        if track_id not in self._light_history:
            self._light_history[track_id] = []
        
        self._light_history[track_id].append(brightness)
        
        # Keep only recent history
        if len(self._light_history[track_id]) > history_frames:
            self._light_history[track_id] = self._light_history[track_id][-history_frames:]
        
        # Check for flashing pattern (alternating high/low brightness)
        history = self._light_history[track_id]
        if len(history) < 5:
            return False
        
        # Calculate variance - flashing lights have high variance
        variance = np.var(history)
        
        # Threshold for flashing detection
        return variance > 500
    
    def process_video_stream(
        self,
        video_source,
        callback=None,
        max_frames: Optional[int] = None,
    ):
        """
        Process a video stream and yield detection results.
        
        Args:
            video_source: Video file path or camera index
            callback: Optional callback function for each frame result
            max_frames: Maximum number of frames to process
            
        Yields:
            DetectionResult for each frame
        """
        if cv2 is None:
            raise ImportError("opencv-python is required. Install with: pip install opencv-python")
            
        cap = cv2.VideoCapture(video_source)
        
        frame_count = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                result = self.detect(frame, timestamp)
                
                if callback:
                    callback(frame, result)
                
                yield result
                
                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    break
                    
        finally:
            cap.release()
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "device": self.device,
            "input_size": self.input_size,
            "frame_count": self.frame_count,
        }
