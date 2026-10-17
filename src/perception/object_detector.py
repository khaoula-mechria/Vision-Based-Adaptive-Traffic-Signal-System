"""YOLOv8 object detection via Ultralytics (PyTorch).

* The model is loaded lazily on the first ``detect`` call (or injected for tests).
* ``device="cuda"`` falls back to CPU with a warning when CUDA is unavailable.
* Weights are never downloaded implicitly: point ``model_path`` at an existing file, or pass
  ``allow_download=True`` to let Ultralytics fetch a stock model by name.
* Emergency vehicles are detected only if the weights contain emergency classes
  (e.g. ``ambulance``); stock COCO weights cannot detect them, and no colour heuristic is used.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from ..errors import DetectionError, ModelNotFoundError
from .detection_result import DetectionResult
from .postprocess import COCO_TRAFFIC_NAMES, FilterConfig, build_detections, validate_frame

logger = logging.getLogger(__name__)


def resolve_device(requested: str) -> str:
    """Return a usable torch device string; CUDA requests fall back to CPU if unavailable."""
    if requested.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return requested
        except ImportError:
            pass
        logger.warning("CUDA requested but not available; falling back to CPU")
        return "cpu"
    return requested


class ObjectDetector:
    def __init__(
        self,
        model_name: str = "yolov8n",
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        device: str = "cpu",
        input_size: Tuple[int, int] = (640, 640),
        filter_config: Optional[FilterConfig] = None,
        allow_download: bool = False,
        model: Any = None,
    ) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.nms_threshold = nms_threshold
        self.requested_device = device
        self.input_size = tuple(input_size)
        self.allow_download = allow_download
        self.filter = filter_config or FilterConfig(confidence_threshold=confidence_threshold)
        self.filter.validate()
        self.model = model  # may be injected (tests, custom runtimes)
        self.device: Optional[str] = None
        self.frame_count = 0

    @property
    def confidence_threshold(self) -> float:
        return self.filter.confidence_threshold

    def load_model(self) -> None:
        path = self.model_path or f"{self.model_name}.pt"
        if not os.path.isfile(path) and not self.allow_download:
            raise ModelNotFoundError(
                f"YOLO weights not found at {path!r}. Place the .pt file there, set perception.model_path, "
                "or enable perception.allow_download. See README 'Model setup'."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is required: pip install -r requirements-vision.txt") from exc
        self.device = resolve_device(self.requested_device)
        self.model = YOLO(path)
        self.model.to(self.device)
        logger.info("loaded %s on %s", path, self.device)

    def _names(self) -> Mapping[int, str]:
        names = getattr(self.model, "names", None)
        return names if isinstance(names, Mapping) and names else COCO_TRAFFIC_NAMES

    def detect(self, frame: np.ndarray, timestamp: Optional[float] = None) -> DetectionResult:
        """Run detection on one BGR frame. Raises ``DetectionError`` on invalid input or model failure."""
        validate_frame(frame)
        if self.model is None:
            self.load_model()
        ts = time.time() if timestamp is None else timestamp
        self.frame_count += 1
        t0 = time.perf_counter()
        try:
            results = self.model(frame, conf=self.filter.confidence_threshold, iou=self.nms_threshold,
                                 imgsz=self.input_size[0], device=self.device, verbose=False)
        except Exception as exc:  # noqa: BLE001 - model/runtime failures must not crash the controller
            raise DetectionError(f"inference failed: {exc}") from exc
        infer_ms = (time.perf_counter() - t0) * 1000.0
        dets = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            dets += build_detections(
                _np(boxes.xyxy), _np(boxes.conf), _np(boxes.cls), self._names(), self.filter, frame.shape[:2]
            )
        result = DetectionResult(frame_id=self.frame_count, timestamp=ts, detections=dets,
                                 inference_time_ms=infer_ms)
        result.update_counts()
        return result

    def get_model_info(self) -> dict:
        return {
            "model_name": self.model_name, "model_path": self.model_path, "device": self.device,
            "confidence_threshold": self.confidence_threshold, "nms_threshold": self.nms_threshold,
            "input_size": self.input_size, "frame_count": self.frame_count,
        }


def _np(x) -> np.ndarray:
    """torch tensor / array-like -> numpy (one bulk transfer per tensor)."""
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x)
