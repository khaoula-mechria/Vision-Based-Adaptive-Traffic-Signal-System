"""ONNX Runtime inference for YOLOv8 (CPU, or CUDA/TensorRT when the provider is present).

* Execution providers are checked against ``ort.get_available_providers()``; unavailable ones
  are skipped with a warning and the *actually active* providers are reported.
* Input handling: letterbox resize (aspect preserved), BGR->RGB, CHW float32 in [0,1].
* Output decoding targets the Ultralytics YOLOv8 export layout ``[1, 4 + num_classes, N]``
  (no objectness score), followed by class-wise NMS. Other layouts raise ``DetectionError``.
* Jetson / Coral: nothing here is verified on that hardware. TensorRT is used only if the
  ``TensorrtExecutionProvider`` is present in the installed onnxruntime build; Coral needs a
  separate full-integer TFLite model and is not implemented.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..errors import DetectionError, ModelNotFoundError
from ..perception.detection_result import DetectionResult
from ..perception.postprocess import COCO_TRAFFIC_NAMES, FilterConfig, build_detections, validate_frame

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

logger = logging.getLogger(__name__)

DEFAULT_PROVIDERS = ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")


def select_providers(requested: Sequence[str] = DEFAULT_PROVIDERS, available: Optional[Sequence[str]] = None) -> List[str]:
    """Keep requested providers that are actually installed; CPU is always the last resort."""
    if available is None:
        if ort is None:
            raise ImportError("onnxruntime is required: pip install -r requirements-vision.txt")
        available = ort.get_available_providers()
    chosen = [p for p in requested if p in available]
    for p in requested:
        if p not in available:
            logger.info("execution provider %s not available; skipping", p)
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return chosen


def letterbox(img: np.ndarray, size: int, pad_value: int = 114) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize keeping aspect ratio and pad to ``size``x``size``. Returns (image, scale, (pad_x, pad_y))."""
    import cv2

    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else img
    out = np.full((size, size, 3), pad_value, dtype=img.dtype)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py:py + nh, px:px + nw] = resized
    return out, scale, (px, py)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS; boxes are xyxy. Returns kept indices."""
    order = scores.argsort()[::-1]
    keep = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=int)


class OnnxYoloDetector:
    """Drop-in alternative to ``ObjectDetector`` with the same ``detect`` interface."""

    def __init__(
        self,
        model_path: str,
        names: Optional[Mapping[int, str]] = None,
        filter_config: Optional[FilterConfig] = None,
        providers: Sequence[str] = DEFAULT_PROVIDERS,
        input_size: int = 640,
        nms_threshold: float = 0.45,
        session=None,
    ) -> None:
        self.model_path = model_path
        self.names = dict(names) if names else dict(COCO_TRAFFIC_NAMES)
        self.filter = filter_config or FilterConfig()
        self.filter.validate()
        self.input_size = int(input_size)
        self.nms_threshold = nms_threshold
        self.requested_providers = tuple(providers)
        self.session = session  # injectable for tests
        self.frame_count = 0
        self.latencies_ms: List[float] = []
        if session is None:
            self.load_model()
        self.input_name = self.session.get_inputs()[0].name

    def load_model(self) -> None:
        if not os.path.isfile(self.model_path):
            raise ModelNotFoundError(f"ONNX model not found: {self.model_path}")
        providers = select_providers(self.requested_providers)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=providers)
        logger.info("ONNX session providers: %s", self.session.get_providers())

    @property
    def active_providers(self) -> List[str]:
        return list(self.session.get_providers()) if hasattr(self.session, "get_providers") else []

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        img, scale, pad = letterbox(frame, self.input_size)
        x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
        return x[None], scale, pad

    def decode(self, out: np.ndarray, scale: float, pad: Tuple[int, int], frame_hw: Tuple[int, int]) -> DetectionResult:
        arr = np.asarray(out)
        if arr.ndim != 3 or arr.shape[0] != 1:
            raise DetectionError(f"unexpected ONNX output shape {arr.shape}; expected [1, 4+C, N]")
        pred = arr[0]
        if pred.shape[0] > 1000 and pred.shape[1] < pred.shape[0]:  # anchors-first export [N, 4+C]
            pred = pred.T
        if pred.shape[0] < 5:
            raise DetectionError(f"unexpected ONNX output shape {arr.shape}")
        boxes_cxcywh, scores = pred[:4].T, pred[4:].T
        cls = scores.argmax(1)
        conf = scores[np.arange(len(cls)), cls]
        m = conf >= min(self.filter.confidence_threshold, *self.filter.per_class_confidence.values()) \
            if self.filter.per_class_confidence else conf >= self.filter.confidence_threshold
        boxes_cxcywh, conf, cls = boxes_cxcywh[m], conf[m], cls[m]
        xyxy = np.stack([boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2, boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2,
                         boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2, boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2], 1)
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad[0]) / scale
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad[1]) / scale
        kept: list = []
        for k in np.unique(cls):  # class-wise NMS
            idx = np.where(cls == k)[0]
            kept.extend(idx[nms(xyxy[idx], conf[idx], self.nms_threshold)])
        keep = np.asarray(sorted(kept), dtype=int)
        dets = build_detections(xyxy[keep], conf[keep], cls[keep], self.names, self.filter, frame_hw) if len(keep) else []
        return DetectionResult(frame_id=self.frame_count, timestamp=0.0, detections=dets)

    def detect(self, frame: np.ndarray, timestamp: Optional[float] = None) -> DetectionResult:
        validate_frame(frame)
        self.frame_count += 1
        x, scale, pad = self.preprocess(frame)
        t0 = time.perf_counter()
        try:
            out = self.session.run(None, {self.input_name: x})[0]
        except Exception as exc:  # noqa: BLE001
            raise DetectionError(f"ONNX inference failed: {exc}") from exc
        ms = (time.perf_counter() - t0) * 1000.0
        self.latencies_ms.append(ms)
        self.latencies_ms = self.latencies_ms[-500:]
        res = self.decode(out, scale, pad, frame.shape[:2])
        res.timestamp = time.time() if timestamp is None else timestamp
        res.inference_time_ms = ms
        res.update_counts()
        return res

    def benchmark(self, frame: Optional[np.ndarray] = None, warmup: int = 10, runs: int = 100) -> Dict[str, float]:
        """Measure end-to-end ``detect`` latency (pre-process + inference + decode) after warm-up."""
        frame = frame if frame is not None else np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(warmup):
            self.detect(frame)
        lat = []
        for _ in range(runs):
            t0 = time.perf_counter()
            self.detect(frame)
            lat.append((time.perf_counter() - t0) * 1000.0)
        a = np.asarray(lat)
        rss = None
        try:
            import psutil

            rss = psutil.Process().memory_info().rss / 1e6
        except ImportError:
            pass
        return {"runs": runs, "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
                "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max()), "fps": float(1000.0 / a.mean()),
                "rss_mb": rss if rss is not None else float("nan")}


def export_yolo_to_onnx(weights: str, imgsz: int = 640) -> str:
    """Export Ultralytics weights to ONNX next to the .pt file (requires ultralytics + onnx)."""
    from ultralytics import YOLO

    return str(YOLO(weights).export(format="onnx", imgsz=imgsz, simplify=False))
