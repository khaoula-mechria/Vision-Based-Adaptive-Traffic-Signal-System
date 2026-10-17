"""Turn raw detector output into validated ``Detection`` objects.

Shared by the Ultralytics detector and the ONNX Runtime detector so that thresholds,
class mapping and sanity checks behave identically on both paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Mapping, Tuple

import numpy as np

from ..errors import DetectionError
from .detection_result import BoundingBox, Detection

# COCO ids that matter for traffic scenes (YOLOv8 pre-trained weights).
COCO_TRAFFIC_NAMES: Dict[int, str] = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

VEHICLE_CLASSES: FrozenSet[str] = frozenset({"car", "bus", "truck", "motorcycle"})
PEDESTRIAN_CLASSES: FrozenSet[str] = frozenset({"person"})
CYCLIST_CLASSES: FrozenSet[str] = frozenset({"bicycle"})
# Emergency classes only exist in custom-trained weights; stock COCO models cannot detect them.
EMERGENCY_CLASSES: FrozenSet[str] = frozenset(
    {"ambulance", "fire_truck", "fire truck", "firetruck", "police_car", "police car", "police"}
)


@dataclass
class FilterConfig:
    confidence_threshold: float = 0.5
    per_class_confidence: Mapping[str, float] = field(default_factory=dict)
    emergency_confidence: float = 0.7  # higher bar: a false emergency triggers preemption
    min_box_area_px: float = 100.0
    max_box_area_fraction: float = 0.9  # a box covering the whole frame is not a vehicle

    def validate(self) -> None:
        for name, v in [("confidence_threshold", self.confidence_threshold),
                        ("emergency_confidence", self.emergency_confidence),
                        ("max_box_area_fraction", self.max_box_area_fraction)]:
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        if self.min_box_area_px < 0:
            raise ValueError("min_box_area_px must be >= 0")


def validate_frame(frame) -> np.ndarray:
    """Reject missing / malformed frames (camera interruption, corrupted decode)."""
    if frame is None:
        raise DetectionError("no frame (camera interruption?)")
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise DetectionError(f"expected HxWx3 image array, got {getattr(frame, 'shape', type(frame))}")
    if frame.shape[0] < 16 or frame.shape[1] < 16:
        raise DetectionError(f"frame too small: {frame.shape}")
    return frame


def build_detections(
    xyxy: np.ndarray,
    conf: np.ndarray,
    cls_ids: np.ndarray,
    names: Mapping[int, str],
    cfg: FilterConfig,
    frame_hw: Tuple[int, int],
) -> List[Detection]:
    """Filter and convert detector output. Invalid rows are dropped, never raised."""
    h, w = frame_hw
    out: List[Detection] = []
    xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
    conf = np.asarray(conf, dtype=float).reshape(-1)
    cls_ids = np.asarray(cls_ids).reshape(-1)
    for box, c, k in zip(xyxy, conf, cls_ids):
        if not (np.all(np.isfinite(box)) and np.isfinite(c)):
            continue
        name = names.get(int(k))
        if name is None:
            continue  # class we do not care about (or unknown id)
        is_emerg_cls = name in EMERGENCY_CLASSES
        is_vehicle = name in VEHICLE_CLASSES or is_emerg_cls
        is_ped, is_cyc = name in PEDESTRIAN_CLASSES, name in CYCLIST_CLASSES
        if not (is_vehicle or is_ped or is_cyc):
            continue
        if c < cfg.per_class_confidence.get(name, cfg.confidence_threshold):
            continue
        x1, y1 = max(0.0, box[0]), max(0.0, box[1])
        x2, y2 = min(float(w), box[2]), min(float(h), box[3])
        area = (x2 - x1) * (y2 - y1)
        if x2 <= x1 or y2 <= y1 or area < cfg.min_box_area_px or area > cfg.max_box_area_fraction * w * h:
            continue
        out.append(Detection(
            bbox=BoundingBox(float(x1), float(y1), float(x2), float(y2)),
            class_id=int(k), class_name=name, confidence=float(c),
            is_vehicle=is_vehicle, is_pedestrian=is_ped, is_cyclist=is_cyc,
            is_emergency=is_emerg_cls and c >= cfg.emergency_confidence,
        ))
    return out
