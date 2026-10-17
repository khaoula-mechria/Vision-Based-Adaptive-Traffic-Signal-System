"""Camera / video file reader that survives interruptions.

``read()`` never raises on a failed grab: it returns ``(None, timestamp)`` and counts
consecutive failures so the controller can mark the sensors degraded, and it tries to
re-open the device after ``reconnect_after`` consecutive failures (live cameras only).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class VideoSource:
    def __init__(self, source: Union[int, str], reconnect_after: int = 5, capture_factory: Any = None) -> None:
        self.source = source
        self.reconnect_after = reconnect_after
        self._factory = capture_factory
        self._cap: Any = None
        self.consecutive_failures = 0
        self.total_failures = 0
        self.reconnects = 0
        self._open()

    def _open(self) -> None:
        if self._factory is None:
            try:
                import cv2
            except ImportError as exc:
                raise ImportError("opencv-python is required for video input: pip install -r requirements-vision.txt") from exc
            self._factory = cv2.VideoCapture
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001
                pass
        self._cap = self._factory(self.source)

    @property
    def is_open(self) -> bool:
        return bool(self._cap is not None and self._cap.isOpened())

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        ts = time.time()
        frame = None
        try:
            if self.is_open:
                ok, img = self._cap.read()
                frame = img if ok else None
        except Exception:  # noqa: BLE001 - backend decode errors count as a failed grab
            frame = None
        if frame is None:
            self.consecutive_failures += 1
            self.total_failures += 1
            if self.consecutive_failures >= self.reconnect_after and isinstance(self.source, int):
                logger.warning("camera %s: %d failed reads, reconnecting", self.source, self.consecutive_failures)
                self.reconnects += 1
                self.consecutive_failures = 0
                self._open()
        else:
            self.consecutive_failures = 0
        return frame, ts

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
