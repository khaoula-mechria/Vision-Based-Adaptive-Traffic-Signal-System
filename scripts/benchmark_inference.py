#!/usr/bin/env python
"""Measure detector latency/throughput (after warm-up) and optionally cross-check ONNX vs PyTorch.

    python scripts/benchmark_inference.py --onnx models/yolov8n.onnx --weights models/yolov8n.pt --image some.jpg

Reports the execution providers that are *actually active*. Numbers are specific to the machine
they are measured on; nothing here implies Jetson/Coral support.
"""

import argparse
import json
import os
import platform
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from src.edge import OnnxYoloDetector  # noqa: E402
from src.perception.postprocess import FilterConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--weights", default=None, help="PyTorch .pt for a detection cross-check / torch benchmark")
    ap.add_argument("--image", default=None, help="BGR image for the cross-check (random frame otherwise)")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--providers", nargs="+", default=None, help="e.g. CUDAExecutionProvider CPUExecutionProvider")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import cv2

    frame = cv2.imread(args.image) if args.image else None
    if frame is None:
        frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    kw = {"providers": args.providers} if args.providers else {}
    onnx = OnnxYoloDetector(args.onnx, filter_config=FilterConfig(0.4, min_box_area_px=100), input_size=args.size, **kw)
    report = {"machine": platform.processor() or platform.machine(), "python": platform.python_version(),
              "frame_shape": list(frame.shape), "onnx_active_providers": onnx.active_providers,
              "onnx": onnx.benchmark(frame, args.warmup, args.runs)}

    if args.weights:
        from src.perception.object_detector import ObjectDetector

        torch_det = ObjectDetector(model_path=args.weights, filter_config=FilterConfig(0.4, min_box_area_px=100),
                                   input_size=(args.size, args.size))
        import time

        for _ in range(args.warmup):
            torch_det.detect(frame)
        lat = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            torch_det.detect(frame)
            lat.append((time.perf_counter() - t0) * 1000)
        report["torch"] = {"device": torch_det.device, "mean_ms": float(np.mean(lat)),
                           "p95_ms": float(np.percentile(lat, 95)), "fps": float(1000 / np.mean(lat))}
        a, b = torch_det.detect(frame), onnx.detect(frame)
        report["crosscheck"] = {"torch_count": len(a.detections), "onnx_count": len(b.detections),
                                "torch_classes": sorted(d.class_name for d in a.detections),
                                "onnx_classes": sorted(d.class_name for d in b.detections),
                                "max_box_diff_px": _max_box_diff(a, b)}
    print(json.dumps(report, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
    return 0


def _max_box_diff(a, b):
    """Greedy same-class matching; returns the largest coordinate difference of matched boxes (px)."""
    worst, used = 0.0, set()
    for da in a.detections:
        best, bj = None, None
        for j, db in enumerate(b.detections):
            if j in used or db.class_name != da.class_name:
                continue
            d = max(abs(da.bbox.x1 - db.bbox.x1), abs(da.bbox.y1 - db.bbox.y1),
                    abs(da.bbox.x2 - db.bbox.x2), abs(da.bbox.y2 - db.bbox.y2))
            if best is None or d < best:
                best, bj = d, j
        if bj is None:
            return float("inf")
        used.add(bj)
        worst = max(worst, best)
    return worst


if __name__ == "__main__":
    sys.exit(main())
