"""Small pixel-space geometry helpers (no OpenCV dependency)."""

from __future__ import annotations

from typing import Tuple

import numpy as np

Point = Tuple[float, float]


def as_polygon(points) -> np.ndarray:
    poly = np.asarray(points, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise ValueError("polygon needs at least 3 (x, y) points")
    if not np.all(np.isfinite(poly)):
        raise ValueError("polygon contains non-finite coordinates")
    return poly


def point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    """Ray casting; points on the boundary may fall either way."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_centroid(poly: np.ndarray) -> Point:
    return float(poly[:, 0].mean()), float(poly[:, 1].mean())


def side_of_line(p: Point, a: Point, b: Point) -> float:
    """Signed side of point ``p`` relative to the directed line a->b (cross product)."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def rect_polygon(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float)
