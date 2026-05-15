"""Geometry utilities used by segmentation and panel packing."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


PointTuple = Tuple[float, float]


def polygon_area(points: Sequence[PointTuple]) -> float:
    """Compute polygon area in pixels using OpenCV contour area."""
    if len(points) < 3:
        return 0.0
    contour = np.array(points, dtype=np.float32)
    return float(abs(cv2.contourArea(contour)))


def polygon_to_mask(points: Sequence[PointTuple], width: int, height: int) -> np.ndarray:
    """Rasterize polygon points into a binary uint8 mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) < 3:
        return mask
    contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [contour], color=255)
    return mask


def mask_to_largest_polygon(mask: np.ndarray, min_area: float = 250.0) -> List[PointTuple]:
    """Extract the largest polygon contour from a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return []
    epsilon = 0.01 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    return [(float(pt[0][0]), float(pt[0][1])) for pt in approx]


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Compute [x, y, width, height] for non-zero mask regions."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def to_int_points(points: Iterable[PointTuple]) -> np.ndarray:
    """Convert points iterable into OpenCV contour int array."""
    return np.array([(int(round(x)), int(round(y))) for x, y in points], dtype=np.int32).reshape((-1, 1, 2))
