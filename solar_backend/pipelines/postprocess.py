"""Postprocessing functions for area and usable-mask computation."""
from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np

from solar_backend.utils.geometry import polygon_area


def obstacles_to_mask(obstacles: Iterable[dict], width: int, height: int) -> np.ndarray:
    """Build a binary obstacle mask from obstacle polygons or bboxes."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for obs in obstacles:
        poly = obs.get("polygon") or []
        if len(poly) >= 3:
            points = [(int(round(x)), int(round(y))) for x, y in poly]
            contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [contour], color=255)
            continue

        bbox = obs.get("bbox", [0, 0, 0, 0])
        if len(bbox) == 4:
            x, y, w, h = [int(round(v)) for v in bbox]
            cv2.rectangle(mask, (x, y), (x + w, y + h), color=255, thickness=-1)
    return mask


def compute_usable_mask(roof_mask: np.ndarray, obstacle_mask: np.ndarray) -> np.ndarray:
    """Compute usable roof mask by subtracting obstacle regions."""
    return cv2.bitwise_and(roof_mask, cv2.bitwise_not(obstacle_mask))


def area_from_mask(mask: np.ndarray) -> float:
    """Compute area in pixels from binary mask."""
    return float(np.count_nonzero(mask))


def area_px_to_m2(area_px: float, meters_per_pixel: float) -> float:
    """Convert pixel area into square meters."""
    return area_px * (meters_per_pixel**2)


def roof_area_from_polygon(polygon: Iterable[Tuple[float, float]]) -> float:
    """Compute roof area in pixels from polygon points."""
    return polygon_area(list(polygon))
