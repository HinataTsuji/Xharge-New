"""Polygon extraction and sanitation helpers for roof masks and model outputs."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from solar_backend.utils.geometry import mask_to_largest_polygon

Point = Tuple[float, float]
POLYGON_COORD_PRECISION = 3


def clamp_polygon(points: Sequence[Point], width: int, height: int) -> List[Point]:
    """Clamp polygon points to image bounds and remove duplicates."""
    if width <= 0 or height <= 0:
        return []
    deduped: List[Point] = []
    for x, y in points:
        cx = float(min(max(x, 0.0), width - 1))
        cy = float(min(max(y, 0.0), height - 1))
        point = (round(cx, POLYGON_COORD_PRECISION), round(cy, POLYGON_COORD_PRECISION))
        if not deduped or deduped[-1] != point:
            deduped.append(point)
    if len(deduped) >= 2 and deduped[0] == deduped[-1]:
        deduped.pop()
    return deduped


def parse_polygon_points(raw_polygon: Iterable[object], width: int, height: int) -> List[Point]:
    """Parse raw polygon points from model output and clamp to bounds."""
    points: List[Point] = []
    for p in raw_polygon:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            try:
                points.append((float(p[0]), float(p[1])))
            except (TypeError, ValueError):
                continue
    return clamp_polygon(points, width=width, height=height)


def polygon_from_mask(mask: np.ndarray, min_area: float = 400.0) -> List[Point]:
    """Extract dominant polygon from binary mask."""
    return clamp_polygon(mask_to_largest_polygon(mask, min_area=min_area), width=mask.shape[1], height=mask.shape[0])
