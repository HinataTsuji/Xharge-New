"""Visualization overlays for segmentation and panel layout."""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from solar_backend.utils.geometry import to_int_points


def draw_segmentation_overlay(
    image_bgr: np.ndarray,
    roof_polygon: Iterable[tuple[float, float]],
    obstacle_polygons: Iterable[Iterable[tuple[float, float]]],
    usable_mask: np.ndarray,
) -> np.ndarray:
    """Render roof, obstacles, and usable mask overlay."""
    overlay = image_bgr.copy()

    roof_points = list(roof_polygon)
    if len(roof_points) >= 3:
        cv2.polylines(overlay, [to_int_points(roof_points)], True, color=(0, 200, 0), thickness=3)

    usable_tint = np.zeros_like(overlay)
    usable_tint[:, :] = (0, 180, 0)
    alpha = (usable_mask > 0).astype(np.uint8) * 120
    overlay = np.where(alpha[..., None] > 0, cv2.addWeighted(overlay, 0.7, usable_tint, 0.3, 0), overlay)

    for poly in obstacle_polygons:
        points = list(poly)
        if len(points) >= 3:
            contour = to_int_points(points)
            cv2.fillPoly(overlay, [contour], color=(0, 0, 255))
            cv2.polylines(overlay, [contour], True, color=(0, 255, 255), thickness=2)

    return overlay


def draw_panel_layout_overlay(image_bgr: np.ndarray, panel_layout: Iterable[dict]) -> np.ndarray:
    """Render panel placement rectangles on top of the image."""
    overlay = image_bgr.copy()
    for panel in panel_layout:
        x = int(panel["x"])
        y = int(panel["y"])
        w = int(panel["width"])
        h = int(panel["height"])
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color=(255, 140, 0), thickness=2)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color=(255, 140, 0), thickness=-1)
    return cv2.addWeighted(image_bgr, 0.65, overlay, 0.35, 0)
