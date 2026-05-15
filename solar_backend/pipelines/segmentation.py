"""Roof and obstacle segmentation logic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from solar_backend.core.exceptions import InferenceError
from solar_backend.pipelines.preprocess import preprocess_for_segmentation
from solar_backend.services.model_registry import model_registry
from solar_backend.utils.geometry import bbox_from_mask, mask_to_largest_polygon, polygon_to_mask


@dataclass
class SegmentationResult:
    roof_mask: np.ndarray
    roof_polygon: List[Tuple[float, float]]
    roof_confidence: float
    obstacles: List[dict]


def _classical_roof_segmentation(image_bgr: np.ndarray) -> tuple[np.ndarray, List[Tuple[float, float]], float]:
    denoised = preprocess_for_segmentation(image_bgr)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    edges = cv2.dilate(edges, np.ones((5, 5), dtype=np.uint8), iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8), iterations=2)

    roof_polygon = mask_to_largest_polygon(edges, min_area=400.0)
    if not roof_polygon:
        h, w = gray.shape
        roof_polygon = [
            (0.05 * w, 0.1 * h),
            (0.95 * w, 0.1 * h),
            (0.95 * w, 0.9 * h),
            (0.05 * w, 0.9 * h),
        ]
        confidence = 0.35
    else:
        confidence = 0.7

    roof_mask = polygon_to_mask(roof_polygon, gray.shape[1], gray.shape[0])
    return roof_mask, roof_polygon, confidence


def _qwen_roof_segmentation(image_bgr: np.ndarray) -> tuple[np.ndarray, List[Tuple[float, float]], float]:
    prompt = (
        "Return JSON only with keys: roof_detected (bool), confidence (0-1), "
        "roof_polygon (list of [x,y] absolute pixel points)."
    )
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = model_registry.infer_qwen_structured(prompt=prompt, images=[image_rgb], max_new_tokens=200)[0]

    raw_polygon = result.get("roof_polygon", [])
    points: List[Tuple[float, float]] = []
    for p in raw_polygon:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            points.append((float(p[0]), float(p[1])))
    if len(points) < 3:
        raise InferenceError("Qwen response did not include a valid roof polygon")
    h, w = image_bgr.shape[:2]
    roof_mask = polygon_to_mask(points, w, h)
    confidence = float(result.get("confidence", 0.75))
    return roof_mask, points, max(0.0, min(1.0, confidence))


def detect_obstacles(image_bgr: np.ndarray, roof_mask: np.ndarray) -> List[dict]:
    """Detect common rooftop obstacles inside the roof region.

    Uses blob analysis over high-contrast structures in the roof ROI.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    roof_gray = cv2.bitwise_and(gray, gray, mask=roof_mask)
    normalized = cv2.normalize(roof_gray, None, 0, 255, cv2.NORM_MINMAX)
    thresh = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        -5,
    )
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obstacles: List[dict] = []

    roof_area = max(float(np.count_nonzero(roof_mask)), 1.0)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 80 or area > roof_area * 0.12:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        candidate_mask = np.zeros_like(roof_mask)
        cv2.drawContours(candidate_mask, [contour], -1, 255, thickness=-1)
        overlap = cv2.bitwise_and(candidate_mask, roof_mask)
        if np.count_nonzero(overlap) / max(np.count_nonzero(candidate_mask), 1) < 0.7:
            continue

        polygon = [(float(pt[0][0]), float(pt[0][1])) for pt in cv2.approxPolyDP(contour, 1.5, True)]
        obstacles.append(
            {
                "label": "obstacle",
                "confidence": 0.6,
                "bbox": [float(x), float(y), float(w), float(h)],
                "polygon": polygon,
            }
        )

    return obstacles


def segment_roof_and_obstacles(image_bgr: np.ndarray, backend: str = "auto") -> SegmentationResult:
    """Run roof and obstacle segmentation with backend fallback chain."""
    roof_mask: np.ndarray
    roof_polygon: List[Tuple[float, float]]
    roof_confidence: float

    if backend in {"auto", "qwen_vl"}:
        try:
            roof_mask, roof_polygon, roof_confidence = _qwen_roof_segmentation(image_bgr)
        except Exception:
            roof_mask, roof_polygon, roof_confidence = _classical_roof_segmentation(image_bgr)
    else:
        roof_mask, roof_polygon, roof_confidence = _classical_roof_segmentation(image_bgr)

    obstacles = detect_obstacles(image_bgr, roof_mask)
    return SegmentationResult(
        roof_mask=roof_mask,
        roof_polygon=roof_polygon,
        roof_confidence=roof_confidence,
        obstacles=obstacles,
    )
