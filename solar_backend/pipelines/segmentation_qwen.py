"""Qwen-VL specific roof segmentation routines."""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from solar_backend.core.exceptions import InferenceError
from solar_backend.pipelines.polygon_extraction import parse_polygon_points
from solar_backend.services.model_registry import model_registry
from solar_backend.utils.geometry import polygon_to_mask


def segment_roof_with_qwen(image_bgr: np.ndarray) -> tuple[np.ndarray, List[Tuple[float, float]], float]:
    """Infer roof polygon with Qwen2.5-VL and rasterize it to a mask."""
    prompt = (
        "Return JSON only with keys: roof_detected (bool), confidence (0-1), "
        "roof_polygon (list of [x,y] absolute pixel points)."
    )
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = model_registry.infer_qwen_structured(prompt=prompt, images=[image_rgb], max_new_tokens=200)[0]

    h, w = image_bgr.shape[:2]
    points = parse_polygon_points(result.get("roof_polygon", []), width=w, height=h)
    if len(points) < 3:
        raise InferenceError("Qwen response did not include a valid roof polygon")

    roof_mask = polygon_to_mask(points, w, h)
    confidence = float(result.get("confidence", 0.75))
    return roof_mask, points, max(0.0, min(1.0, confidence))
