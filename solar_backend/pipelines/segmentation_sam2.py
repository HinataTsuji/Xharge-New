"""SAM2-style segmentation refinement with graceful fallback."""
from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

from solar_backend.core.config import settings
from solar_backend.core.exceptions import InferenceError
from solar_backend.pipelines.polygon_extraction import polygon_from_mask
from solar_backend.utils.geometry import polygon_to_mask

logger = logging.getLogger(__name__)
# Keep confidence away from extreme values because refinement quality is heuristic and backend-dependent.
REFINED_CONFIDENCE_LOWER_BOUND = 0.4
REFINED_CONFIDENCE_UPPER_BOUND = 0.95


def _fallback_refine_with_grabcut(image_bgr: np.ndarray, prompt_polygon: List[Tuple[float, float]]) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    prompt_mask = polygon_to_mask(prompt_polygon, w, h)
    if np.count_nonzero(prompt_mask) == 0:
        raise InferenceError("Prompt polygon does not cover any pixels")

    x, y, bw, bh = cv2.boundingRect(np.array(prompt_polygon, dtype=np.int32).reshape((-1, 1, 2)))
    rect = (max(0, x), max(0, y), max(1, bw), max(1, bh))
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[prompt_mask > 0] = cv2.GC_PR_FGD
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image_bgr, gc_mask, rect, bg_model, fg_model, 2, mode=cv2.GC_INIT_WITH_MASK)
    binary = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return cv2.bitwise_and(binary, prompt_mask)


def refine_roof_with_sam2(
    image_bgr: np.ndarray,
    prompt_polygon: List[Tuple[float, float]],
) -> tuple[np.ndarray, List[Tuple[float, float]], float]:
    """Refine a prompt polygon with SAM2 if available, otherwise OpenCV fallback."""
    if len(prompt_polygon) < 3:
        raise InferenceError("Prompt polygon requires at least 3 points")

    refined_mask: np.ndarray | None = None
    try:  # pragma: no cover - optional dependency path
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

        predictor = SAM2ImagePredictor(build_sam2(settings.sam2_model_config, settings.sam2_checkpoint_path))
        predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        point_coords = np.array(prompt_polygon, dtype=np.float32)
        point_labels = np.ones((point_coords.shape[0],), dtype=np.int32)
        masks, scores, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=True)
        best_idx = int(np.argmax(scores))
        refined_mask = (masks[best_idx].astype(np.uint8) * 255)
    except Exception as exc:
        logger.info("SAM2 unavailable; using OpenCV refinement fallback: %s", exc)
        refined_mask = _fallback_refine_with_grabcut(image_bgr, prompt_polygon)

    refined_polygon = polygon_from_mask(refined_mask, min_area=300.0)
    if len(refined_polygon) < 3:
        raise InferenceError("SAM2 refinement did not produce a valid polygon")

    prompt_mask = polygon_to_mask(prompt_polygon, refined_mask.shape[1], refined_mask.shape[0])
    overlap = cv2.bitwise_and(prompt_mask, refined_mask)
    prompt_pixels = max(np.count_nonzero(prompt_mask), 1)
    confidence = float(np.count_nonzero(overlap) / prompt_pixels)
    confidence = max(REFINED_CONFIDENCE_LOWER_BOUND, min(REFINED_CONFIDENCE_UPPER_BOUND, confidence))
    return refined_mask, refined_polygon, confidence
