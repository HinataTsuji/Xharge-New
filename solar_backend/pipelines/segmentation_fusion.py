"""Fusion backend combining Qwen-VL prompts with SAM2-style refinement."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from solar_backend.pipelines.segmentation_qwen import segment_roof_with_qwen
from solar_backend.pipelines.segmentation_sam2 import refine_roof_with_sam2


def segment_roof_with_qwen_sam2(image_bgr: np.ndarray) -> tuple[np.ndarray, List[Tuple[float, float]], float]:
    """Run Qwen prompt extraction then refine with SAM2-compatible refinement."""
    _, qwen_polygon, qwen_conf = segment_roof_with_qwen(image_bgr)
    refined_mask, refined_polygon, sam_conf = refine_roof_with_sam2(image_bgr, qwen_polygon)
    return refined_mask, refined_polygon, float((qwen_conf + sam_conf) / 2.0)
