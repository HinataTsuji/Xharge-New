"""Business service that orchestrates roof analysis pipeline."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import uuid4

import numpy as np

from solar_backend.core.config import settings
from solar_backend.pipelines.postprocess import (
    area_from_mask,
    area_px_to_m2,
    compute_usable_mask,
    obstacles_to_mask,
    roof_area_from_polygon,
)
from solar_backend.pipelines.segmentation import segment_roof_and_obstacles
from solar_backend.pipelines.visualization import draw_segmentation_overlay
from solar_backend.utils.image import save_png


@dataclass
class RoofAnalysisOutput:
    roof_detected: bool
    confidence: float
    roof_polygon: list[tuple[float, float]]
    roof_area_px: float
    roof_area_m2: float
    usable_area_px: float
    usable_area_m2: float
    obstacles: list[dict]
    usable_mask: np.ndarray
    overlay_url: str


class RoofAnalysisService:
    """Facade for running end-to-end roof and obstacle analysis."""

    def analyze(self, image_bgr: np.ndarray, meters_per_pixel: float, backend: str = "auto") -> RoofAnalysisOutput:
        segmentation = segment_roof_and_obstacles(image_bgr=image_bgr, backend=backend)

        height, width = image_bgr.shape[:2]
        obstacle_mask = obstacles_to_mask(segmentation.obstacles, width=width, height=height)
        usable_mask = compute_usable_mask(segmentation.roof_mask, obstacle_mask)

        roof_area_px = roof_area_from_polygon(segmentation.roof_polygon)
        usable_area_px = area_from_mask(usable_mask)

        overlay = draw_segmentation_overlay(
            image_bgr=image_bgr,
            roof_polygon=segmentation.roof_polygon,
            obstacle_polygons=[o.get("polygon", []) for o in segmentation.obstacles],
            usable_mask=usable_mask,
        )
        file_name = f"roof_overlay_{uuid4().hex}.png"
        output_path = settings.output_dir / file_name
        save_png(overlay, output_path)

        return RoofAnalysisOutput(
            roof_detected=len(segmentation.roof_polygon) >= 3,
            confidence=segmentation.roof_confidence,
            roof_polygon=segmentation.roof_polygon,
            roof_area_px=round(roof_area_px, 3),
            roof_area_m2=round(area_px_to_m2(roof_area_px, meters_per_pixel), 3),
            usable_area_px=round(usable_area_px, 3),
            usable_area_m2=round(area_px_to_m2(usable_area_px, meters_per_pixel), 3),
            obstacles=segmentation.obstacles,
            usable_mask=usable_mask,
            overlay_url=f"{settings.static_url_prefix}/{file_name}",
        )

    async def analyze_async(self, image_bgr: np.ndarray, meters_per_pixel: float, backend: str = "auto") -> RoofAnalysisOutput:
        """Run analysis in a worker thread to avoid blocking the event loop."""
        return await asyncio.to_thread(self.analyze, image_bgr=image_bgr, meters_per_pixel=meters_per_pixel, backend=backend)


roof_analysis_service = RoofAnalysisService()
