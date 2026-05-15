"""Service for panel layout estimation and overlay rendering."""
from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import cv2

from solar_backend.core.config import settings
from solar_backend.pipelines.placement import pack_panels
from solar_backend.pipelines.postprocess import compute_usable_mask, obstacles_to_mask
from solar_backend.pipelines.visualization import draw_panel_layout_overlay
from solar_backend.utils.geometry import polygon_to_mask
from solar_backend.utils.image import blank_canvas, save_png


@dataclass
class PanelEstimateOutput:
    estimated_panel_count: int
    estimated_power_kw: float
    estimated_annual_energy_kwh: float
    used_area_m2: float
    panel_layout: list[dict]
    overlay_url: str


class PanelEstimationService:
    """Coordinates panel packing and layout visualization."""

    def estimate(
        self,
        image_width: int,
        image_height: int,
        roof_polygon: list[tuple[float, float]],
        obstacles: list[dict],
        meters_per_pixel: float,
        panel_width_m: float,
        panel_height_m: float,
        row_spacing_m: float,
        col_spacing_m: float,
        panel_power_kw: float,
        annual_yield_factor_kwh_per_kw: float,
    ) -> PanelEstimateOutput:
        roof_mask = polygon_to_mask(roof_polygon, image_width, image_height)
        obstacle_mask = obstacles_to_mask(obstacles=obstacles, width=image_width, height=image_height)
        usable_mask = compute_usable_mask(roof_mask=roof_mask, obstacle_mask=obstacle_mask)

        placement = pack_panels(
            usable_mask=usable_mask,
            meters_per_pixel=meters_per_pixel,
            panel_width_m=panel_width_m,
            panel_height_m=panel_height_m,
            row_spacing_m=row_spacing_m,
            col_spacing_m=col_spacing_m,
            panel_power_kw=panel_power_kw,
            annual_yield_factor_kwh_per_kw=annual_yield_factor_kwh_per_kw,
        )

        canvas = blank_canvas(image_width, image_height)
        roof_tint = cv2.cvtColor(usable_mask, cv2.COLOR_GRAY2BGR)
        canvas = cv2.addWeighted(canvas, 0.7, roof_tint, 0.3, 0)
        layout_overlay = draw_panel_layout_overlay(canvas, placement.panels)

        file_name = f"panel_layout_{uuid4().hex}.png"
        output_path = settings.output_dir / file_name
        save_png(layout_overlay, output_path)

        return PanelEstimateOutput(
            estimated_panel_count=placement.panel_count,
            estimated_power_kw=placement.estimated_power_kw,
            estimated_annual_energy_kwh=placement.estimated_annual_energy_kwh,
            used_area_m2=placement.used_area_m2,
            panel_layout=placement.panels,
            overlay_url=f"{settings.static_url_prefix}/{file_name}",
        )


panel_estimation_service = PanelEstimationService()
