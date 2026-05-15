"""Analysis endpoints for roof and panel estimation."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from solar_backend.core.config import settings
from solar_backend.schemas.api import (
    AnalyzeRoofData,
    AnalyzeRoofResponse,
    EstimatePanelsData,
    EstimatePanelsRequest,
    EstimatePanelsResponse,
    Obstacle,
    PanelPlacement,
    Point,
)
from solar_backend.services.panel_estimation_service import panel_estimation_service
from solar_backend.services.roof_analysis_service import roof_analysis_service
from solar_backend.utils.image import ALLOWED_IMAGE_TYPES, decode_image_bytes

router = APIRouter(tags=["analysis"])


@router.post("/analyze-roof", response_model=AnalyzeRoofResponse)
async def analyze_roof(
    image: UploadFile = File(...),
    meters_per_pixel: float = Form(default=settings.default_meters_per_pixel),
    backend: str = Form(default="auto"),
) -> AnalyzeRoofResponse:
    """Analyze uploaded rooftop image and return roof/usable area results."""
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported image content type")

    image_bytes = await image.read()
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise HTTPException(status_code=413, detail=f"Image exceeds {settings.max_upload_mb} MB limit")

    try:
        image_bgr = decode_image_bytes(image_bytes)
        output = roof_analysis_service.analyze(image_bgr=image_bgr, meters_per_pixel=meters_per_pixel, backend=backend)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to analyze roof: {exc}") from exc

    return AnalyzeRoofResponse(
        data=AnalyzeRoofData(
            roof_detected=output.roof_detected,
            confidence=output.confidence,
            roof_polygon=[Point(x=x, y=y) for x, y in output.roof_polygon],
            roof_area_px=output.roof_area_px,
            roof_area_m2=output.roof_area_m2,
            usable_area_px=output.usable_area_px,
            usable_area_m2=output.usable_area_m2,
            obstacles=[
                Obstacle(
                    label=o.get("label", "obstacle"),
                    confidence=float(o.get("confidence", 0.5)),
                    bbox=[float(v) for v in o.get("bbox", [0, 0, 0, 0])],
                    polygon=[Point(x=p[0], y=p[1]) for p in o.get("polygon", [])],
                )
                for o in output.obstacles
            ],
            mask_overlay_url=output.overlay_url,
        )
    )


@router.post("/estimate-panels", response_model=EstimatePanelsResponse)
def estimate_panels(payload: EstimatePanelsRequest) -> EstimatePanelsResponse:
    """Estimate panel arrangement and generation from geometry inputs."""
    roof_polygon = [(p.x, p.y) for p in payload.roof_polygon]
    obstacles = [
        {
            "label": obs.label,
            "confidence": obs.confidence,
            "bbox": obs.bbox,
            "polygon": [(p.x, p.y) for p in obs.polygon],
        }
        for obs in payload.obstacles
    ]

    output = panel_estimation_service.estimate(
        image_width=payload.image_width,
        image_height=payload.image_height,
        roof_polygon=roof_polygon,
        obstacles=obstacles,
        meters_per_pixel=payload.meters_per_pixel,
        panel_width_m=payload.panel_config.panel_width_m,
        panel_height_m=payload.panel_config.panel_height_m,
        row_spacing_m=payload.panel_config.row_spacing_m,
        col_spacing_m=payload.panel_config.col_spacing_m,
        panel_power_kw=payload.panel_config.panel_power_kw,
        annual_yield_factor_kwh_per_kw=payload.annual_yield_factor_kwh_per_kw,
    )

    return EstimatePanelsResponse(
        data=EstimatePanelsData(
            estimated_panel_count=output.estimated_panel_count,
            estimated_power_kw=output.estimated_power_kw,
            estimated_annual_energy_kwh=output.estimated_annual_energy_kwh,
            used_area_m2=output.used_area_m2,
            panel_layout=[PanelPlacement(**panel) for panel in output.panel_layout],
            panel_layout_overlay_url=output.overlay_url,
        )
    )
