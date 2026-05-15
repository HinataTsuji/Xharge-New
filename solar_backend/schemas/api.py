"""Pydantic schemas for request/response contracts."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class Point(BaseModel):
    x: float
    y: float


class Obstacle(BaseModel):
    label: str = "obstacle"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    bbox: List[float] = Field(description="[x, y, width, height] in pixels", min_length=4, max_length=4)
    polygon: List[Point] = Field(default_factory=list)


class AnalyzeRoofData(BaseModel):
    roof_detected: bool
    confidence: float = Field(ge=0.0, le=1.0)
    roof_polygon: List[Point]
    roof_area_px: float
    roof_area_m2: float
    usable_area_px: float
    usable_area_m2: float
    obstacles: List[Obstacle]
    mask_overlay_url: Optional[str] = None


class AnalyzeRoofResponse(BaseModel):
    status: Literal["success"] = "success"
    data: AnalyzeRoofData


class PanelConfig(BaseModel):
    panel_width_m: float = Field(default=1.134, gt=0)
    panel_height_m: float = Field(default=2.278, gt=0)
    row_spacing_m: float = Field(default=0.1, ge=0)
    col_spacing_m: float = Field(default=0.1, ge=0)
    panel_power_kw: float = Field(default=0.62, gt=0)


class PanelPlacement(BaseModel):
    x: int
    y: int
    width: int
    height: int


class EstimatePanelsRequest(BaseModel):
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    roof_polygon: List[Point] = Field(min_length=3)
    obstacles: List[Obstacle] = Field(default_factory=list)
    meters_per_pixel: float = Field(default=0.1, gt=0)
    panel_config: PanelConfig = Field(default_factory=PanelConfig)
    annual_yield_factor_kwh_per_kw: float = Field(
        default=1450.0,
        gt=0,
        description="Site-specific annual energy factor",
    )

    @model_validator(mode="after")
    def ensure_image_size_reasonable(self) -> "EstimatePanelsRequest":
        if self.image_width > 20000 or self.image_height > 20000:
            raise ValueError("Image dimensions are too large for panel estimation")
        return self


class EstimatePanelsData(BaseModel):
    estimated_panel_count: int
    estimated_power_kw: float
    estimated_annual_energy_kwh: float
    used_area_m2: float
    panel_layout: List[PanelPlacement]
    panel_layout_overlay_url: Optional[str] = None


class EstimatePanelsResponse(BaseModel):
    status: Literal["success"] = "success"
    data: EstimatePanelsData


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    message: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str
    version: str


class ModelInfoResponse(BaseModel):
    status: Literal["success"] = "success"
    model: dict
    recommendations: dict
