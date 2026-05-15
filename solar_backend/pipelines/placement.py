"""Panel packing and energy estimation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class PlacementResult:
    panel_count: int
    estimated_power_kw: float
    estimated_annual_energy_kwh: float
    used_area_m2: float
    panels: List[dict]


def pack_panels(
    usable_mask: np.ndarray,
    meters_per_pixel: float,
    panel_width_m: float,
    panel_height_m: float,
    row_spacing_m: float,
    col_spacing_m: float,
    panel_power_kw: float,
    annual_yield_factor_kwh_per_kw: float,
) -> PlacementResult:
    """Pack panels via deterministic top-left raster scan grid search.

    This favors maintainability and deterministic output over expensive optimization.
    """
    h, w = usable_mask.shape
    panel_w_px = max(1, int(round(panel_width_m / meters_per_pixel)))
    panel_h_px = max(1, int(round(panel_height_m / meters_per_pixel)))
    row_gap_px = max(0, int(round(row_spacing_m / meters_per_pixel)))
    col_gap_px = max(0, int(round(col_spacing_m / meters_per_pixel)))

    step_x = panel_w_px + col_gap_px
    step_y = panel_h_px + row_gap_px

    placed: List[dict] = []
    occupied = np.zeros_like(usable_mask, dtype=np.uint8)

    for y in range(0, max(h - panel_h_px + 1, 1), max(step_y, 1)):
        for x in range(0, max(w - panel_w_px + 1, 1), max(step_x, 1)):
            window = usable_mask[y : y + panel_h_px, x : x + panel_w_px]
            if window.shape[0] != panel_h_px or window.shape[1] != panel_w_px:
                continue
            if np.count_nonzero(window) != panel_w_px * panel_h_px:
                continue

            occ_window = occupied[y : y + panel_h_px, x : x + panel_w_px]
            if np.any(occ_window > 0):
                continue

            occupied[y : y + panel_h_px, x : x + panel_w_px] = 255
            placed.append(
                {
                    "x": x,
                    "y": y,
                    "width": panel_w_px,
                    "height": panel_h_px,
                }
            )

    panel_count = len(placed)
    estimated_power_kw = round(panel_count * panel_power_kw, 3)
    estimated_annual_energy_kwh = round(estimated_power_kw * annual_yield_factor_kwh_per_kw, 2)
    used_area_m2 = round(panel_count * panel_width_m * panel_height_m, 3)

    return PlacementResult(
        panel_count=panel_count,
        estimated_power_kw=estimated_power_kw,
        estimated_annual_energy_kwh=estimated_annual_energy_kwh,
        used_area_m2=used_area_m2,
        panels=placed,
    )
