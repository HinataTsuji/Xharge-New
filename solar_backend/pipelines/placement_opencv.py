"""OpenCV-accelerated panel packing strategy."""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

# Keep parity with the legacy optimizer's 6x6 offset trial heuristic.
from solar_backend.pipelines.placement import PlacementResult

PLACEMENT_OFFSET_TRIALS = 6


def _window_sum(integral: np.ndarray, x: int, y: int, w: int, h: int) -> int:
    x2 = x + w
    y2 = y + h
    return int(integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x])


def pack_panels_opencv(
    usable_mask: np.ndarray,
    meters_per_pixel: float,
    panel_width_m: float,
    panel_height_m: float,
    row_spacing_m: float,
    col_spacing_m: float,
    panel_power_kw: float,
    annual_yield_factor_kwh_per_kw: float,
) -> PlacementResult:
    """Pack panels using integral-image accelerated occupancy checks."""
    h, w = usable_mask.shape
    panel_w_px = max(1, int(round(panel_width_m / meters_per_pixel)))
    panel_h_px = max(1, int(round(panel_height_m / meters_per_pixel)))
    row_gap_px = max(0, int(round(row_spacing_m / meters_per_pixel)))
    col_gap_px = max(0, int(round(col_spacing_m / meters_per_pixel)))
    step_x = max(panel_w_px + col_gap_px, 1)
    step_y = max(panel_h_px + row_gap_px, 1)

    if panel_w_px > w or panel_h_px > h:
        return PlacementResult(0, 0.0, 0.0, 0.0, [])

    usable_binary = (usable_mask > 0).astype(np.uint8)
    usable_integral = cv2.integral(usable_binary)
    occupancy = np.zeros_like(usable_binary, dtype=np.uint8)
    best_layout: List[dict] = []

    trials = PLACEMENT_OFFSET_TRIALS
    for oy in range(trials):
        for ox in range(trials):
            placed: List[dict] = []
            occupancy.fill(0)
            offset_denominator = max(trials - 1, 1)
            x_offset = int(round((ox / offset_denominator) * step_x))
            y_offset = int(round((oy / offset_denominator) * step_y))
            for y in range(y_offset, h - panel_h_px + 1, step_y):
                for x in range(x_offset, w - panel_w_px + 1, step_x):
                    area_px = panel_w_px * panel_h_px
                    usable_px = _window_sum(usable_integral, x, y, panel_w_px, panel_h_px)
                    if usable_px != area_px:
                        continue
                    if np.any(occupancy[y : y + panel_h_px, x : x + panel_w_px] > 0):
                        continue
                    occupancy[y : y + panel_h_px, x : x + panel_w_px] = 1
                    placed.append({"x": x, "y": y, "width": panel_w_px, "height": panel_h_px})
            if len(placed) > len(best_layout):
                best_layout = placed

    panel_count = len(best_layout)
    estimated_power_kw = round(panel_count * panel_power_kw, 3)
    estimated_annual_energy_kwh = round(estimated_power_kw * annual_yield_factor_kwh_per_kw, 2)
    used_area_m2 = round(panel_count * panel_width_m * panel_height_m, 3)
    return PlacementResult(
        panel_count=panel_count,
        estimated_power_kw=estimated_power_kw,
        estimated_annual_energy_kwh=estimated_annual_energy_kwh,
        used_area_m2=used_area_m2,
        panels=best_layout,
    )
