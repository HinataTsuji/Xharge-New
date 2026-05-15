from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient

from solar_backend.main import app
from solar_backend.services.model_registry import ModelHandle, model_registry


client = TestClient(app)
BACKGROUND_GRAY = (90, 90, 90)


def _make_test_image_bytes() -> bytes:
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:] = BACKGROUND_GRAY
    roof = np.array([[20, 20], [300, 20], [280, 210], [30, 220]], dtype=np.int32)
    cv2.fillPoly(img, [roof], color=(180, 180, 180))
    cv2.rectangle(img, (140, 100), (180, 140), color=(35, 35, 35), thickness=-1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "AI Solar Planner API"


def test_model_info_endpoint_with_mocked_model(monkeypatch) -> None:
    monkeypatch.setattr(
        model_registry,
        "load_qwen_vl",
        lambda: ModelHandle(name="Qwen/Qwen2.5-VL-3B-Instruct", model=None, processor=None, device="cpu", ready=False),
    )

    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert "comparison" in body["recommendations"]


def test_analyze_roof_endpoint_returns_standardized_payload() -> None:
    image_bytes = _make_test_image_bytes()
    response = client.post(
        "/analyze-roof",
        files={"image": ("roof.png", image_bytes, "image/png")},
        data={"meters_per_pixel": "0.1", "backend": "classical"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["data"]["roof_detected"] is True
    assert body["data"]["usable_area_m2"] >= 0
    assert isinstance(body["data"]["obstacles"], list)
    assert body["data"]["mask_overlay_url"].startswith("/static/")


def test_estimate_panels_endpoint() -> None:
    payload = {
        "image_width": 320,
        "image_height": 240,
        "roof_polygon": [
            {"x": 20, "y": 20},
            {"x": 300, "y": 20},
            {"x": 300, "y": 220},
            {"x": 20, "y": 220},
        ],
        "obstacles": [
            {
                "label": "ac",
                "confidence": 0.9,
                "bbox": [140, 90, 40, 40],
                "polygon": [],
            }
        ],
        "meters_per_pixel": 0.1,
        "panel_config": {
            "panel_width_m": 1.0,
            "panel_height_m": 1.5,
            "row_spacing_m": 0.2,
            "col_spacing_m": 0.2,
            "panel_power_kw": 0.5,
        },
        "annual_yield_factor_kwh_per_kw": 1400,
    }
    response = client.post("/estimate-panels", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["data"]["estimated_panel_count"] >= 0
    assert body["data"]["estimated_power_kw"] >= 0
    assert body["data"]["panel_layout_overlay_url"].startswith("/static/")
