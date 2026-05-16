from __future__ import annotations

import sys
import types
from contextlib import contextmanager

import cv2
import numpy as np
from fastapi.testclient import TestClient

from solar_backend.core.config import settings
from solar_backend.main import app
from solar_backend.services.model_registry import ModelHandle, model_registry


client = TestClient(app)
BACKGROUND_GRAY = (90, 90, 90)


@contextmanager
def _override_settings(**overrides):
    originals = {name: getattr(settings, name) for name in overrides}
    for name, value in overrides.items():
        object.__setattr__(settings, name, value)
    try:
        yield
    finally:
        for name, value in originals.items():
            object.__setattr__(settings, name, value)


def _install_fake_model_modules(monkeypatch, model_loader, processor_loader) -> None:
    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = types.SimpleNamespace(from_pretrained=processor_loader)
    fake_transformers.Qwen2_5_VLForConditionalGeneration = types.SimpleNamespace(from_pretrained=model_loader)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


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


def test_model_registry_falls_back_to_second_candidate_and_model_info_reports_loaded_name(monkeypatch) -> None:
    calls: list[tuple[str, str, dict]] = []

    class _LoadedModel:
        def to(self, _device: str):
            return self

    def _model_from_pretrained(name: str, **kwargs):
        calls.append(("model", name, kwargs))
        if name == "bad/model":
            raise OSError("404 not found")
        return _LoadedModel()

    def _processor_from_pretrained(name: str, **kwargs):
        calls.append(("processor", name, kwargs))
        return object()

    _install_fake_model_modules(monkeypatch, _model_from_pretrained, _processor_from_pretrained)
    with _override_settings(
        model_name="bad/model",
        model_candidates_raw="good/model",
        model_revision="main",
        model_local_files_only=False,
        model_load_timeout_seconds=31,
        model_load_etag_timeout_seconds=11,
        device="cpu",
        model_dtype="float16",
        qwen_lora_adapter_path=None,
    ):
        model_registry._qwen = None
        handle = model_registry.load_qwen_vl()
        assert handle.ready is True
        assert handle.name == "good/model"
        assert any(kind == "model" and name == "bad/model" for kind, name, _ in calls)
        assert any(kind == "model" and name == "good/model" for kind, name, _ in calls)
        assert any(
            kind == "model" and name == "good/model" and kwargs.get("revision") == "main"
            for kind, name, kwargs in calls
        )

        response = client.get("/model-info")
        assert response.status_code == 200
        body = response.json()
        assert body["model"]["primary_model"] == "good/model"
    model_registry._qwen = None


def test_model_registry_all_candidates_fail_sets_qwen_unready_and_api_falls_back_to_classical(monkeypatch) -> None:
    def _model_from_pretrained(_name: str, **_kwargs):
        raise TimeoutError("timed out")

    def _processor_from_pretrained(_name: str, **_kwargs):
        return object()

    _install_fake_model_modules(monkeypatch, _model_from_pretrained, _processor_from_pretrained)
    with _override_settings(
        model_name="bad/primary",
        model_candidates_raw="bad/secondary",
        model_revision=None,
        model_local_files_only=False,
        model_load_timeout_seconds=5,
        model_load_etag_timeout_seconds=5,
        device="cpu",
        model_dtype="float16",
        qwen_lora_adapter_path=None,
    ):
        model_registry._qwen = None
        handle = model_registry.load_qwen_vl()
        assert handle.ready is False

        model_info_response = client.get("/model-info")
        assert model_info_response.status_code == 200
        assert model_info_response.json()["model"]["qwen_ready"] is False

        image_bytes = _make_test_image_bytes()
        analyze_response = client.post(
            "/analyze-roof",
            files={"image": ("roof.png", image_bytes, "image/png")},
            data={"meters_per_pixel": "0.1", "backend": "qwen_vl"},
        )
        assert analyze_response.status_code == 200
        assert analyze_response.json()["data"]["roof_detected"] is True
    model_registry._qwen = None


def test_model_registry_respects_local_files_only_and_retries_without_safetensors(monkeypatch) -> None:
    model_calls: list[dict] = []
    processor_calls: list[dict] = []

    class _LoadedModel:
        def to(self, _device: str):
            return self

    def _model_from_pretrained(_name: str, **kwargs):
        model_calls.append(kwargs)
        if len(model_calls) == 1:
            raise OSError("safetensors weights not found")
        return _LoadedModel()

    def _processor_from_pretrained(_name: str, **kwargs):
        processor_calls.append(kwargs)
        return object()

    _install_fake_model_modules(monkeypatch, _model_from_pretrained, _processor_from_pretrained)
    with _override_settings(
        model_name="local/model",
        model_candidates_raw="",
        model_revision="rev-123",
        model_local_files_only=True,
        model_load_timeout_seconds=12,
        model_load_etag_timeout_seconds=6,
        device="cpu",
        model_dtype="float16",
        qwen_lora_adapter_path=None,
    ):
        model_registry._qwen = None
        handle = model_registry.load_qwen_vl()
        assert handle.ready is True
        assert len(model_calls) == 2
        assert model_calls[0]["local_files_only"] is True
        assert model_calls[1]["use_safetensors"] is False
        assert model_calls[1]["revision"] == "rev-123"
        assert processor_calls[0]["local_files_only"] is True
        assert processor_calls[0]["revision"] == "rev-123"
    model_registry._qwen = None


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
    assert body["data"]["estimated_panel_count"] > 0
    assert len(body["data"]["panel_layout"]) == body["data"]["estimated_panel_count"]
    assert body["data"]["estimated_power_kw"] >= 0
    assert body["data"]["panel_layout_overlay_url"].startswith("/static/")


def test_estimate_panels_endpoint_opencv_backend() -> None:
    payload = {
        "image_width": 240,
        "image_height": 240,
        "roof_polygon": [
            {"x": 10, "y": 10},
            {"x": 230, "y": 10},
            {"x": 230, "y": 230},
            {"x": 10, "y": 230},
        ],
        "obstacles": [],
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
    raster_response = client.post("/estimate-panels", json=payload)
    assert raster_response.status_code == 200
    raster_count = raster_response.json()["data"]["estimated_panel_count"]

    opencv_payload = {**payload, "placement_backend": "opencv"}
    response = client.post("/estimate-panels", json=opencv_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["data"]["estimated_panel_count"] > 0
    assert len(body["data"]["panel_layout"]) == body["data"]["estimated_panel_count"]
    assert raster_count > 0


def test_estimate_panels_endpoint_handles_zero_panel_case() -> None:
    payload = {
        "image_width": 80,
        "image_height": 80,
        "roof_polygon": [
            {"x": 10, "y": 10},
            {"x": 20, "y": 10},
            {"x": 20, "y": 20},
            {"x": 10, "y": 20},
        ],
        "obstacles": [],
        "meters_per_pixel": 0.1,
        "panel_config": {
            "panel_width_m": 2.0,
            "panel_height_m": 2.0,
            "row_spacing_m": 1.0,
            "col_spacing_m": 1.0,
            "panel_power_kw": 0.5,
        },
        "annual_yield_factor_kwh_per_kw": 1400,
    }
    response = client.post("/estimate-panels", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["data"]["estimated_panel_count"] == 0
    assert body["data"]["panel_layout"] == []
