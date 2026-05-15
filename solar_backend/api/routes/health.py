"""Health and metadata endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from solar_backend.core.config import settings
from solar_backend.schemas.api import HealthResponse, ModelInfoResponse
from solar_backend.services.model_registry import model_registry

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness endpoint for orchestration and monitoring systems."""
    return HealthResponse(service=settings.app_name, version=settings.app_version)


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Return loaded model details and architecture recommendations."""
    qwen_handle = model_registry.load_qwen_vl()
    return ModelInfoResponse(
        model={
            "primary_model": qwen_handle.name,
            "device": qwen_handle.device,
            "qwen_ready": qwen_handle.ready,
            "supported_backends": ["qwen_vl", "sam2", "sam2_qwen", "yolov8_seg", "detectron2", "classical_cv"],
            "placement_backends": ["raster", "opencv"],
        },
        recommendations={
            "comparison": {
                "qwen_vl": "Best for multimodal reasoning and weakly supervised polygon extraction; not optimized for pixel-accurate masks alone.",
                "sam2": "Strong zero-shot segmentation masks; combine with prompts from Qwen for precision.",
                "yolov8_seg": "Fast production inference for roof/obstacle instance segmentation when labels are available.",
                "detectron2": "Most flexible for custom instance/semantic segmentation pipelines but heavier ops burden.",
            },
            "fine_tuning": {
                "qwen_lora": "Use LoRA/QLoRA with instruction-style image-text annotations and JSON polygon targets.",
                "datasets": [
                    "Inria Aerial Image Labeling",
                    "SpaceNet (building footprints)",
                    "xView2",
                    "Massachusetts Buildings Dataset",
                    "DeepRoof and local drone roof datasets with obstacle labels",
                ],
                "annotation_formats": ["COCO polygons", "COCO instance masks", "YOLOv8 segmentation", "GeoJSON footprints"],
                "augmentation": ["random shadows", "perspective jitter", "color shifts", "seasonal/weather transforms"],
            },
        },
    )
