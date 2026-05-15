"""Runtime configuration for the solar planning backend."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    app_name: str = "AI Solar Planner API"
    app_version: str = "1.0.0"
    model_name: str = os.getenv("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    model_dtype: str = os.getenv("MODEL_DTYPE", "float16")
    device: str = os.getenv("MODEL_DEVICE", "cuda")
    qwen_lora_adapter_path: str | None = os.getenv("QWEN_LORA_ADAPTER")
    qwen_lora_merge: bool = os.getenv("QWEN_LORA_MERGE", "false").lower() in {"1", "true", "yes"}
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "outputs"))
    static_url_prefix: str = "/static"
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "30"))
    default_meters_per_pixel: float = float(os.getenv("DEFAULT_METERS_PER_PIXEL", "0.1"))


settings = Settings()
settings.output_dir.mkdir(parents=True, exist_ok=True)
