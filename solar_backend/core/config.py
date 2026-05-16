"""Runtime configuration for the solar planning backend."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    app_name: str = "AI Solar Planner API"
    app_version: str = "1.0.0"
    model_name: str = os.getenv("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    model_candidates_raw: str = os.getenv("QWEN_VL_MODEL_CANDIDATES", "")
    model_revision: str | None = os.getenv("QWEN_VL_REVISION")
    model_local_files_only: bool = _env_bool("QWEN_VL_LOCAL_FILES_ONLY", False)
    model_load_timeout_seconds: int = _env_int("QWEN_VL_MODEL_LOAD_TIMEOUT_SECONDS", 30)
    model_load_etag_timeout_seconds: int = _env_int("QWEN_VL_MODEL_LOAD_ETAG_TIMEOUT_SECONDS", 10)
    model_dtype: str = os.getenv("MODEL_DTYPE", "float16")
    device: str = os.getenv("MODEL_DEVICE", "cuda")
    qwen_lora_adapter_path: str | None = os.getenv("QWEN_LORA_ADAPTER")
    qwen_lora_merge: bool = os.getenv("QWEN_LORA_MERGE", "false").lower() in {"1", "true", "yes"}
    sam2_model_config: str = os.getenv("SAM2_MODEL_CONFIG", "sam2_hiera_l.yaml")
    sam2_checkpoint_path: str = os.getenv("SAM2_CHECKPOINT_PATH", "sam2_hiera_large.pt")
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "outputs"))
    static_url_prefix: str = "/static"
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "30"))
    default_meters_per_pixel: float = float(os.getenv("DEFAULT_METERS_PER_PIXEL", "0.1"))

    @cached_property
    def model_candidates(self) -> tuple[str, ...]:
        candidates: list[str] = []
        for candidate in [self.model_name, *self.model_candidates_raw.split(",")]:
            normalized = candidate.strip()
            if normalized and normalized not in candidates:
                candidates.append(normalized)
        return tuple(candidates)


settings = Settings()
settings.output_dir.mkdir(parents=True, exist_ok=True)
