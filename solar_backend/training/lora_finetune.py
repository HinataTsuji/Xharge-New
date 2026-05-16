"""LoRA fine-tuning entrypoint for Qwen2.5-VL."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from solar_backend.core.config import settings


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_model_candidates(model_name: str | None, model_candidates: str) -> list[str]:
    candidates: list[str] = []
    primary = (model_name or settings.model_name).strip()
    if primary:
        candidates.append(primary)
    for candidate in model_candidates.split(","):
        normalized = candidate.strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    return candidates

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen2.5-VL roof extraction tasks")
    parser.add_argument("--model-name", default=None, help="Primary base model name/path; defaults to QWEN_VL_MODEL")
    parser.add_argument(
        "--model-candidates",
        default=settings.model_candidates_raw,
        help="Comma-separated fallback model names/paths (QWEN_VL_MODEL_CANDIDATES)",
    )
    parser.add_argument("--model-revision", default=settings.model_revision, help="Optional model revision pin")
    parser.add_argument(
        "--model-local-files-only",
        default=settings.model_local_files_only,
        type=_str_to_bool,
        help="Load only from local cache/files (QWEN_VL_LOCAL_FILES_ONLY)",
    )
    parser.add_argument(
        "--model-load-timeout-seconds",
        type=int,
        default=settings.model_load_timeout_seconds,
        help="HF Hub download timeout seconds (QWEN_VL_MODEL_LOAD_TIMEOUT_SECONDS)",
    )
    parser.add_argument(
        "--model-load-etag-timeout-seconds",
        type=int,
        default=settings.model_load_etag_timeout_seconds,
        help="HF Hub metadata timeout seconds (QWEN_VL_MODEL_LOAD_ETAG_TIMEOUT_SECONDS)",
    )
    parser.add_argument("--dataset-path", type=Path, required=True, help="Path to JSONL multimodal training data")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/qwen-lora"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(args.model_load_timeout_seconds)
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(args.model_load_etag_timeout_seconds)

    # Lazy imports keep runtime dependencies optional for inference-only environments.
    import torch  # type: ignore
    from peft import LoraConfig, get_peft_model  # type: ignore
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # type: ignore

    model = None
    processor = None
    selected_model_name = None
    candidates = _resolve_model_candidates(args.model_name, args.model_candidates)
    model_kwargs = {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else None,
        "local_files_only": bool(args.model_local_files_only),
    }
    processor_kwargs = {"local_files_only": bool(args.model_local_files_only)}
    if args.model_revision:
        model_kwargs["revision"] = args.model_revision
        processor_kwargs["revision"] = args.model_revision

    failures: list[str] = []
    for candidate in candidates:
        try:
            try:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(candidate, **model_kwargs)
            except Exception as exc:
                lowered = str(exc).lower()
                if "safetensors" in lowered and ("not found" in lowered or "missing" in lowered or "unable to" in lowered):
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        candidate,
                        use_safetensors=False,
                        **model_kwargs,
                    )
                else:
                    raise
            processor = AutoProcessor.from_pretrained(candidate, **processor_kwargs)
            selected_model_name = candidate
            break
        except Exception as exc:
            failures.append(f"{candidate}: {exc.__class__.__name__}: {exc}")

    if model is None or processor is None or selected_model_name is None:
        joined = "; ".join(failures) if failures else "No model candidates configured"
        raise RuntimeError(f"Failed to load any base model candidate for fine-tuning: {joined}")

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # Dataset/training loop intentionally kept minimal for repository integration.
    # Replace with project-specific multimodal dataset loader and trainer as needed.
    print("Loaded base model and LoRA config")
    print(f"Base model selected: {selected_model_name}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Processor vocab size: {len(processor.tokenizer)}")
    print("Training scaffold ready. Implement project dataset + trainer in this module.")

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
