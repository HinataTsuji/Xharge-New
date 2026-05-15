"""Model loading utilities with optional GPU and batching support."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Iterable, List

from solar_backend.core.config import settings
from solar_backend.core.exceptions import InferenceError

logger = logging.getLogger(__name__)


@dataclass
class ModelHandle:
    """Container for a loaded model and associated processor."""

    name: str
    model: Any
    processor: Any
    device: str
    ready: bool


class ModelRegistry:
    """Centralized lazy model loader for all optional AI backends."""

    def __init__(self) -> None:
        self._qwen: ModelHandle | None = None

    @staticmethod
    def _resolve_device(device_preference: str) -> str:
        try:
            import torch  # type: ignore
        except Exception:
            return "cpu"
        if device_preference == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load_qwen_vl(self) -> ModelHandle:
        """Load Qwen2.5-VL model with transformers when available."""
        if self._qwen is not None:
            return self._qwen

        device = self._resolve_device(settings.device)
        try:
            import torch  # type: ignore
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # type: ignore

            dtype = torch.float16 if settings.model_dtype == "float16" and device == "cuda" else torch.float32
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                settings.model_name,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
            )
            if device != "cuda":
                model = model.to(device)
            processor = AutoProcessor.from_pretrained(settings.model_name)
            self._qwen = ModelHandle(
                name=settings.model_name,
                model=model,
                processor=processor,
                device=device,
                ready=True,
            )
        except Exception as exc:  # pragma: no cover - model availability is environment dependent
            logger.warning("Qwen2.5-VL unavailable, falling back to classical CV path: %s", exc)
            self._qwen = ModelHandle(
                name=settings.model_name,
                model=None,
                processor=None,
                device=device,
                ready=False,
            )
        return self._qwen

    def infer_qwen_structured(
        self,
        prompt: str,
        images: Iterable[Any],
        max_new_tokens: int = 256,
    ) -> List[dict[str, Any]]:
        """Run batched VLM inference and parse JSON outputs.

        If model is unavailable, raises InferenceError so caller can fallback safely.
        """
        handle = self.load_qwen_vl()
        if not handle.ready or handle.model is None or handle.processor is None:
            raise InferenceError("Qwen2.5-VL model is not available in current runtime")
        import torch  # type: ignore

        image_list = list(images)
        outputs: List[dict[str, Any]] = []
        batch_size = 2 if handle.device == "cuda" else 1

        for i in range(0, len(image_list), batch_size):
            batch = image_list[i : i + batch_size]
            messages = [
                [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
                for img in batch
            ]
            text_inputs = [handle.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            inputs = handle.processor(text=text_inputs, images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(handle.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.inference_mode():
                generated_ids = handle.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_text = handle.processor.batch_decode(generated_ids, skip_special_tokens=True)
            for text in generated_text:
                outputs.append(self._safe_parse_json(text))
        return outputs

    @staticmethod
    def _safe_parse_json(text: str) -> dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"raw": text}
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return {"raw": text}


model_registry = ModelRegistry()
