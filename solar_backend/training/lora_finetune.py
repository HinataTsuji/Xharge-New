"""LoRA fine-tuning entrypoint for Qwen2.5-VL."""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen2.5-VL roof extraction tasks")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
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

    # Lazy imports keep runtime dependencies optional for inference-only environments.
    import torch  # type: ignore
    from peft import LoraConfig, get_peft_model  # type: ignore
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # type: ignore

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    processor = AutoProcessor.from_pretrained(args.model_name)

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
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Processor vocab size: {len(processor.tokenizer)}")
    print("Training scaffold ready. Implement project dataset + trainer in this module.")

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
