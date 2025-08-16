"""Training script for LoRA-finetuned SFT models.

This script loads an instruction-following dataset in JSONL format and
fine-tunes a base causal language model using LoRA adapters.  The training
configuration (model name, LoRA hyperparameters, and training arguments) is
provided via a YAML config file.  The resulting adapters and tokenizer are
saved under ``models/<run_tag>/`` along with a README documenting the run and
dataset commit hash.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict

import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer


def load_config(path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_sft_dataset(path: str) -> Dataset:
    """Load an SFT dataset from a JSONL file via memory-mapped streaming."""

    dataset = load_dataset("json", data_files=path, split="train")

    def format_sample(sample: Dict[str, Any]) -> Dict[str, str]:
        instruction = sample.get("instruction", "").strip()
        response = sample.get("response", "").strip()
        return {"text": f"{instruction}\n{response}".strip()}

    return dataset.map(format_sample, remove_columns=list(dataset.features))


def git_commit(path: Path) -> str:
    """Return the git commit hash for ``path`` if available."""

    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--data", required=True, help="Path to SFT JSONL")
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory (e.g. models/<run_tag>)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = load_sft_dataset(args.data)

    model_name = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, use_cache=False, device_map="auto"
    )

    lora_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir=args.out,
        learning_rate=cfg["lr"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        num_train_epochs=cfg["epochs"],
        logging_steps=10,
        save_strategy="no",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        peft_config=lora_cfg,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_len"],
    )

    trainer.train()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    data_commit = git_commit(Path(args.data).resolve().parent)
    readme_lines = [f"data: {args.data}", f"data_commit: {data_commit}"]
    for key, value in sorted(cfg.items()):
        readme_lines.append(f"{key}: {value}")
    readme = "\n".join(readme_lines) + "\n"
    (out_dir / "README.txt").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()

