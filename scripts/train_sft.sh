#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate || true
python training/train_sft_lora.py --config configs/train_sft_lora.yaml --data datasets/sft/latin_v1_001.jsonl --out models/latin_v1_001_lora
