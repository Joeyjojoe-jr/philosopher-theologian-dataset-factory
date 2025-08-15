#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate || true
python -m src.auto_runner --config configs/default.yaml --topics topics/queue.latin_v1_001.yaml "$@"
