"""
File: src/config.py
Purpose: Load and validate YAML config for the pipeline.
Inputs: --config path
Outputs: Python dict with keys: batch_id, seed, personas, generator, auditor, gate, paths
"""
import yaml, json
from pathlib import Path

def load_config(path: str):
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg
