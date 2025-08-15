"""
File: src/debate_loop.py
Purpose: Generate persona turns in Latin with citations (real impl uses transformers).
Inputs: --config YAML path, --topics YAML path
Outputs: runs/<batch_id>/generated/*.json
Dry-run: no-op (auto_runner fabricates SFT/DPO directly).
"""
import argparse
from .config import load_config
from .persona_loader import load_personas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--topics", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    personas = load_personas(cfg["personas"]["order"])
    print("[debate_loop] Loaded personas:", ", ".join(personas.keys()))
    print("[debate_loop] Placeholder. Use auto_runner --dry-run or implement model generation.")


if __name__ == "__main__":
    main()
