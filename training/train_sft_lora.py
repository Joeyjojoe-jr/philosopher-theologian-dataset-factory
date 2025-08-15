# training/train_sft_lora.py (stub)
# Purpose: Fine-tune a small instruct model using LoRA on SFT JSONL.
# This is a placeholder to be completed by AI.
import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out)/"README.txt").write_text("LoRA model artifacts will be saved here.")

if __name__ == "__main__":
    main()
