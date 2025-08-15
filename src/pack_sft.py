"""
File: src/pack_sft.py
Purpose: Package accepted turns into SFT JSONL.
Inputs: --batch <batch_id>
Outputs: datasets/sft/<batch_id>.jsonl
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.parse_args()
    print(
        "[pack_sft] Placeholder. Use auto_runner --dry-run to generate example SFT JSONL."
    )


if __name__ == "__main__":
    main()
