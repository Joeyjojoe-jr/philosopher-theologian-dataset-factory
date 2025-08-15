"""
File: src/debate_loop.py
Purpose: Generate persona turns in Latin with citations (real impl uses transformers).
Inputs: --topics YAML path
Outputs: runs/<batch_id>/generated/*.json
Dry-run: no-op (auto_runner fabricates SFT/DPO directly).
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", required=True)
    ap.parse_args()
    print(
        "[debate_loop] Placeholder. Use auto_runner --dry-run or implement model generation."
    )


if __name__ == "__main__":
    main()
