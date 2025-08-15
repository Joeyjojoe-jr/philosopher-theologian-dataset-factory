"""
File: src/quality_gate.py
Purpose: Apply thresholded quality gate per audited turn.
Inputs: --batch <batch_id>
Outputs: runs/<batch_id>/accepted/*, runs/<batch_id>/rejected/*
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.parse_args()
    print("[quality_gate] Placeholder. To be implemented per thresholds in config.")


if __name__ == "__main__":
    main()
