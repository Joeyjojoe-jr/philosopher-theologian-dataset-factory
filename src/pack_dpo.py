"""
File: src/pack_dpo.py
Purpose: Package accepted vs rejected into DPO JSONL.
Inputs: --batch <batch_id>
Outputs: datasets/dpo/<batch_id>.jsonl
"""
import argparse
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    args = ap.parse_args()
    print("[pack_dpo] Placeholder. Use auto_runner --dry-run to generate example DPO JSONL.")
if __name__ == "__main__":
    main()
