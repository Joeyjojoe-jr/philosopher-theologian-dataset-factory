"""
File: src/audit_loop.py
Purpose: Split claims, retrieve evidence (BM25 + dense), verdict each claim.
Inputs: --batch <batch_id>
Outputs: runs/<batch_id>/audits/*.json
"""
import argparse
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    args = ap.parse_args()
    print("[audit_loop] Placeholder. To be implemented with hybrid retrieval and claim verdicts.")
if __name__ == "__main__":
    main()
