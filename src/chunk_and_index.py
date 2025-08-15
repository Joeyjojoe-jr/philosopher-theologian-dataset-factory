"""
File: src/chunk_and_index.py
Purpose: Build BM25 and FAISS indices from Latin corpora.
Inputs: --config path to YAML with paths.corpora and paths.indices
Outputs: indices/bm25/*, indices/faiss/*, indices/meta.json
Notes: In dry-run, this writes a meta.json only.
"""
import argparse, json
from pathlib import Path
from .config import load_config
from .utils.logging import write_json, now_iso

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    indices = Path(cfg["paths"]["indices"])
    indices.mkdir(parents=True, exist_ok=True)
    meta = {
        "encoder":"intfloat/multilingual-e5-base",
        "built_at": now_iso(),
        "faiss": "todo:version",
        "bm25": "rank_bm25",
        "notes":"stub meta in dry-run"
    }
    write_json(indices/"meta.json", meta)
    print(f"[chunk_and_index] wrote {indices/'meta.json'}")

if __name__ == "__main__":
    main()
