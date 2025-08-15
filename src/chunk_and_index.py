"""
File: src/chunk_and_index.py
Purpose: Build BM25 and FAISS indices from Latin corpora.
Inputs: --config path to YAML with paths.corpora and paths.indices
Outputs: indices/bm25/*, indices/faiss/*, indices/meta.json
Notes: In dry-run, this writes a meta.json only.
"""

import argparse
from pathlib import Path
from typing import List
from .config import load_config
from .utils.logging import write_json, now_iso


def chunk_text(text: str, max_chars: int = 800, overlap: int = 80) -> List[str]:
    """Split *text* into overlapping character chunks.

    Parameters
    ----------
    text:
        Input string to split.
    max_chars:
        Maximum characters per chunk.
    overlap:
        Number of overlapping characters between consecutive chunks.

    Returns
    -------
    list[str]
        Ordered list of chunk strings.
    """

    if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
        raise ValueError("max_chars must be >0 and 0 <= overlap < max_chars")

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    indices = Path(cfg["paths"]["indices"])
    indices.mkdir(parents=True, exist_ok=True)
    meta = {
        "encoder": "intfloat/multilingual-e5-base",
        "built_at": now_iso(),
        "faiss": "todo:version",
        "bm25": "rank_bm25",
        "notes": "stub meta in dry-run",
    }
    write_json(indices / "meta.json", meta)
    print(f"[chunk_and_index] wrote {indices/'meta.json'}")


if __name__ == "__main__":
    main()
