"""Build dense and sparse retrieval indices from text corpora.

This script loads raw text files specified in the configuration, splits
them into fixed-size word chunks, embeds the chunks with the
``intfloat/multilingual-e5-base`` encoder, and constructs both a FAISS
vector index and a BM25 sparse index.  Metadata describing the generated
indices is saved to ``indices/meta.json`` including checksums for the
index files.

Use ``--dry-run`` to write a lightweight ``meta.json`` without building
the indices (useful for tests).
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path
from typing import Iterable, List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .config import load_config
from .utils.logging import now_iso, write_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHUNK_WORDS = 200  # number of whitespace-delimited tokens per chunk


def chunk_text(text: str, size: int = CHUNK_WORDS) -> Iterable[List[str]]:
    """Yield ``size``-word chunks (as lists of strings) from ``text``."""

    words = text.split()
    for i in range(0, len(words), size):
        yield words[i : i + size]


def file_checksum(path: Path) -> str:
    """Return SHA256 checksum of ``path``."""

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    corpora_dir = Path(cfg["paths"]["corpora"])
    indices_dir = Path(cfg["paths"]["indices"])
    indices_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        meta = {
            "encoder": "intfloat/multilingual-e5-base",
            "built_at": now_iso(),
            "dry_run": True,
        }
        write_json(indices_dir / "meta.json", meta)
        print(f"[chunk_and_index] wrote {indices_dir/'meta.json'}")
        return

    # ------------------------------------------------------------------
    # Load corpora and create chunks
    # ------------------------------------------------------------------
    chunks: List[str] = []
    tokenized: List[List[str]] = []
    for path in sorted(corpora_dir.rglob("*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        for token_list in chunk_text(text):
            chunks.append(" ".join(token_list))
            tokenized.append(token_list)

    if not chunks:
        raise ValueError(f"no .txt corpora found under {corpora_dir}")

    # ------------------------------------------------------------------
    # Dense embeddings and FAISS index
    # ------------------------------------------------------------------
    encoder_name = "intfloat/multilingual-e5-base"
    model = SentenceTransformer(encoder_name)
    embeddings = model.encode(
        chunks,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    faiss_path = indices_dir / "dense.faiss"
    faiss.write_index(index, str(faiss_path))

    # ------------------------------------------------------------------
    # BM25 index
    # ------------------------------------------------------------------
    bm25 = BM25Okapi(tokenized)
    bm25_path = indices_dir / "bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    meta = {
        "encoder": encoder_name,
        "built_at": now_iso(),
        "vector_dim": dim,
        "num_chunks": len(chunks),
        "faiss": {
            "version": faiss.__version__,
            "file": faiss_path.name,
            "count": index.ntotal,
            "checksum": file_checksum(faiss_path),
        },
        "bm25": {
            "file": bm25_path.name,
            "count": len(tokenized),
            "checksum": file_checksum(bm25_path),
        },
    }

    write_json(indices_dir / "meta.json", meta)
    print(f"[chunk_and_index] wrote {indices_dir/'meta.json'}")


if __name__ == "__main__":  # pragma: no cover
    main()

