"""src.debate_loop
===================

This module performs the *real* debate generation.  It loads persona
definitions, retrieves supporting context via a hybrid BM25/FAISS stack and
invokes a local LLM to produce Latin responses with citations.  Generated turns
are written to ``runs/<batch_id>/generated``.

The implementation is intentionally lightweight – corpora are loaded from plain
text files and a small HuggingFace model is used by default so the module can be
executed in the test environment.  Nevertheless the plumbing mirrors the
expected production behaviour and can be swapped for larger models or more
elaborate indices without changing the public API.
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Dict, List, Sequence

import faiss
import numpy as np
import yaml
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

from .config import load_config
from .utils.logging import now_iso, write_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PERSONA_SCHEMA = {
    "type": "object",
    "required": ["name", "prompt"],
    "properties": {
        "name": {"type": "string"},
        "prompt": {"type": "string"},
    },
}


def _load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _validate_persona(p: Dict) -> Dict:
    """Validate persona YAML structure.

    Only a minimal schema is enforced to keep the implementation lightweight.
    """

    from jsonschema import validate

    validate(p, PERSONA_SCHEMA)
    return p


def _load_personas(paths: Sequence[str]) -> List[Dict]:
    personas = []
    for path in paths:
        data = _validate_persona(_load_yaml(path))
        persona = {
            "name": data["name"],
            "prompt": data["prompt"],
        }
        personas.append(persona)
    return personas


def _build_corpus(corpora_dir: Path) -> List[str]:
    """Return list of documents (one per text file)."""

    docs = []
    if not corpora_dir.exists():
        return docs
    for p in sorted(corpora_dir.glob("*.txt")):
        with open(p, "r", encoding="utf-8") as f:
            docs.append(f.read())
    return docs


def _prepare_retrieval(corpora_dir: Path):
    """Create BM25 and FAISS indices from corpora."""

    docs = _build_corpus(corpora_dir)
    bm25 = BM25Okapi([d.split() for d in docs]) if docs else None

    if docs:
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = encoder.encode(docs, show_progress_bar=False)
        faiss.normalize_L2(embeddings)
        f_index = faiss.IndexFlatIP(embeddings.shape[1])
        f_index.add(embeddings)
    else:
        encoder = None
        f_index = None
    return docs, bm25, encoder, f_index


def _hybrid_search(query: str, docs, bm25, encoder, f_index, k=6):
    """Return top-k context snippets using BM25 and FAISS fused via RRF."""

    if not docs:
        return []

    q_tokens = query.split()
    bm_scores = bm25.get_scores(q_tokens)
    bm_order = np.argsort(bm_scores)[::-1][:k]

    q_emb = encoder.encode([query], show_progress_bar=False)
    faiss.normalize_L2(q_emb)
    dense_scores, dense_ids = f_index.search(q_emb, k)
    dense_order = dense_ids[0]

    scores = {}
    for rank, idx in enumerate(bm_order, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (60 + rank)
    for rank, idx in enumerate(dense_order, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (60 + rank)

    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [
        {"source": f"doc_{i}", "text": docs[i], "score": s} for i, s in top
    ]


def _load_model(model_name: str):
    """Load a causal LM, falling back to a tiny model if necessary."""

    fallback = "sshleifer/tiny-gpt2"
    used = model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(fallback)
        model = AutoModelForCausalLM.from_pretrained(fallback)
        used = fallback
    model.eval()
    return tokenizer, model, used


def _generate(model, tokenizer, prompt: str, max_new_tokens=256) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=max_new_tokens,
        )
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return text.strip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--topics", required=True)
    ap.add_argument("--personas", nargs="+", required=True, help="Persona YAML paths")
    args = ap.parse_args()

    cfg = load_config(args.config)
    topics_yaml = _load_yaml(args.topics)
    personas = _load_personas(args.personas)

    # prepare retrieval
    corpora_dir = Path(cfg["paths"]["corpora"])
    docs, bm25, encoder, f_index = _prepare_retrieval(corpora_dir)

    # load model
    model_name = cfg["personas"].get("model", "sshleifer/tiny-gpt2")
    tokenizer, model, model_name = _load_model(model_name)

    # run conversation
    batch_id = cfg["batch_id"]
    out_dir = Path(cfg["paths"]["runs"]) / batch_id / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    persona_order = topics_yaml.get("persona_order") or cfg["personas"]["order"]
    set_seed(cfg.get("seed", 0))

    max_turns = topics_yaml.get("turns", len(persona_order))

    for topic in topics_yaml["topics"]:
        history: List[str] = []
        for i in range(max_turns):
            persona = personas_by_name[persona_order[i % len(persona_order)]]
            # retrieval
            query = topic + " " + " ".join(history)
            ctx = _hybrid_search(query, docs, bm25, encoder, f_index, k=6)

            context_text = "\n".join(c["text"] for c in ctx)
            prompt = f"{persona['prompt']}\n\nTopic: {topic}\n\nContext:\n{context_text}\n\nResponse:"  # simple template
            response = _generate(model, tokenizer, prompt, max_new_tokens=256)
            history.append(response)

            turn_id = f"{batch_id}.{uuid.uuid4().hex[:8]}"
            item = {
                "id": turn_id,
                "topic": topic,
                "speaker": persona["name"],
                "text": response,
                "citations": [{"source": c["source"]} for c in ctx],
                "meta": {
                    "batch_id": batch_id,
                    "created_at": now_iso(),
                    "model": model_name,
                    "sampler": {"temperature": 0.7, "top_p": 0.9},
                },
            }
            write_json(out_dir / f"{turn_id}.json", item)


if __name__ == "__main__":
    main()

