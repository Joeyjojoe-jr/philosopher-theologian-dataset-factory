"""
File: src/audit_loop.py
Purpose: Split claims, retrieve evidence (BM25 + dense), verdict each claim.
Inputs: --batch <batch_id>
Outputs: runs/<batch_id>/audits/*.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
try:  # optional faiss
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:  # pragma: no cover - fallback when faiss missing
    faiss = None  # type: ignore
    _HAS_FAISS = False
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import pipeline

from .config import load_config
from .utils.logging import now_iso, write_json


CLAIM_LABELS = ["claim", "non-claim"]

try:
    _CLAIM_CLF = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")
except Exception:  # pragma: no cover - offline fallback
    _CLAIM_CLF = None


def _split_sentences(text: str) -> List[str]:
    """Regex split into sentences."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _classify_claims(sentences: List[str]):
    """Return subset of sentences predicted to be factual claims."""
    if _CLAIM_CLF is None:
        return sentences
    claims = []
    for s in sentences:
        res = _CLAIM_CLF(s, candidate_labels=CLAIM_LABELS)
        if res["labels"][0] == "claim":
            claims.append(s)
    return claims


class HybridRetriever:
    """BM25 + dense FAISS search with optional cross-encoder rerank."""

    def __init__(self, corpus: List[Dict[str, str]]):
        self.corpus = corpus
        self.texts = [d["text"] for d in corpus]
        tokenized = [t.split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)
        try:
            self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.embeddings = self.embedder.encode(
                self.texts, normalize_embeddings=True, show_progress_bar=False
            )
            dim = self.embeddings.shape[1]
            if _HAS_FAISS:
                self.index = faiss.IndexFlatIP(dim)
                self.index.add(self.embeddings)
            else:
                self.index = None
            self.has_dense = True
        except Exception:  # pragma: no cover - offline fallback
            self.embedder = None
            self.embeddings = np.zeros((len(self.texts), 1), dtype="float32")
            self.index = None
            self.has_dense = False

    def search(
        self,
        query: str,
        bm25_k: int = 50,
        dense_k: int = 50,
        rerank: bool = False,
    ) -> List[Dict[str, float]]:
        tokens = query.split()
        bm_scores = self.bm25.get_scores(tokens)
        bm_idx = np.argsort(bm_scores)[::-1][:bm25_k]

        if self.has_dense and self.embedder is not None:
            q_emb = self.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)
            if _HAS_FAISS:
                dense_scores, dense_idx = self.index.search(q_emb, dense_k)
                dense_idx = dense_idx[0]
                dense_scores = dense_scores[0]
            else:
                sims = np.dot(self.embeddings, q_emb[0])
                order = np.argsort(sims)[::-1][:dense_k]
                dense_idx = order
                dense_scores = sims[order]
        else:
            dense_idx = np.array([], dtype=int)
            dense_scores = np.array([], dtype=float)

        if bm_scores.max() > 0:
            bm_scores = bm_scores / bm_scores.max()
        if dense_scores.size and dense_scores.max() > 0:
            dense_scores = dense_scores / dense_scores.max()

        scores: Dict[int, float] = {}
        for i in bm_idx:
            scores[i] = scores.get(i, 0.0) + bm_scores[i]
        for i, s in zip(dense_idx, dense_scores):
            scores[i] = scores.get(i, 0.0) + s

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:50]
        cand_idx = [i for i, _ in top]

        if rerank and cand_idx:
            ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [(query, self.texts[i]) for i in cand_idx]
            ce_scores = ce.predict(pairs)
            order = np.argsort(ce_scores)[::-1][:10]
            cand_idx = [cand_idx[i] for i in order]
        else:
            cand_idx = cand_idx[:10]

        return [dict(self.corpus[i], score=scores[i]) for i in cand_idx]


def _entailment_model():
    for name in ["microsoft/deberta-v3-large-mnli", "microsoft/deberta-v3-base-mnli"]:
        try:
            return pipeline("text-classification", model=name)
        except Exception:
            continue
    return None


def _verdict(claim: str, evidences: List[Dict[str, str]], nli) -> Dict:
    """Run NLI on evidence snippets and decide verdict."""
    evidence_results = []
    best_ent = best_con = best_neu = 0.0
    for ev in evidences:
        if nli is None:
            score_map = {}
        else:
            scores = nli({"text": ev["text"], "text_pair": claim}, return_all_scores=True)[0]
            score_map = {s["label"].lower(): s["score"] for s in scores}
            best_ent = max(best_ent, score_map.get("entailment", 0.0))
            best_con = max(best_con, score_map.get("contradiction", 0.0))
            best_neu = max(best_neu, score_map.get("neutral", 0.0))
        evidence_results.append({
            "text": ev["text"],
            "retrieval_score": ev.get("score", 0.0),
            "entailment": score_map.get("entailment", 0.0),
            "contradiction": score_map.get("contradiction", 0.0),
            "neutral": score_map.get("neutral", 0.0),
        })

    if nli is None:
        verdict = "uncertain"
    elif best_ent > best_con and best_ent > best_neu:
        verdict = "supported"
    elif best_con > best_ent:
        verdict = "refuted"
    else:
        verdict = "uncertain"

    return {
        "verdict": verdict,
        "evidence": evidence_results,
        "scores": {
            "entailment": best_ent,
            "contradiction": best_con,
            "neutral": best_neu,
        },
    }


def _load_corpus(path: Path) -> List[Dict[str, str]]:
    corpus = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                corpus.append(json.loads(line))
    return corpus


def audit_turn(turn_id: str, text: str, retriever: HybridRetriever, nli, cfg, out_dir: Path):
    sentences = _split_sentences(text)
    claims = _classify_claims(sentences)
    correct = 0
    for idx, claim in enumerate(claims):
        retrieved = retriever.search(
            claim,
            bm25_k=cfg.get("bm25_k", 50),
            dense_k=cfg.get("dense_k", 50),
            rerank=cfg.get("reranker", False),
        )[:6]
        result = _verdict(claim, retrieved, nli)
        if result["verdict"] == "supported":
            correct += 1
        out = {
            "turn_id": turn_id,
            "claim_id": idx,
            "claim_text": claim,
            **result,
            "created_at": now_iso(),
        }
        out_path = out_dir / f"{turn_id}.claim{idx}.json"
        write_json(out_path, out)

    summary = {
        "turn_id": turn_id,
        "claims": len(claims),
        "correct": correct,
        "support_rate": round(correct / len(claims), 2) if claims else 0.0,
        "created_at": now_iso(),
    }
    write_json(out_dir / f"{turn_id}.summary.json", summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--config", default="configs/default.ci.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    runs_dir = Path(cfg["paths"]["runs"]) / args.batch
    audits_dir = runs_dir / "audits"
    gen_dir = runs_dir / "generated"
    audits_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = Path(cfg["paths"]["indices"]) / "corpus.jsonl"
    corpus = _load_corpus(corpus_path)
    if not corpus:
        raise RuntimeError(f"no corpus found at {corpus_path}")
    retriever = HybridRetriever(corpus)

    nli = _entailment_model()

    for f in gen_dir.glob("*.json"):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        turn_id = data.get("id", f.stem)
        text = data.get("response") or data.get("text") or ""
        audit_turn(turn_id, text, retriever, nli, cfg["auditor"], audits_dir)

    print(f"[audit_loop] wrote audits to {audits_dir}")


if __name__ == "__main__":
    main()

