"""audit_loop
=================

Minimal auditing pass over generated debate turns.  For each generated turn we
split the text into simple "claims" (sentence level), retrieve supporting
context using the same hybrid BM25 + dense search utility as ``debate_loop``,
and assign a binary verdict (``supported`` / ``unsupported``).  A per-turn audit
summary with a ``support_rate`` is written to ``runs/<batch_id>/audits``.

The implementation is deliberately lightweight – it does not attempt to perform
deep factual validation but provides the plumbing needed by downstream modules.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List

from .config import load_config
from .debate_loop import _prepare_retrieval, _hybrid_search
from .utils.logging import write_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_claims(text: str) -> List[str]:
    """Return rough sentence-level claims from ``text``.

    The splitter is intentionally naive but suffices for auditing in tests.
    """

    sentences = re.split(r"(?<=[.!?]) +", text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="Batch identifier")
    ap.add_argument(
        "--config",
        default="configs/default.ci.yaml",
        help="Config path providing retrieval paths",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    runs_dir = Path(cfg["paths"]["runs"]) / args.batch
    gen_dir = runs_dir / "generated"
    audits_dir = runs_dir / "audits"
    audits_dir.mkdir(parents=True, exist_ok=True)

    # prepare retrieval stack from corpora
    corpora_dir = Path(cfg["paths"]["corpora"])
    docs, bm25, encoder, f_index = _prepare_retrieval(corpora_dir)

    for p in sorted(gen_dir.glob("*.json")):
        with open(p, "r", encoding="utf-8") as f:
            turn = json.load(f)

        claims = _split_claims(turn.get("text", ""))
        claim_results = []
        supported = 0
        for cl in claims:
            evidence = _hybrid_search(cl, docs, bm25, encoder, f_index, k=5)
            verdict = "supported" if evidence else "unsupported"
            if verdict == "supported":
                supported += 1
            claim_results.append({"text": cl, "verdict": verdict, "evidence": evidence})

        support_rate = round(supported / len(claims), 2) if claims else 0.0

        audit = {
            "turn_id": turn.get("id"),
            "claims": claim_results,
            "support_rate": support_rate,
        }

        write_json(audits_dir / f"{turn.get('id')}.json", audit)


if __name__ == "__main__":
    main()
