"""
File: src/quality_gate.py
Purpose: Apply thresholded quality gate per audited turn.
Inputs: --batch <batch_id>
Outputs: runs/<batch_id>/accepted/*, runs/<batch_id>/rejected/*, runs/<batch_id>/review/*
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .utils.logging import write_json, now_iso

# --- Configuration -----------------------------------------------------------------

# Default metric thresholds. These may eventually be loaded from config but are
# hard coded for now to keep the example lightweight.
THRESHOLDS = {
    "support_rate": 0.80,     # fraction of supported claims
    "jaccard": 0.85,          # maximum allowable Jaccard similarity
    "min_citations": 1,       # minimum citation count
    "min_length": 50,         # minimum word count
    "max_toxicity": 0.5,      # maximum toxicity ratio
}

TOXIC_WORDS = {
    "hate", "stupid", "idiot", "kill", "trash",
}

# ----------------------------------------------------------------------------

@dataclass
class Metrics:
    support_rate: float
    citation_count: int
    jaccard: float
    length: int
    toxicity: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "support_rate": self.support_rate,
            "citation_count": self.citation_count,
            "jaccard": self.jaccard,
            "length": self.length,
            "toxicity": self.toxicity,
        }


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def compute_metrics(turn: Dict, accepted_texts: List[str]) -> Metrics:
    """Compute metrics for a single audited turn."""
    claims = turn.get("claims", [])
    supported = sum(1 for c in claims if c.get("supported"))
    total_claims = len(claims)
    support_rate = supported / total_claims if total_claims else 0.0

    citation_count = sum(len(c.get("citations", [])) for c in claims)

    text = turn.get("text", "")
    tokens_list = _tokenize(text)
    tokens = set(tokens_list)
    length = len(tokens_list)

    # Novelty via Jaccard similarity with previously accepted texts.
    max_jaccard = 0.0
    for prior in accepted_texts:
        prior_tokens = set(_tokenize(prior))
        if not prior_tokens:
            continue
        inter = len(tokens & prior_tokens)
        union = len(tokens | prior_tokens)
        if union:
            j = inter / union
            if j > max_jaccard:
                max_jaccard = j

    # Toxicity as ratio of toxic words to total tokens.
    if tokens_list:
        toxic = sum(1 for t in tokens_list if t in TOXIC_WORDS)
        toxicity = toxic / len(tokens_list)
    else:
        toxicity = 0.0

    return Metrics(
        support_rate=support_rate,
        citation_count=citation_count,
        jaccard=max_jaccard,
        length=length,
        toxicity=toxicity,
    )


def check_metrics(metrics: Metrics) -> Tuple[bool, List[str]]:
    """Compare metrics against thresholds; return pass flag and reasons."""
    reasons = []
    if metrics.support_rate < THRESHOLDS["support_rate"]:
        reasons.append(
            f"support_rate {metrics.support_rate:.2f} < {THRESHOLDS['support_rate']:.2f}"
        )
    if metrics.jaccard > THRESHOLDS["jaccard"]:
        reasons.append(
            f"jaccard {metrics.jaccard:.2f} > {THRESHOLDS['jaccard']:.2f}"
        )
    if metrics.citation_count < THRESHOLDS["min_citations"]:
        reasons.append(
            f"citation_count {metrics.citation_count} < {THRESHOLDS['min_citations']}"
        )
    if metrics.length < THRESHOLDS["min_length"]:
        reasons.append(
            f"length {metrics.length} < {THRESHOLDS['min_length']}"
        )
    if metrics.toxicity > THRESHOLDS["max_toxicity"]:
        reasons.append(
            f"toxicity {metrics.toxicity:.2f} > {THRESHOLDS['max_toxicity']:.2f}"
        )
    return (len(reasons) == 0, reasons)


def _route(dest_dir: Path, audit_path: Path, gate_result: Dict):
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / audit_path.name
    with open(audit_path, "r", encoding="utf-8") as src:
        data = json.load(src)
    write_json(dest_dir / f"{audit_path.stem}_gate.json", gate_result)
    with open(dest_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_turn(audit_path: Path, accepted_texts: List[str]) -> None:
    with open(audit_path, "r", encoding="utf-8") as f:
        turn = json.load(f)

    metrics = compute_metrics(turn, accepted_texts)
    passed, reasons = check_metrics(metrics)

    gate_result = {
        "metrics": metrics.as_dict(),
        "passed": passed,
        "reasons": reasons,
        "timestamp": now_iso(),
        "retried": False,
    }

    run_dir = audit_path.parents[1]
    if passed:
        _route(run_dir / "accepted", audit_path, gate_result)
        accepted_texts.append(turn.get("text", ""))
        return

    # Auto-retry with adjusted sampler settings (stub)
    logging.info("Retrying %s with adjusted sampler settings", audit_path.name)
    gate_result["retried"] = True
    metrics_retry = compute_metrics(turn, accepted_texts)
    passed_retry, reasons_retry = check_metrics(metrics_retry)

    gate_result["metrics_retry"] = metrics_retry.as_dict()
    gate_result["passed_retry"] = passed_retry
    gate_result["retry_reasons"] = reasons_retry

    if passed_retry:
        _route(run_dir / "accepted", audit_path, gate_result)
        accepted_texts.append(turn.get("text", ""))
    else:
        _route(run_dir / "rejected", audit_path, gate_result)
        # Also copy to review queue for manual inspection
        _route(run_dir / "review", audit_path, gate_result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    run_dir = Path("runs") / args.batch
    audit_dir = run_dir / "audits"
    if not audit_dir.exists():
        print(f"[quality_gate] audit dir not found: {audit_dir}")
        return

    accepted_texts: List[str] = []
    for path in sorted(audit_dir.glob("*.json")):
        process_turn(path, accepted_texts)


if __name__ == "__main__":
    main()
