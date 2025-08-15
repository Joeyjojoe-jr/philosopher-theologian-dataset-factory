"""Quality gate for audited turns.

This module evaluates per-turn metrics against configuration thresholds and
routes each turn to either the ``accepted`` or ``rejected`` folder.  The actual
metrics extraction is intentionally minimal – the dry-run pipeline generates
stub audit files that already contain the necessary ``metrics`` object.

The previous implementation performed the gating comparison directly inside the
main loop with a long boolean expression.  For readability and unit testing, the
comparison is now encapsulated in :func:`_passes_gate`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .utils.logging import write_json


def _passes_gate(metrics: dict, thresholds: dict) -> bool:
    """Check if a turn's metrics meet the quality thresholds."""

    return (
        thresholds["min_words"] <= metrics["words"] <= thresholds["max_words"]
        and thresholds["min_citations"]
        <= metrics["citations"]
        <= thresholds["max_citations"]
        and metrics["support_rate"] >= thresholds["min_support_rate"]
        and metrics["latin_score"] >= thresholds["min_latin_score"]
        and metrics["novelty"] <= thresholds["novelty_jaccard_max"]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    thresholds = {
        "min_words": cfg["generator"]["min_words"],
        "max_words": cfg["generator"]["max_words"],
        "min_citations": cfg["generator"]["min_citations"],
        "max_citations": cfg["generator"]["max_citations"],
        "min_support_rate": cfg["gate"]["min_support_rate"],
        "min_latin_score": cfg["gate"]["min_latin_score"],
        "novelty_jaccard_max": cfg["gate"]["novelty_jaccard_max"],
    }

    runs_dir = Path(cfg["paths"]["runs"]) / args.batch
    audits_dir = runs_dir / "audits"
    accepted_dir = runs_dir / "accepted"
    rejected_dir = runs_dir / "rejected"
    accepted_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for audit_path in audits_dir.glob("*.json"):
        with open(audit_path, "r", encoding="utf-8") as f:
            audit = json.load(f)
        metrics = audit.get("metrics", {})
        passed = _passes_gate(metrics, thresholds)
        dest = accepted_dir if passed else rejected_dir
        result = {
            "turn_id": audit.get("turn_id"),
            "passed": passed,
            "metrics": metrics,
            "thresholds": thresholds,
        }
        write_json(dest / audit_path.name, result)
        count += 1

    print(f"[quality_gate] Evaluated {count} audits")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
