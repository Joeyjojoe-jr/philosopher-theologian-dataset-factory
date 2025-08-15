"""Quality Gate
-----------------
Routes generated turns into ``accepted`` or ``rejected`` directories
based on audited metrics and thresholds loaded from ``configs/default.yaml``.

This implements feature **F4 – quality_gate** described in the project
documentation.  For each audit result under
``runs/<batch_id>/audits/*.json`` the corresponding generated turn is
loaded, simple metrics are computed and compared against the configured
thresholds.  The turn JSON is then moved into either
``runs/<batch_id>/accepted`` or ``runs/<batch_id>/rejected``.  A
``summary.json`` file is updated with the resulting counts.

Inputs
~~~~~~
``--batch``:  The batch identifier (e.g. ``latin_v1_001``)
``--config``: Optional path to YAML config (defaults to
``configs/default.yaml``)

Outputs
~~~~~~~
``runs/<batch_id>/accepted/*.json``
``runs/<batch_id>/rejected/*.json``
``runs/<batch_id>/summary.json`` (updated counts)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .config import load_config
from .utils.logging import write_json


def _latin_score(text: str) -> float:
    """Return a crude "Latinness" score.

    The score is the ratio of Latin alphabet characters (a–z) to all
    alphabetic characters.  This is a light‑weight heuristic sufficient
    for gating; if no alphabetic characters are present the score is 0.
    """

    latin_count = 0
    alpha_count = 0
    for c in text:
        if c.isalpha():
            alpha_count += 1
            if "a" <= c.lower() <= "z":
                latin_count += 1
    return latin_count / alpha_count if alpha_count > 0 else 0.0


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config with gate thresholds",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    batch_id = args.batch

    runs_dir = Path(cfg["paths"]["runs"]) / batch_id
    gen_dir = runs_dir / "generated"
    audits_dir = runs_dir / "audits"
    accepted_dir = runs_dir / "accepted"
    rejected_dir = runs_dir / "rejected"
    accepted_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    try:
        thresholds = {
            "min_words": cfg["generator"]["min_words"],
            "max_words": cfg["generator"]["max_words"],
            "min_citations": cfg["generator"]["min_citations"],
            "max_citations": cfg["generator"]["max_citations"],
            "min_support_rate": cfg["gate"]["min_support_rate"],
            "min_latin_score": cfg["gate"]["min_latin_score"],
            "novelty_jaccard_max": cfg["gate"].get("novelty_jaccard_max", 1.0),
        }
    except KeyError as e:
        raise ValueError(f"Missing required key in config file: {e}") from e

    counts = {"accepted": 0, "rejected": 0}

    for audit_path in audits_dir.glob("*.json"):
        audit = _load_json(audit_path)
        turn_id = audit.get("turn_id") or audit_path.stem
        gen_path = gen_dir / f"{turn_id}.json"
        if not gen_path.exists():
            # Skip if generation missing
            continue

        turn = _load_json(gen_path)
        text = turn.get("text") or turn.get("response") or ""
        citations = (
            turn.get("citations")
            or turn.get("meta", {}).get("citations")
            or []
        )

        support_rate = audit.get("support_rate")
        if support_rate is None:
            correct = audit.get("correct")
            total = audit.get("claims")
            support_rate = (correct / total) if (correct is not None and total) else 0.0

        metrics = {
            "words": len(text.split()),
            "citations": len(citations),
            "support_rate": support_rate,
            "latin_score": _latin_score(text),
            "novelty": audit.get("novelty", 0.0),
        }

        passed = (
            thresholds["min_words"] <= metrics["words"] <= thresholds["max_words"]
            and thresholds["min_citations"] <= metrics["citations"] <= thresholds["max_citations"]
            and metrics["support_rate"] >= thresholds["min_support_rate"]
            and metrics["latin_score"] >= thresholds["min_latin_score"]
            and metrics["novelty"] <= thresholds["novelty_jaccard_max"]
        )

        dest_dir = accepted_dir if passed else rejected_dir
        shutil.move(str(gen_path), dest_dir / gen_path.name)
        counts["accepted" if passed else "rejected"] += 1

    summary_path = runs_dir / "summary.json"
    if summary_path.exists():
        summary = _load_json(summary_path)
    else:
        summary = {"batch_id": batch_id}

    summary_counts = summary.get("counts", {})
    summary_counts.update(
        {
            "accepted": counts["accepted"],
            "rejected": counts["rejected"],
            "turns_total": counts["accepted"] + counts["rejected"],
        }
    )
    summary["counts"] = summary_counts
    write_json(summary_path, summary)

    print(
        f"[quality_gate] accepted={counts['accepted']} rejected={counts['rejected']}"
    )


if __name__ == "__main__":
    main()
