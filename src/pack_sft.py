"""src.pack_sft
================

Collect accepted turns for a given ``batch_id`` and write them to a single
JSONL shard compatible with :mod:`schemas/sft.schema.json`.

The module expects the following directory layout::

    runs/<batch_id>/accepted/*.json

Each file should contain the generated response together with metadata such as
``speaker`` and ``topic``.  The exact structure is flexible – only the fields
required by the final schema are extracted.  A minimal example of an accepted
turn is::

    {
        "instruction": "De libero arbitrio",
        "response": "... Latin response ...",
        "meta": {
            "speaker": "Aquinas",
            "topic": "De libero arbitrio",
            "citations": [{"work": "ST I-II", "ref": "q109 a2"}],
            "provenance": [{"work": "ST I-II", "ref": "q109 a2", "snippet": "..."}],
            "audit_summary": {"claims": 5, "correct": 4, "support_rate": 0.8},
            "encoder": "intfloat/multilingual-e5-base",
            "model": "Meta-Llama-3-8B-Instruct",
            "commit": "abcdef"
        }
    }

Only files under ``runs/<batch_id>/accepted`` are inspected.  The resulting
shard is written to ``datasets/sft/<batch_id>.jsonl`` and can be validated via
``python -m src.validate_jsonl``.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from .utils.turns import load_turn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="Batch identifier")
    args = ap.parse_args()

    batch_id = args.batch
    runs_dir = Path("runs") / batch_id / "accepted"
    out_dir = Path("datasets") / "sft"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not runs_dir.exists():
        raise SystemExit(f"[pack_sft] Missing directory: {runs_dir}")

    items = []
    for fp in sorted(runs_dir.glob("*.json")):
        turn = load_turn(fp)
        item = {
            "id": f"{batch_id}.{uuid.uuid4().hex[:8]}",
            "instruction": turn["instruction"],
            "response": turn["response"],
            "meta": {
                "speaker": turn["speaker"],
                "topic": turn["topic"],
                "citations": turn["citations"],
                "provenance": turn["provenance"],
                "audit_summary": turn["audit_summary"],
                "batch_id": batch_id,
                "encoder": turn["encoder"],
                "model": turn["model"],
                "commit": turn["commit"],
            },
        }
        items.append(item)

    out_path = out_dir / f"{batch_id}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"[pack_sft] Wrote {out_path} ({len(items)} items)")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()

