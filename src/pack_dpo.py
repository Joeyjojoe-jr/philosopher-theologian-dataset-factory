"""src.pack_dpo
================

Create DPO (Direct Preference Optimisation) training pairs for a given batch.
The script reads accepted and rejected turns from ``runs/<batch_id>`` and
emits a JSONL shard matching :mod:`schemas/dpo.schema.json`.

For each ``(speaker, topic)`` combination exactly one entry is produced.  A
rejected turn with the same ``speaker`` and ``topic`` is preferred; if none is
available an ablated version of the accepted response is used instead so that
the resulting JSONL still conforms to the schema.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Dict, Any, Tuple

from .utils.turns import load_turn


def _load_turns_from_dir(dir_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Loads all JSON turns from a directory keyed by (speaker, topic)."""
    turns: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not dir_path.exists():
        return turns

    for fp in sorted(dir_path.glob("*.json")):
        turn = load_turn(fp)
        turns[(turn["speaker"], turn["topic"])] = turn
    return turns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="Batch identifier")
    args = ap.parse_args()

    batch_id = args.batch
    base_dir = Path("runs") / batch_id
    acc_dir = base_dir / "accepted"
    rej_dir = base_dir / "rejected"
    out_dir = Path("datasets") / "dpo"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not acc_dir.exists():
        raise SystemExit(f"[pack_dpo] Missing directory: {acc_dir}")

    accepted = _load_turns_from_dir(acc_dir)
    rejected = _load_turns_from_dir(rej_dir)

    items = []
    for key, acc in accepted.items():
        rej = rejected.get(key)
        if rej:
            rejected_text = rej["response"]
            audit_diffs = "rejected"
        else:
            # Ablated negative: prefix to indicate non-preferred variant
            rejected_text = f"(ablated) {acc['response']}"
            audit_diffs = "ablated accepted; no rejected turn"

        items.append(
            {
                "id": f"{batch_id}.{uuid.uuid4().hex[:8]}",
                "prompt": acc["instruction"],
                "chosen": acc["response"],
                "rejected": rejected_text,
                "meta": {
                    "speaker": acc["speaker"],
                    "topic": acc["topic"],
                    "batch_id": batch_id,
                    "audit_diffs": audit_diffs,
                },
            }
        )

    out_path = out_dir / f"{batch_id}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"[pack_dpo] Wrote {out_path} ({len(items)} items)")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()

