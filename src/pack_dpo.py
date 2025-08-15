"""src.pack_dpo
=================

Package accepted and rejected turns into a DPO (Direct Preference Optimization)
JSONL shard.  The script scans ``runs/<batch>/accepted`` and
``runs/<batch>/rejected`` for per-turn JSON files keyed by ``speaker`` and
``topic``.  Each pair is written as a DPO training item under
``datasets/dpo/<batch>.jsonl``.

The implementation is intentionally minimal and assumes each turn JSON contains
at least the fields ``id``, ``speaker``, ``topic`` and ``text``.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_turn(path: Path) -> Dict[str, Any]:
    """Return a single turn loaded from ``path``."""

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_turns_from_dir(dir_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Loads all JSON turns from a directory into a dictionary keyed by
    ``(speaker, topic)``.

    Missing directories yield an empty mapping, allowing callers to avoid
    repetitive existence checks.
    """

    turns: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not dir_path.exists():
        return turns

    for fp in sorted(dir_path.glob("*.json")):
        try:
            turn = _load_turn(fp)
            turns[(turn["speaker"], turn["topic"])] = turn
        except (json.JSONDecodeError, KeyError):
            print(f"[pack_dpo] Warning: Skipping invalid turn file: {fp}")
    return turns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--runs-dir", default="runs", help="Root runs directory")
    ap.add_argument(
        "--out-dir", default="datasets/dpo", help="Output directory for JSONL"
    )
    args = ap.parse_args()

    batch_id = args.batch
    runs_dir = Path(args.runs_dir) / batch_id
    acc_dir = runs_dir / "accepted"
    rej_dir = runs_dir / "rejected"

    accepted = _load_turns_from_dir(acc_dir)
    rejected = _load_turns_from_dir(rej_dir)

    items = []
    for key, acc in accepted.items():
        rej = rejected.get(key)
        if not rej:
            continue
        speaker, topic = key
        items.append(
            {
                "id": f"{batch_id}.{uuid.uuid4().hex[:8]}",
                "prompt": topic,
                "chosen": acc.get("text", ""),
                "rejected": rej.get("text", ""),
                "meta": {"speaker": speaker, "topic": topic, "batch_id": batch_id},
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{batch_id}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"[pack_dpo] Wrote {len(items)} items to {out_path}")


if __name__ == "__main__":
    main()
