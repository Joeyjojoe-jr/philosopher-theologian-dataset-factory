"""src.pack_dpo
=================

Package accepted and rejected turns into a DPO (Direct Preference Optimization)
JSONL shard. The script scans ``runs/<batch>/accepted`` and
``runs/<batch>/rejected`` for per-turn JSON files keyed by ``speaker`` and
``topic``. Only rejected turns are loaded into memory; accepted turns are
streamed from disk so each matching pair is written directly to
``datasets/dpo/<batch>.jsonl``.

The implementation is intentionally minimal and assumes each turn JSON contains
at least the fields ``id``, ``speaker``, ``topic`` and ``text``.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_turn(path: Path) -> dict[str, Any]:
    """Return a single turn loaded from ``path``."""

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_turns_from_dir(dir_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load all JSON turns from ``dir_path``.

    The result is a mapping keyed by ``(speaker, topic)``. Invalid JSON files or
    those missing required keys are skipped with a warning. Non-directories yield
    an empty mapping, allowing callers to avoid repetitive checks.
    """

    turns: dict[tuple[str, str], dict[str, Any]] = {}
    if not dir_path.is_dir():
        return turns

    for fp in sorted(dir_path.glob("*.json")):
        try:
            turn = _load_turn(fp)
            key = (turn["speaker"], turn["topic"])
        except (json.JSONDecodeError, KeyError) as e:
            print(
                f"[pack_dpo] Warning: Skipping invalid turn file {fp}. Reason: {e}"
            )
            continue
        turns[key] = turn
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

    # Only load rejected turns into memory to keep peak usage low.
    rejected = _load_turns_from_dir(rej_dir)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{batch_id}.jsonl"

    item_count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for fp in sorted(acc_dir.glob("*.json")):
            try:
                acc = _load_turn(fp)
                key = (acc["speaker"], acc["topic"])
            except (json.JSONDecodeError, KeyError) as e:
                print(
                    f"[pack_dpo] Warning: Skipping invalid accepted turn file {fp}. Reason: {e}"
                )
                continue

            rej = rejected.get(key)
            if rej is None:
                continue

            speaker, topic = key
            item = {
                "id": f"{batch_id}.{uuid.uuid4().hex[:8]}",
                "prompt": topic,
                "chosen": acc.get("text", ""),
                "rejected": rej.get("text", ""),
                "meta": {"speaker": speaker, "topic": topic, "batch_id": batch_id},
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            item_count += 1

    print(f"[pack_dpo] Wrote {item_count} items to {out_path}")


if __name__ == "__main__":
    main()
