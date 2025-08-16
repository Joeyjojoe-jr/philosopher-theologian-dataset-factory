"""src.pack_dpo
================

Create DPO (Direct Preference Optimisation) training pairs for a given batch.
The script reads accepted and rejected turns from ``runs/<batch_id>`` by
default and emits a JSONL shard matching :mod:`schemas/dpo.schema.json`.
Alternative root directories can be provided via ``--runs-dir`` and
``--datasets-dir``.

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


def _load_turn(path: Path) -> Dict[str, Any]:
    """Normalise an accepted or rejected turn into a flat dict."""

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("meta", {})
    return {
        "prompt": data.get("instruction")
        or data.get("prompt")
        or meta.get("topic")
        or data.get("topic", ""),
        "response": data.get("response") or data.get("text", ""),
        "speaker": meta.get("speaker") or data.get("speaker", ""),
        "topic": meta.get("topic")
        or data.get("topic")
        or data.get("instruction")
        or data.get("prompt", ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="Batch identifier")
    ap.add_argument(
        "--runs-dir",
        default="runs",
        help="Root directory containing run artifacts (default: runs)",
    )
    ap.add_argument(
        "--datasets-dir",
        default="datasets",
        help="Root directory for dataset shards (default: datasets)",
    )
    args = ap.parse_args()

    batch_id = args.batch
    base_dir = Path(args.runs_dir) / batch_id
    acc_dir = base_dir / "accepted"
    rej_dir = base_dir / "rejected"
    out_dir = Path(args.datasets_dir) / "dpo"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not acc_dir.exists():
        raise SystemExit(f"[pack_dpo] Missing directory: {acc_dir}")

    accepted: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for fp in sorted(acc_dir.glob("*.json")):
        t = _load_turn(fp)
        accepted[(t["speaker"], t["topic"])] = t

    rejected: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if rej_dir.exists():
        for fp in sorted(rej_dir.glob("*.json")):
            t = _load_turn(fp)
            rejected[(t["speaker"], t["topic"])] = t

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
                "prompt": acc["prompt"],
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
