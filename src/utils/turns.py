from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_turn(path: Path) -> Dict[str, Any]:
    """Return a normalised representation of a turn.

    Input files from earlier pipeline stages may vary slightly in structure.
    This helper extracts the fields required by the packaging scripts,
    providing sensible defaults for optional metadata.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("meta", {})
    return {
        "instruction": data.get("instruction")
        or data.get("prompt")
        or meta.get("topic")
        or data.get("topic", ""),
        "response": data.get("response") or data.get("text", ""),
        "speaker": meta.get("speaker") or data.get("speaker", ""),
        "topic": meta.get("topic")
        or data.get("topic")
        or data.get("instruction")
        or data.get("prompt", ""),
        "citations": meta.get("citations") or data.get("citations", []),
        "provenance": meta.get("provenance") or data.get("provenance", []),
        "audit_summary": meta.get("audit_summary")
        or data.get("audit_summary", {}),
        "encoder": meta.get("encoder", ""),
        "model": meta.get("model", ""),
        "commit": meta.get("commit", ""),
    }
