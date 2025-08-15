"""
File: src/utils/logging.py
Purpose: Minimal logging helpers.
"""

from datetime import datetime
import json


def now_iso():
    return datetime.utcnow().isoformat() + "Z"


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
