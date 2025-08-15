"""Utilities for loading persona definitions from YAML with JSON schema validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import yaml
from jsonschema import ValidationError, validate


def _load_schema(schema_path: str) -> dict:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_persona_file(path: Path, schema: dict) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    validate(instance=data, schema=schema)
    return data


def load_personas(order: List[str], base_dir: str = "configs/personas", schema_path: str = "schemas/persona.schema.json") -> Dict[str, dict]:
    """Load persona YAML files in the given order.

    Parameters
    ----------
    order: list of str
        Persona names matching file stems (case-insensitive).
    base_dir: str
        Directory containing persona YAML files.
    schema_path: str
        Path to the JSON schema for validation.
    """
    schema = _load_schema(schema_path)
    base = Path(base_dir)
    personas: Dict[str, dict] = {}
    for name in order:
        file = base / f"{name.lower()}_v1.0.yaml"
        if not file.exists():
            raise FileNotFoundError(f"Persona file not found: {file}")
        personas[name.lower()] = _load_persona_file(file, schema)
    return personas

__all__ = ["load_personas", "ValidationError"]
