"""
File: src/validate_jsonl.py
Purpose: Validate JSONL file against a JSON schema.
CLI: python -m src.validate_jsonl --input <file.jsonl> --schema schemas/sft.schema.json
"""

import argparse
import json
from jsonschema import Draft7Validator


def validate_file(input_path: str, schema_path: str) -> None:
    """Validate *input_path* JSONL against *schema_path* JSON schema."""
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    v = Draft7Validator(schema)

    errors = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            obj = json.loads(line)
            for e in v.iter_errors(obj):
                errors += 1
                print(f"[line {i}] {e.message} at {'/'.join(map(str,e.absolute_path))}")
    if errors:
        raise ValueError(f"{errors} validation errors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--schema", required=True)
    args = ap.parse_args()

    try:
        validate_file(args.input, args.schema)
    except ValueError as e:
        print(f"[validate_jsonl] {e}")
        raise SystemExit(1)
    else:
        print("[validate_jsonl] OK")


if __name__ == "__main__":
    main()
