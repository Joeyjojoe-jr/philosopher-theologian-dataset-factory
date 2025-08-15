"""
File: src/validate_jsonl.py
Purpose: Validate JSONL file against a JSON schema.
CLI: python -m src.validate_jsonl --input <file.jsonl> --schema schemas/sft.schema.json
"""
import argparse, json
from jsonschema import Draft7Validator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--schema", required=True)
    args = ap.parse_args()

    with open(args.schema, "r", encoding="utf-8") as f:
        schema = json.load(f)
    v = Draft7Validator(schema)

    errors = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            obj = json.loads(line)
            for e in v.iter_errors(obj):
                errors += 1
                print(f"[line {i}] {e.message} at {'/'.join(map(str,e.absolute_path))}")
    if errors == 0:
        print("[validate_jsonl] OK")
    else:
        print(f"[validate_jsonl] {errors} errors")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
