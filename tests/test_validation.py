import json
from pathlib import Path
from src.validate_jsonl import validate_file


def test_validate_jsonl(tmp_path):
    sample = {
        "id": "1",
        "instruction": "Test",
        "response": "Resp",
        "meta": {
            "speaker": "Aquinas",
            "topic": "Test",
            "citations": [{"work": "Summa", "ref": "I-II q109 a2"}],
            "provenance": [
                {
                    "work": "Summa",
                    "ref": "I-II q109 a2",
                    "snippet": "gratia non tollit naturam",
                }
            ],
            "audit_summary": {"claims": 1, "correct": 1, "support_rate": 1.0},
            "batch_id": "b1",
            "encoder": "e",
            "model": "m",
            "commit": "",
        },
    }
    jsonl = tmp_path / "sample.jsonl"
    jsonl.write_text(json.dumps(sample) + "\n")
    schema = Path("schemas/sft.schema.json")
    validate_file(str(jsonl), str(schema))
