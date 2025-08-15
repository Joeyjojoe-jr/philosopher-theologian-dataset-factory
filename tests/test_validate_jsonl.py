import subprocess
import sys
from pathlib import Path


def test_validate_jsonl_ok(valid_sft_jsonl):
    schema = Path(__file__).parent.parent / "schemas/sft.schema.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.validate_jsonl",
            "--input",
            str(valid_sft_jsonl),
            "--schema",
            str(schema),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[validate_jsonl] OK" in result.stdout
