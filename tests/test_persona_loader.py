from pathlib import Path
import sys
from jsonschema.exceptions import ValidationError
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.persona_loader import load_personas


def test_load_personas_success():
    personas = load_personas(["aquinas", "aristotle", "augustine"])
    assert personas["aquinas"]["id"] == "theologian"
    assert personas["aristotle"]["id"] == "philosopher"
    assert personas["augustine"]["id"] == "judge"


def test_invalid_persona(tmp_path: Path):
    bad = tmp_path / "bad_v1.0.yaml"
    bad.write_text("id: missing")
    with pytest.raises(ValidationError):
        load_personas(["bad"], base_dir=str(tmp_path))
