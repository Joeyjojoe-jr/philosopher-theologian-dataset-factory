from pathlib import Path
from src.claims import extract_claims


def test_extract_claims():
    text = Path("tests/fixtures/aquinas.txt").read_text()
    claims = extract_claims(text)
    assert claims == [
        "Gratia non tollit naturam, sed perficit",
        "Homo est animal rationale",
    ]
