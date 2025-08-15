from pathlib import Path
from src.chunk_and_index import chunk_text


def test_chunking_overlap():
    text = Path("tests/fixtures/aquinas.txt").read_text()
    chunks = chunk_text(text, max_chars=50, overlap=10)
    assert len(chunks) == 2
    assert len(chunks[0]) <= 50
    assert chunks[0][-10:] == chunks[1][:10]
