from pathlib import Path

import pytest

from src.chunk_and_index import chunk_text


def test_chunking_overlap():
    text = Path("tests/fixtures/aquinas.txt").read_text()
    chunks = chunk_text(text, max_chars=50, overlap=10)
    assert len(chunks) == 2
    assert len(chunks[0]) <= 50
    assert chunks[0][-10:] == chunks[1][:10]


@pytest.mark.parametrize(
    "max_chars,overlap",
    [
        (10, 10),  # overlap equal to chunk size
        (0, 0),  # non-positive chunk size
        (10, -1),  # negative overlap
    ],
)
def test_chunk_text_invalid_args(max_chars, overlap):
    with pytest.raises(ValueError):
        chunk_text("lorem", max_chars=max_chars, overlap=overlap)
