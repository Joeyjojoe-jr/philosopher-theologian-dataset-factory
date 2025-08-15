import pytest


def test_hybrid_search_rrf_ranking(hybrid_env):
    search, docs, bm25, encoder, index = hybrid_env
    results = search("alpha", docs, bm25, encoder, index, k=3)

    # ensure ordering respects reciprocal rank fusion between dense and sparse
    assert [r["source"] for r in results] == ["doc_2", "doc_1", "doc_0"]

    # RRF uses 1/(60 + rank); verify combined scores from both systems
    expected = {
        "doc_2": 1 / (60 + 2) + 1 / (60 + 1),
        "doc_1": 1 / (60 + 1) + 1 / (60 + 3),
        "doc_0": 1 / (60 + 3) + 1 / (60 + 2),
    }
    scores = {r["source"]: r["score"] for r in results}
    assert scores == pytest.approx(expected)
