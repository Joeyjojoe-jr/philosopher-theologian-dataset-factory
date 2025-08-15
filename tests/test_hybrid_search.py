def test_hybrid_search_rrf_ranking(hybrid_env):
    search, docs, bm25, encoder, index = hybrid_env
    results = search("alpha", docs, bm25, encoder, index, k=3)
    sources = [r["source"] for r in results]
    assert sources == ["doc_2", "doc_1", "doc_0"]
