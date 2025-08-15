from src.retrieval import SimpleRetriever


def test_simple_retrieval():
    corpus = [
        "Gratia non tollit naturam, sed perficit",
        "Homo est animal rationale",
    ]
    retriever = SimpleRetriever(corpus)
    results = retriever.search("gratia naturam", k=1)
    assert results[0][0] == corpus[0]
