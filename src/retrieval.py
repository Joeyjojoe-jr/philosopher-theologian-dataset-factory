"""Simple retrieval module using token overlap scoring."""

from typing import List, Tuple


class SimpleRetriever:
    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        self.index = [doc.lower().split() for doc in corpus]

    def search(self, query: str, k: int = 3) -> List[Tuple[str, int]]:
        q_tokens = query.lower().split()
        scores = []
        for tokens in self.index:
            score = sum(1 for t in tokens if t in q_tokens)
            scores.append(score)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked[:k]:
            if score > 0:
                results.append((self.corpus[idx], score))
        return results
