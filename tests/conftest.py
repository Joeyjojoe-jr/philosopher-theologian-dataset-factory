import json
import sys
import types
import importlib
from types import ModuleType
from pathlib import Path

import pytest


class DummyBM25:
    def __init__(self, scores):
        self.scores = scores

    def get_scores(self, tokens):
        return self.scores


class DummyEncoder:
    def encode(self, inputs, show_progress_bar=False):
        return [[0.0]]  # minimal embedding


class DummyIndex:
    def search(self, q_emb, k):
        dense_ids = [[2, 0, 1]]  # predefined order
        dense_scores = [[1.0, 0.5, 0.1]]
        return dense_scores, dense_ids


@pytest.fixture
def hybrid_env(monkeypatch):
    """Provide components for _hybrid_search without heavy deps."""
    fake_faiss = ModuleType("faiss")
    fake_faiss.normalize_L2 = lambda x: None
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)

    fake_np = ModuleType("numpy")
    fake_np.argsort = lambda arr: sorted(range(len(arr)), key=lambda i: arr[i])
    fake_np.isscalar = lambda x: isinstance(x, (int, float))
    fake_np.bool_ = bool
    monkeypatch.setitem(sys.modules, "numpy", fake_np)

    fake_bm25 = ModuleType("rank_bm25")
    fake_bm25.BM25Okapi = object
    monkeypatch.setitem(sys.modules, "rank_bm25", fake_bm25)

    fake_st = ModuleType("sentence_transformers")
    class _ST:
        def encode(self, texts, show_progress_bar=False):
            return [[0.0]]
    fake_st.SentenceTransformer = lambda *a, **k: _ST()
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    fake_tf = ModuleType("transformers")
    fake_tf.AutoModelForCausalLM = object
    fake_tf.AutoTokenizer = object
    fake_tf.set_seed = lambda x: None
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    debate_loop = importlib.import_module("src.debate_loop")
    importlib.reload(debate_loop)

    docs = ["alpha beta", "gamma delta", "alpha gamma"]
    bm25 = DummyBM25([0.1, 0.3, 0.2])  # Ranks doc1 > doc2 > doc0
    encoder = DummyEncoder()
    index = DummyIndex()
    return debate_loop._hybrid_search, docs, bm25, encoder, index


@pytest.fixture
def valid_sft_jsonl(tmp_path):
    """Create a minimal JSONL file matching the SFT schema."""
    sample = {
        "id": "1",
        "instruction": "Do something",
        "response": "A result",
        "meta": {
            "speaker": "X",
            "topic": "Y",
            "citations": [{"work": "W", "ref": "R"}],
            "provenance": [{"work": "W", "ref": "R", "snippet": "S"}],
            "audit_summary": {"claims": 0, "correct": 0, "support_rate": 0.0},
            "batch_id": "b1",
            "encoder": "enc",
            "model": "mod",
        },
    }
    path = tmp_path / "sample.jsonl"
    path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    return path
