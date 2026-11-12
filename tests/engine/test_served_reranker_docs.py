"""B27: the served retriever's module docstring and the query_index tool's description (the text the chat model
reads about the tool) state the served defaults, not the top-50 rerank they used to describe: the cross-encoder
reranks the fused top RERANK_DEPTH and the rest keep their RRF order."""

from app.engine import retriever as mod
from app.engine.tools import query_engine
from app.retrieval.fuse import RRF_K

SETTINGS = ("RETRIEVE_K", "RERANK_TOP_N", "TOP_K", "RERANK_DEPTH", "ENABLE_ROUTER", "RETRIEVAL_MODE")


def _served_defaults(monkeypatch) -> dict[str, int]:
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)
    built = mod.create_hybrid_retriever(None, object(), use_reranker=False, mode="sql", embed_fn=lambda q: [0.0])
    return {"RETRIEVE_K": built.depth, "RERANK_TOP_N": built._rerank_top_n, "RERANK_DEPTH": mod.reranker_settings()["depth"]}


def test_retriever_docstring_states_the_served_defaults(monkeypatch):
    doc = " ".join((mod.__doc__ or "").split())
    for name, value in _served_defaults(monkeypatch).items():
        assert f"{name} (default {value})" in doc, (name, value)
    assert "top-50" not in doc and "top-8" not in doc


def test_query_index_tool_description_states_the_served_defaults(monkeypatch):
    defaults = _served_defaults(monkeypatch)
    text = query_engine.QUERY_INDEX_DESCRIPTION
    assert f"RRF (k={RRF_K})" in text
    for name in ("RERANK_DEPTH", "RERANK_TOP_N"):
        assert f"{name} (default {defaults[name]})" in text, (name, defaults[name])
    assert "top-50" not in text and "top-8" not in text
