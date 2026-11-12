"""M17 (A19): the served embedder's query instruction is a setting, EMBEDDING_QUERY_INSTRUCTION.

LlamaIndex's HuggingFaceEmbedding puts its per-model instruction in front of every query (for the BGE English
models "Represent this question for searching relevant passages: "); every committed model-backed benchmark row
embeds the bare query (app.retrieval.embedders.SentenceTransformerEmbedder). "none" makes the served path embed
queries as the benchmark does. Without torch here: a stand-in with the sentence-transformers `prompts` dict the
real class keeps (llama-index-embeddings-huggingface hands its instruction over as the "query" prompt).
"""

import sys
import types

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.settings import Settings
from pydantic import PrivateAttr

from app.settings import init_embed_model, query_instruction, query_instruction_setting, set_query_instruction

BGE_INSTRUCTION = "Represent this question for searching relevant passages: "


class _StandInHuggingFaceEmbedding(BaseEmbedding):
    """What init_embed_model reads of HuggingFaceEmbedding: the loaded sentence-transformers model's prompts."""

    _model: object = PrivateAttr()

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name)
        self._model = types.SimpleNamespace(prompts={"query": BGE_INSTRUCTION, "text": ""})

    def _get_query_embedding(self, query):
        return [0.0]

    async def _aget_query_embedding(self, query):
        return [0.0]

    def _get_text_embedding(self, text):
        return [0.0]


@pytest.fixture
def huggingface(monkeypatch):
    """init_embed_model's `from llama_index.embeddings.huggingface import HuggingFaceEmbedding`, answered by the stand-in."""
    package = types.ModuleType("llama_index.embeddings")
    module = types.ModuleType("llama_index.embeddings.huggingface")
    module.HuggingFaceEmbedding = _StandInHuggingFaceEmbedding
    monkeypatch.setitem(sys.modules, "llama_index.embeddings", package)
    monkeypatch.setitem(sys.modules, "llama_index.embeddings.huggingface", module)
    monkeypatch.setattr(Settings, "_embed_model", None)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    for name in ("EMBEDDING_DEVICE", "RETRIEVAL_REQUIRE_DEVICE", "EMBEDDING_QUERY_INSTRUCTION"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("value, expected", [(None, None), ("", None), ("default", None), ("DEFAULT", None), ("none", ""), ("None", ""),
                                             ("query: ", "query: ")])
def test_the_setting(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("EMBEDDING_QUERY_INSTRUCTION", raising=False)
    else:
        monkeypatch.setenv("EMBEDDING_QUERY_INSTRUCTION", value)
    assert query_instruction_setting() == expected


def test_by_default_the_served_embedder_keeps_llamaindexs_instruction(huggingface):
    init_embed_model()
    assert query_instruction(Settings.embed_model) == BGE_INSTRUCTION


def test_none_makes_the_served_embedder_embed_the_bare_query(huggingface, monkeypatch):
    monkeypatch.setenv("EMBEDDING_QUERY_INSTRUCTION", "none")
    init_embed_model()
    assert query_instruction(Settings.embed_model) == ""
    assert Settings.embed_model._model.prompts == {"query": "", "text": ""}  # documents unchanged


def test_an_embedder_without_prompts_is_an_error_not_a_no_op():
    with pytest.raises(RuntimeError, match="no sentence-transformers query prompt"):
        set_query_instruction(types.SimpleNamespace(_model=object()), "")


def test_the_hash_embedder_has_no_instruction():
    from app.engine.embeddings import HashEmbedding

    assert query_instruction(HashEmbedding(16)) == ""
    assert query_instruction(object()) is None
