"""M41: each model's call conventions. Query-side prompts (QUERY_PROMPT_NAMES), the dtype a model loads in
(MODEL_DTYPES, and get_cross_encoder's precision), and that the query side reaches every dense search: the fixture's
search_dense and the BEIR runner.

No torch on this machine: stand-in `sentence_transformers` and `torch` modules record what the code asks of them. The
prompts are the ones the cached models' config_sentence_transformers.json files carry.
"""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from app.eval import beir
from app.eval.latency import StageTimer
from app.retrieval import embedders as emb
from app.retrieval import rerank
from app.retrieval.embedders import HashEmbedder, SentenceTransformerEmbedder, encode_queries
from app.retrieval.hybrid import InMemoryHybridIndex
from app.retrieval.types import Chunk

QWEN_QUERY = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
HARRIER_QUERY = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
PROMPTS = {
    "BAAI/bge-small-en-v1.5": {},
    "Qwen/Qwen3-Embedding-0.6B": {"query": QWEN_QUERY, "document": ""},
    "Qwen/Qwen3-Embedding-8B": {"query": QWEN_QUERY, "document": ""},
    "microsoft/harrier-oss-v1-27b": {"web_search_query": HARRIER_QUERY, "sts_query": "Instruct: Retrieve semantically similar text\nQuery: "},
}
LOADED = {None: "torch.float32", "float32": "torch.float32", "auto": "torch.bfloat16"}  # bge checkpoints are float32, the new ones bfloat16


class _Param:
    def __init__(self, dtype):
        self.dtype = dtype


class _SentenceTransformer:
    """Records the constructor's kwargs and every encode() call; a text's vector is the hash embedding of the text the
    real library would encode (the prompt, then the text)."""

    def __init__(self, name, **kwargs):
        self.name, self.kwargs, self.calls = name, kwargs, []
        # a hub id, or a directory holding the model's files (a snapshot view): its prompts are in its own config
        self.prompts = dict(PROMPTS[name]) if name in PROMPTS else json.loads((Path(name) / "config_sentence_transformers.json").read_text())["prompts"]
        self.max_seq_length = 512

    def get_sentence_embedding_dimension(self):
        return 16

    def parameters(self):
        yield _Param(LOADED[(self.kwargs.get("model_kwargs") or {}).get("dtype")])

    def encode(self, texts, **kwargs):
        self.calls.append({"texts": list(texts), **kwargs})
        prefix = self.prompts[kwargs["prompt_name"]] if kwargs.get("prompt_name") else ""
        return HashEmbedder(dim=16).encode([prefix + t for t in texts])


@pytest.fixture
def st(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _SentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


COMMITTED_CALL = {"convert_to_numpy": True, "normalize_embeddings": True, "show_progress_bar": False}


def test_bge_gets_no_prompt_and_no_model_kwargs_so_its_rows_reproduce(st):
    e = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    assert e._model.kwargs == {} and e.query_prompt_name is None and e.batch_size is None
    queries = ["which run used 40 stereo pairs", "where is the config"]
    assert np.array_equal(e.encode_queries(queries), e.encode(queries))
    assert np.array_equal(encode_queries(e, queries), e.encode(queries))
    # every call is exactly the one the committed rows made: no prompt, no batch size, no dtype
    assert e._model.calls == [{"texts": queries, **COMMITTED_CALL}] * 4
    assert e.provenance() == {"query_prompt_name": None, "query_prompt": None, "dtype_requested": None, "dtype": "torch.float32",
                              "batch_size": None, "max_seq_length": 512, "snapshot_view": None}
    assert e._model.name == "BAAI/bge-small-en-v1.5"  # loaded by its hub id, as every committed row


def test_a_registered_model_gets_its_prompt_on_queries_only_and_loads_in_bfloat16(st):
    e = SentenceTransformerEmbedder("Qwen/Qwen3-Embedding-8B")
    assert e._model.kwargs == {"model_kwargs": {"dtype": "auto"}}
    docs, queries = ["The capital of China is Beijing."], ["What is the capital of China?"]
    e.encode(docs)
    q = e.encode_queries(queries)
    assert e._model.calls == [{"texts": docs, **COMMITTED_CALL, "batch_size": 16},
                              {"texts": queries, **COMMITTED_CALL, "batch_size": 16, "prompt_name": "query"}]
    assert np.array_equal(q, HashEmbedder(dim=16).encode([QWEN_QUERY + queries[0]]))  # the prompt is in front of the query
    assert not np.array_equal(q, e.encode(queries))  # and documents (encode) get none
    assert e.provenance() == {"query_prompt_name": "query", "query_prompt": QWEN_QUERY, "dtype_requested": "auto", "dtype": "torch.bfloat16",
                              "batch_size": 16, "max_seq_length": 512, "snapshot_view": None}
    assert e._model.name == "Qwen/Qwen3-Embedding-8B"


@pytest.fixture
def harrier_snapshot(tmp_path, monkeypatch):
    """A stand-in for harrier's cached snapshot: its tokenizer_config.json names the image processor the checkpoint does
    not ship; a weights file and the pooling folder stand for the rest."""
    snap = tmp_path / "snapshots" / "0c0fc62"
    (snap / "1_Pooling").mkdir(parents=True)
    (snap / "1_Pooling" / "config.json").write_text(json.dumps({"pooling_mode_lasttoken": True, "include_prompt": True}))
    (snap / "tokenizer_config.json").write_text(json.dumps({"processor_class": "Gemma3Processor", "add_bos_token": True, "add_eos_token": True}))
    (snap / "config_sentence_transformers.json").write_text(json.dumps({"prompts": {k: v for k, v in PROMPTS["microsoft/harrier-oss-v1-27b"].items()}}))
    (snap / "model-00001-of-00012.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(emb, "_snapshot_path", lambda name: snap if name == "microsoft/harrier-oss-v1-27b" else pytest.fail(name))
    return snap


def test_harrier_loads_from_a_view_of_its_snapshot_without_the_image_processor_key(st, harrier_snapshot):
    """sentence-transformers 6.1 reads processor_class Gemma3Processor from harrier's tokenizer_config.json and fails on
    the image processor the text-only checkpoint lacks; the model loads from a view that drops only that key."""
    e = SentenceTransformerEmbedder("microsoft/harrier-oss-v1-27b")
    view = Path(e._model.name)
    assert view != harrier_snapshot and e._model.kwargs == {"model_kwargs": {"dtype": "auto"}}
    assert json.loads((view / "tokenizer_config.json").read_text()) == {"add_bos_token": True, "add_eos_token": True}
    for name in ("model-00001-of-00012.safetensors", "config_sentence_transformers.json", "1_Pooling"):
        assert (view / name).is_symlink() and (view / name).resolve() == (harrier_snapshot / name).resolve()  # the snapshot's own
    assert json.loads((harrier_snapshot / "tokenizer_config.json").read_text())["processor_class"] == "Gemma3Processor"  # cache untouched
    shim = e.provenance()["snapshot_view"]
    assert shim["file"] == "tokenizer_config.json" and shim["dropped"] == {"processor_class": "Gemma3Processor"}
    assert e.query_prompt == HARRIER_QUERY  # the prompts still come from the model's own config


def test_a_snapshot_view_that_no_longer_applies_raises(st, harrier_snapshot):
    (harrier_snapshot / "tokenizer_config.json").write_text(json.dumps({"add_eos_token": True}))  # an upstream fix
    with pytest.raises(RuntimeError, match="no longer applies"):
        SentenceTransformerEmbedder("microsoft/harrier-oss-v1-27b")


def test_the_cpu_servable_embedder_is_named_float32_and_harrier_uses_its_web_search_prompt(st, harrier_snapshot):
    small = SentenceTransformerEmbedder("Qwen/Qwen3-Embedding-0.6B")
    assert small._model.kwargs == {"model_kwargs": {"dtype": "float32"}} and small.provenance()["dtype"] == "torch.float32"
    small.encode_queries(["q"])
    assert small._model.calls[-1] == {"texts": ["q"], **COMMITTED_CALL, "prompt_name": "query"}  # library batch size
    big = SentenceTransformerEmbedder("microsoft/harrier-oss-v1-27b")
    assert big._model.kwargs == {"model_kwargs": {"dtype": "auto"}} and big.query_prompt == HARRIER_QUERY
    big.encode_queries(["q"])
    assert big._model.calls[-1] == {"texts": ["q"], **COMMITTED_CALL, "batch_size": 8, "prompt_name": "web_search_query"}
    # every model with a query prompt names its dtype: transformers >= 5 would otherwise load its bfloat16 checkpoint as is
    assert set(emb.QUERY_PROMPT_NAMES) <= set(emb.MODEL_DTYPES)


def test_a_missing_prompt_raises_instead_of_silently_embedding_the_bare_query(st, monkeypatch):
    monkeypatch.setitem(PROMPTS, "Qwen/Qwen3-Embedding-0.6B", {"document": ""})
    with pytest.raises(ValueError, match="no prompt named 'query'"):
        SentenceTransformerEmbedder("Qwen/Qwen3-Embedding-0.6B")


def test_an_embedder_without_a_query_side_encodes_queries_as_documents():
    e = HashEmbedder(dim=32)
    assert np.array_equal(encode_queries(e, ["a query", "b"]), e.encode(["a query", "b"]))


class _Recorder:
    """An embedder with both sides, recording which side each text went through."""

    name, dim = "recorder", 16

    def __init__(self):
        self.docs, self.queries = [], []

    def encode(self, texts):
        self.docs.append(list(texts))
        return HashEmbedder(dim=16).encode(texts)

    def encode_queries(self, texts):
        self.queries.append(list(texts))
        return HashEmbedder(dim=16).encode(["query side: " + t for t in texts])


def _chunks():
    texts = ["stereo pairs for run 40", "config of the camera rig", "a log of run 47"]
    return [Chunk(chunk_id=f"c{i}", path=f"f{i}.txt", text=t, start_byte=0, end_byte=len(t)) for i, t in enumerate(texts)]


def test_search_dense_embeds_the_query_through_the_query_side():
    rec = _Recorder()
    index = InMemoryHybridIndex(_chunks(), embedder=rec, enable_router=False, enable_rerank=False)
    assert rec.docs == [[c.text for c in _chunks()]] and rec.queries == []  # documents at build time, through encode()
    index.search_dense("which run used stereo pairs")
    index.retrieve("camera rig config")
    assert rec.queries == [["which run used stereo pairs"], ["camera rig config"]]
    assert len(rec.docs) == 1  # no query went through the document side
    # a caller's own query vector (first-stage HyDE) still replaces the query's embedding
    index.search_dense("ignored", query_vec=np.ones(16, dtype=np.float32))
    assert rec.queries == [["which run used stereo pairs"], ["camera rig config"]]


def _engine(rec):
    engine = beir.Engine(_chunks(), device="cpu")
    engine._embedders["st:recorder"] = rec
    engine.provenance["st:recorder"] = {"model": "recorder", "revision": None, "query_instruction": None,
                                        "query_prompt_name": "query", "query_prompt": "query side: "}
    return engine


def test_beir_prepends_a_typed_instruction_and_otherwise_uses_the_query_side():
    rec = _Recorder()
    engine = _engine(rec)
    typed = beir.BeirConfig(name="dense@recorder+instruction", mode="dense", embedder="st:recorder", query_instruction="query: ")
    plain = beir.BeirConfig(name="dense@recorder", mode="dense", embedder="st:recorder")
    beir.retrieve(engine, typed, "stereo pairs", StageTimer())
    assert rec.docs[-1] == ["query: stereo pairs"] and rec.queries == []  # the bge "+instruction" row, as before
    beir.retrieve(engine, plain, "stereo pairs", StageTimer())
    assert rec.queries == [["stereo pairs"]]
    queries, qrels = {"q1": "stereo pairs"}, {"q1": {"c0": 1}}
    with_prompt = beir.run_config(engine, plain, queries, qrels)["models"]["embedder"]
    assert (with_prompt["query_prompt_name"], with_prompt["query_instruction"]) == ("query", None)
    replaced = beir.run_config(engine, typed, queries, qrels)["models"]["embedder"]
    assert (replaced["query_prompt_name"], replaced["query_prompt"], replaced["query_instruction"]) == (None, None, "query: ")
    assert with_prompt["doc_embeddings"] == "computed in this run"


# ---- the reranker's dtype (get_cross_encoder) and what a row records about it


class _CrossEncoder:
    made = []

    def __init__(self, name, **kwargs):
        if name == "broken/model":
            raise OSError("model.safetensors is truncated")
        self.name, self.kwargs = name, kwargs
        _CrossEncoder.made.append(self)


def _torch(cuda: bool):
    return types.SimpleNamespace(float16="torch.float16", bfloat16="torch.bfloat16", float32="torch.float32",
                                 cuda=types.SimpleNamespace(is_available=lambda: cuda))


@pytest.fixture
def cross_encoder(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = _CrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(rerank, "_hf_cache_has", lambda name: name != "not/cached")
    monkeypatch.delenv("BGE_ALLOW_DOWNLOAD", raising=False)
    return monkeypatch


@pytest.mark.parametrize("precision, device, cuda, dtype", [
    ("fp32", None, True, "torch.float32"),    # named: a bfloat16 checkpoint (Qwen3-Reranker, mxbai) would load as bfloat16
    ("fp32", "cpu", False, "torch.float32"),
    ("fp16", None, True, "torch.float16"),
    ("bf16", None, True, "torch.bfloat16"),
    ("bf16", "cpu", True, "torch.float32"),   # half precisions are CUDA only
    ("fp16", None, False, "torch.float32"),
])
def test_the_precision_is_the_dtype_the_weights_load_in(cross_encoder, precision, device, cuda, dtype):
    cross_encoder.setitem(sys.modules, "torch", _torch(cuda))
    model = rerank.get_cross_encoder("Qwen/Qwen3-Reranker-8B", precision=precision, max_length=512, device=device)
    assert model.kwargs["model_kwargs"] == {"torch_dtype": dtype} and model.kwargs["max_length"] == 512
    assert model.kwargs["local_files_only"] is True


def test_a_load_failure_is_none_unless_the_caller_requires_the_model(cross_encoder):
    cross_encoder.setitem(sys.modules, "torch", _torch(False))
    assert rerank.get_cross_encoder("broken/model") is None
    with pytest.raises(OSError, match="truncated"):
        rerank.get_cross_encoder("broken/model", raise_errors=True)
    assert rerank.get_cross_encoder("not/cached") is None
    with pytest.raises(FileNotFoundError, match="not in the Hugging Face cache"):
        rerank.get_cross_encoder("not/cached", raise_errors=True)


def test_the_beir_runner_stops_with_the_reason_a_reranker_did_not_load(monkeypatch):
    def failing(name, **kwargs):
        assert kwargs["raise_errors"] is True
        raise OSError(f"{name}: weights missing")

    monkeypatch.setattr(beir, "get_cross_encoder", failing)
    with pytest.raises(OSError, match="weights missing"):
        beir.Engine(_chunks(), device="cpu").reranker("mixedbread-ai/mxbai-rerank-base-v2")


def test_bf16_rows_need_cuda_in_the_beir_runner(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _torch(False))
    cfg = beir.BeirConfig(name="hybrid+qwen3-rerank-8b", mode="bm25", reranker="Qwen/Qwen3-Reranker-8B", precision="bf16")
    with pytest.raises(RuntimeError, match="bf16 ablations require CUDA"):
        beir.run_config(beir.Engine(_chunks(), device="cpu"), cfg, {"q1": "x"}, {"q1": {"c0": 1}})


def test_cross_encoder_provenance_names_the_dtype_cap_and_default_prompt():
    class Loaded:
        default_prompt_name = "query"
        prompts = {"query": "Given a web search query, retrieve relevant passages that answer the query"}
        max_seq_length = 512

        def parameters(self):
            yield _Param("torch.bfloat16")

    assert rerank.cross_encoder_provenance(Loaded()) == {
        "dtype": "torch.bfloat16", "loaded_max_length": 512, "default_prompt_name": "query",
        "default_prompt": "Given a web search query, retrieve relevant passages that answer the query"}
    bare = types.SimpleNamespace(max_length=8192)  # an older CrossEncoder: no prompts, max_length only
    assert rerank.cross_encoder_provenance(bare) == {"dtype": None, "loaded_max_length": 8192, "default_prompt_name": None, "default_prompt": None}
