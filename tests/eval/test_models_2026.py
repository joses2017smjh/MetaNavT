"""M41 (make bench-models-2026): the current open embedders and rerankers on fixture v1, each one change from a
committed bge row of bench/results/neural.json that the same run recomputes.

No torch here: the cross-encoder is a stand-in scoring by token overlap, the embedder the hash embedder. What is
tested is the wiring: which rows exist and what each changes, that a model that does not load or cannot rank stops
the run (no overlap fallback, nothing written), what a row records, and the reproduction check.
"""

import json
import math
from dataclasses import fields
from pathlib import Path

import pytest

from app.eval import harness, models_2026
from app.eval.gold import load_gold
from app.eval.harness import BenchConfig, models_2026_comparisons, models_2026_configs, run_bench, run_config
from app.eval.index_loader import build_index
from app.retrieval.bm25 import tokenize

ROOT = Path(__file__).resolve().parents[2]
FILES = ROOT / "bench" / "corpus" / "files"


@pytest.fixture(scope="module")
def gold():
    return load_gold(ROOT / "bench" / "gold" / "questions.jsonl")


def _changed(cfg, parent):
    return {f.name for f in fields(BenchConfig) if f.name not in {"name", "parent"} and getattr(cfg, f.name) != getattr(parent, f.name)}


def test_every_new_row_is_one_model_away_from_a_committed_bge_row():
    configs = models_2026_configs("main")
    by = {c.name: c for c in configs}
    assert [c.name for c in configs[:4]] == list(harness.MODELS_2026_PARENTS)
    neural = {c.name: c for c in harness.NEURAL_CONFIGS}
    for name in harness.MODELS_2026_PARENTS:  # the parents are neural.json's rows, unchanged
        assert by[name] == neural[name]
    assert set(by) == set(harness.MODELS_2026_PARENTS) | {
        "dense_only@qwen3-emb-0.6b", "hybrid@qwen3-emb-0.6b", "dense_only@qwen3-emb-8b", "hybrid@qwen3-emb-8b",
        "hybrid+mxbai-rerank-base-v2@bge-small", "hybrid+mxbai-rerank-large-v2@bge-small",
        "hybrid+qwen3-rerank-0.6b@bge-small", "hybrid+qwen3-rerank-8b@bge-small"}
    for cfg in configs[4:]:
        parent = by[cfg.parent]
        changed = _changed(cfg, parent)
        if cfg.name.startswith(("dense_only@", "hybrid@")):
            assert changed == {"embedder"} and parent.name.endswith("@bge-small"), cfg.name
        else:  # a reranker at its parent's settings; Qwen3-Reranker-8B also in bf16
            assert parent.name == "hybrid+bge-rerank@bge-small"
            assert changed == ({"reranker", "rerank_precision"} if "8b" in cfg.name else {"reranker"}), (cfg.name, changed)
    assert by["hybrid+qwen3-rerank-8b@bge-small"].rerank_precision == "bf16"
    assert "harrier" not in " ".join(by)  # 54 GB of weights: its own run on an 80 GB GPU


def test_the_harrier_and_cpu_sets():
    harrier = {c.name: c for c in models_2026_configs("harrier")}
    assert set(harrier) == {"bm25_only", "dense_only@bge-small", "hybrid@bge-small", "dense_only@harrier-27b", "hybrid@harrier-27b"}
    assert harrier["hybrid@harrier-27b"].embedder == "st:microsoft/harrier-oss-v1-27b"
    cpu = {c.name: c for c in models_2026_configs("cpu")}
    served = "hybrid+bge-rerank@bge-small/fp32-512/top20"  # the served CPU settings, as rerank_depth.json's row
    assert set(cpu) == {"bm25_only", "dense_only@bge-small", "hybrid@bge-small", served, "dense_only@qwen3-emb-0.6b",
                        "hybrid@qwen3-emb-0.6b", "hybrid+mxbai-rerank-base-v2@bge-small/fp32-512/top20",
                        "hybrid+qwen3-rerank-0.6b@bge-small/fp32-512/top20"}
    for name in (served, "hybrid+mxbai-rerank-base-v2@bge-small/fp32-512/top20", "hybrid+qwen3-rerank-0.6b@bge-small/fp32-512/top20"):
        c = cpu[name]
        assert (c.rerank_depth, c.rerank_max_length, c.rerank_precision) == (20, 512, "fp32")
        assert c.parent == ("hybrid@bge-small" if name == served else served)
    with pytest.raises(ValueError, match="main, harrier or cpu"):
        models_2026_configs("everything")


def test_the_comparisons_pair_hybrid_with_its_own_dense_row_and_each_reranker_with_the_unreranked_hybrid():
    pairs = models_2026_comparisons(models_2026_configs("main"))
    assert ("hybrid@qwen3-emb-8b", "dense_only@qwen3-emb-8b") in pairs
    assert ("hybrid+qwen3-rerank-8b@bge-small", "hybrid@bge-small") in pairs
    assert ("hybrid+bge-rerank@bge-small", "hybrid@bge-small") not in pairs  # that one is its parent delta already


def test_the_cli_flag():
    assert harness._parse_args(["--models-2026"]).models_2026 == "main"
    assert harness._parse_args(["--models-2026", "harrier"]).models_2026 == "harrier"
    assert harness._parse_args([]).models_2026 is None
    with pytest.raises(SystemExit):
        harness._parse_args(["--models-2026", "all"])


# ---- the load-and-score check


class _OverlapCrossEncoder:
    """A stand-in CrossEncoder: the logit is the share of query tokens in the passage, minus a bias."""

    default_prompt_name, prompts, max_seq_length = None, {}, 512

    def __init__(self, bias=0.5, nan=False, inverted=False):
        self.bias, self.nan, self.inverted = bias, nan, inverted

    def predict(self, pairs, batch_size=32, show_progress_bar=False, activation_fn=None):
        out = []
        for q, d in pairs:
            qt, dt = set(tokenize(q)), set(tokenize(d))
            x = len(qt & dt) / max(len(qt), 1) - self.bias
            x = -x if self.inverted else x
            out.append(float("nan") if self.nan else (activation_fn(x) if activation_fn else 1 / (1 + math.exp(-x))))
        return out

    def parameters(self):
        yield type("P", (), {"dtype": "torch.float32"})()


def test_the_reranker_check_passes_a_model_that_ranks_and_stops_one_that_cannot():
    out = models_2026.check_reranker(_OverlapCrossEncoder(), "stand-in")
    assert out["passed"] and out["score_answer"] > out["score_other"] and out["score_answer_long"] > out["score_other"]
    with pytest.raises(models_2026.ModelCheckFailed, match="does not outscore"):
        models_2026.check_reranker(_OverlapCrossEncoder(inverted=True), "inverted")
    with pytest.raises(models_2026.ModelCheckFailed, match="not finite"):
        models_2026.check_reranker(_OverlapCrossEncoder(nan=True), "nan")


def test_the_embedder_check():
    class Keyword:  # one dimension per check word: the answer shares the query's words, the other passage none
        name, words = "keyword", ("capital", "china", "gravity")

        def encode(self, texts):
            import numpy as np
            return np.array([[float(w in t.lower()) + 0.01 for w in self.words] for t in texts])

        def encode_queries(self, texts):
            return self.encode(texts)

    out = models_2026.check_embedder(Keyword())
    assert out["passed"] and out["cosine_answer"] > out["cosine_other"]

    class Backwards(Keyword):
        def encode_queries(self, texts):
            return self.encode(["gravity"])

    with pytest.raises(models_2026.ModelCheckFailed):
        models_2026.check_embedder(Backwards())


@pytest.fixture
def stand_in_cross_encoder(monkeypatch):
    calls = []

    def load(name, precision="fp32", max_length=None, device=None, raise_errors=False):
        calls.append({"name": name, "precision": precision, "max_length": max_length, "raise_errors": raise_errors})
        if name == "broken/reranker":  # what get_cross_encoder does with a load that fails
            if raise_errors:
                raise OSError("model.safetensors is truncated")
            return None
        return _OverlapCrossEncoder(inverted=name == "inverted/reranker")

    monkeypatch.setattr("app.eval.index_loader.get_cross_encoder", load)
    monkeypatch.setattr("app.retrieval.rerank.get_cross_encoder", load)
    return calls


def _row(name, reranker):
    return BenchConfig(name=name, mode="hybrid", embedder="hash", reranker=reranker, enable_rerank=True, enable_router=False,
                       staleness_tier1=False, e2e=False, log_triples=False)


def test_a_checked_row_records_its_check_and_its_reranker_without_falling_back(gold, stand_in_cross_encoder):
    row = run_config(_row("hybrid+stand-in@hash", "some/reranker"), gold, FILES, mtimes="manifest", check_models=True)
    assert stand_in_cross_encoder == [{"name": "some/reranker", "precision": "fp32", "max_length": None, "raise_errors": True}]
    assert row["reranker_loaded"] is True and "reranker_fallback" not in row
    assert row["load_and_score"]["reranker"]["passed"] is True and "embedder" not in row["load_and_score"]  # hash: no model
    r = row["models"]["reranker"]
    assert (r["precision"], r["dtype"], r["loaded_max_length"], r["default_prompt_name"]) == ("fp32", "torch.float32", 512, None)


def test_a_model_that_does_not_load_or_cannot_rank_stops_the_row(gold, stand_in_cross_encoder):
    with pytest.raises(OSError, match="truncated"):
        run_config(_row("hybrid+broken@hash", "broken/reranker"), gold, FILES, mtimes="manifest", check_models=True)
    with pytest.raises(models_2026.ModelCheckFailed):
        run_config(_row("hybrid+inverted@hash", "inverted/reranker"), gold, FILES, mtimes="manifest", check_models=True)
    # unchecked (every committed ladder) a load failure still falls back to the overlap reranker, and says so
    row = run_config(_row("hybrid+broken@hash", "broken/reranker"), gold, FILES, mtimes="manifest")
    assert row["reranker_loaded"] is False and row["reranker_fallback"] == "overlap" and "load_and_score" not in row


def test_half_precisions_need_cuda_and_other_names_are_refused(monkeypatch):
    monkeypatch.setattr("app.eval.index_loader._cuda_available", lambda: False)
    for precision in ("fp16", "bf16"):
        with pytest.raises(RuntimeError, match=f"at {precision} needs CUDA"):
            build_index(FILES, embedder_name="hash", reranker="Qwen/Qwen3-Reranker-8B", rerank_precision=precision)
    with pytest.raises(ValueError, match="fp32, fp16 or bf16"):
        build_index(FILES, embedder_name="hash", reranker="overlap", rerank_precision="fp8")


# ---- the reproduction check and the results header


def _results(**retrieval_by_name):
    return [{"config": n, "retrieval": r} for n, r in retrieval_by_name.items()]


def test_reproduces_compares_every_field_of_the_retrieval_block(tmp_path):
    (tmp_path / "bench").mkdir()
    (tmp_path / "bench" / "x.json").write_text(json.dumps({"git_sha": "abc1234", "device": {"device_name": "Quadro RTX 8000"},
                                                           "results": _results(a={"ndcg@10": 0.5, "by_category": {"x": 0.1}},
                                                                               b={"ndcg@10": 0.6})}))
    same = models_2026.reproduces(tmp_path, "bench/x.json", _results(a={"ndcg@10": 0.5, "by_category": {"x": 0.1}}), ["a"])
    assert same["all"] is True and same["rows"] == {"a": {"equal": True}} and same["device"] == "Quadro RTX 8000"
    moved = models_2026.reproduces(tmp_path, "bench/x.json", _results(a={"ndcg@10": 0.5, "by_category": {"x": 0.2}}, b={"ndcg@10": 0.6}), ["a", "b", "c"])
    assert moved["all"] is False and moved["rows"]["a"] == {"equal": False, "differs": {"by_category": [{"x": 0.1}, {"x": 0.2}]}}
    assert moved["rows"]["b"] == {"equal": True} and "c" not in moved["rows"]  # c is not in this run
    assert models_2026.reproduces(tmp_path, "bench/missing.json", [], ["a"]) == {"file": "bench/missing.json", "found": False, "all": False}


def test_a_checked_run_records_the_parents_reproduction_and_writes_nothing_else(tmp_path):
    """A hash-embedder run of the same machinery: its bm25_only row is main.json's, recomputed."""
    configs = [BenchConfig(name="bm25_only", mode="bm25", enable_rerank=False, enable_router=False, staleness_tier1=False, e2e=False,
                           log_triples=False),
               BenchConfig(name="hybrid@hash", mode="hybrid", enable_rerank=False, enable_router=False, staleness_tier1=False, e2e=False,
                           log_triples=False, parent="bm25_only")]
    out = tmp_path / "models_2026_check.json"
    blob = run_bench(configs=configs, out=out, check_models=True, compare_lists=True,
                     reproduce={"bench/results/main.json": ("bm25_only", "not_in_this_run"), "bench/results/neural.json": ("dense_only@bge-small",)},
                     header={"models_2026": {"set": "test"}})
    written = json.loads(out.read_text())
    assert [b["file"] for b in written["reproduces"]] == ["bench/results/main.json"]  # neural.json: none of its rows ran
    assert written["reproduces"][0]["rows"] == {"bm25_only": {"equal": True}} and written["reproduces_all"] is True
    assert written["models_2026"] == {"set": "test"} and blob["results"][1]["vs_parent"]["parent"] == "bm25_only"
    assert (tmp_path / "models_2026_check.per_query.json").exists()
