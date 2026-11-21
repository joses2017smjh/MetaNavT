"""M41 (make bench-models-2026): the current open embedders and rerankers on fixture v1, each one change from a
committed bge row of bench/results/neural.json that the same run recomputes.

No torch here: the cross-encoder is a stand-in scoring by token overlap, the embedder the hash embedder. What is
tested is the wiring: which rows exist and what each changes, that a model that does not load or cannot rank stops
the run (no overlap fallback, nothing written), what a row records, and the reproduction check.
"""

import hashlib
import json
import math
from dataclasses import asdict, fields
from pathlib import Path

import pytest

from app.eval import beir, harness, models_2026
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


# ---- M41 on BEIR (make bench-beir-models-2026, bench-beir-fiqa-models-2026): the new models beside the committed bge
# rows recomputed in the same run, on a synthetic six-document dataset with stand-in models. Tested: the ladder (each
# row one change from a committed row), that the runner releases a model after its last row, the extra paired
# comparisons, the ranking hash a recomputed parent is checked with, and the load-and-score check.


def _write_beir_dataset(root):
    """Six documents, two judged queries (the layout BEIR's archives extract to)."""
    d = root / "scifact"
    (d / "qrels").mkdir(parents=True)
    docs = {"d1": ("Cell division", "Cells divide rapidly during embryonic growth."), "d2": ("Star formation", "Stars form in collapsing molecular clouds."),
            "d3": ("Vaccines", "Vaccination reduces measles incidence in children."), "d4": ("Diet", "Dietary fibre intake lowers colon cancer risk."),
            "d5": ("Sleep", "Sleep deprivation impairs memory consolidation."), "d6": ("Exercise", "Aerobic exercise improves cardiac output.")}
    (d / "corpus.jsonl").write_text("".join(json.dumps({"_id": k, "title": t, "text": x}) + "\n" for k, (t, x) in docs.items()))
    (d / "queries.jsonl").write_text("".join(json.dumps({"_id": k, "text": x}) + "\n" for k, x in
                                             {"q1": "do cells divide during growth", "q2": "vaccination and measles in children"}.items()))
    (d / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td3\t1\n")


@pytest.fixture
def snowball():
    pytest.importorskip("snowballstemmer")  # the BEIR analyzer's stemmer


def _beir_changed(cfg, parent):
    a, b = asdict(cfg), asdict(parent)
    return {k for k in a if k not in {"name", "parent"} and a[k] != b[k]}


def test_every_beir_row_is_one_model_away_from_a_committed_bge_row():
    ladder = beir.models_2026_configs("main")
    by = {c.name: c for c in ladder}
    assert [c.name for c in ladder[:4]] == list(beir.MODELS_2026_PARENTS)
    default = {c.name: c for c in beir.DEFAULT_CONFIGS}
    for name in beir.MODELS_2026_PARENTS:  # the committed rows' settings; the capped reranker row's parent is hybrid, as in beir_fiqa.json
        assert _beir_changed(by[name], default[name]) == set()
    assert by[beir.RERANK_FP32_512].parent == "hybrid@bge-small"
    assert set(by) - set(beir.MODELS_2026_PARENTS) == {
        "dense@qwen3-emb-0.6b", "hybrid@qwen3-emb-0.6b", "dense@qwen3-emb-8b", "hybrid@qwen3-emb-8b",
        "hybrid+mxbai-rerank-base-v2@bge-small/fp32-512/top20", "hybrid+mxbai-rerank-large-v2@bge-small/fp32-512/top20",
        "hybrid+qwen3-rerank-0.6b@bge-small/fp32-512/top20", "hybrid+qwen3-rerank-8b@bge-small/bf16-512/top20"}
    for cfg in ladder[4:]:
        changed = _beir_changed(cfg, by[cfg.parent])
        if cfg.reranker:  # the bge row's settings (top 20, max_length 512); Qwen3-Reranker-8B also in bf16
            assert cfg.parent == beir.RERANK_FP32_512 and (cfg.rerank_top, cfg.max_length) == (20, 512)
            assert changed == ({"reranker", "precision"} if cfg.precision == "bf16" else {"reranker"}), cfg.name
        else:
            assert changed == {"embedder"} and by[cfg.parent].embedder == beir.BGE_SMALL, cfg.name
    # a new embedder's two rows are next to each other, so the runner releases it after the second
    for tag in ("qwen3-emb-0.6b", "qwen3-emb-8b"):
        assert [c.name for c in ladder].index(f"hybrid@{tag}") == [c.name for c in ladder].index(f"dense@{tag}") + 1
    harrier = [c.name for c in beir.models_2026_configs("harrier")]
    assert harrier == ["bm25", "dense@bge-small", "hybrid@bge-small", "dense@harrier-27b", "hybrid@harrier-27b"]
    assert {"models-2026", "models-2026-harrier"} <= set(beir.LADDERS)


def test_the_beir_comparisons():
    pairs = beir.models_2026_comparisons(beir.models_2026_configs("main"))
    assert ("hybrid@qwen3-emb-8b", "dense@qwen3-emb-8b") in pairs
    assert ("hybrid+qwen3-rerank-8b@bge-small/bf16-512/top20", "hybrid@bge-small") in pairs
    assert (beir.RERANK_FP32_512, "hybrid@bge-small") not in pairs  # its parent delta already


def test_the_runner_releases_each_model_after_its_last_row():
    engine = beir.Engine(beir.chunks_from_corpus({"d1": "a b", "d2": "c d"}), device="cpu")
    engine._embedders.update({"st:a": object(), "st:b": object()})
    engine._doc_mats.update({"st:a": 1, "st:b": 2})
    engine._unit_mats.update({"st:a": 1, "st:b": 2})
    key_x, key_y = engine.reranker_key("x/rerank", "fp32", 512), engine.reranker_key("y/rerank", "bf16", 512)
    engine._rerankers.update({key_x: object(), key_y: object()})
    engine.provenance.update({"st:b": {"model": "b"}, key_y: {"model": "y"}})
    later = [beir.BeirConfig(name="hybrid@a+x", mode="hybrid", embedder="st:a", reranker="x/rerank", max_length=512)]
    engine.release_unused(later)
    assert set(engine._embedders) == {"st:a"} and set(engine._doc_mats) == set(engine._unit_mats) == {"st:a"}
    assert set(engine._rerankers) == {key_x}
    assert engine.provenance["st:b"] == {"model": "b"} and engine.provenance[key_y] == {"model": "y"}  # what ran stays recorded
    engine.release_unused([])
    assert not engine._embedders and not engine._rerankers


def test_a_recomputed_parent_is_checked_against_the_committed_rows_and_their_trec_run(tmp_path, snowball):
    _write_beir_dataset(tmp_path)
    configs = [beir.BeirConfig(name="bm25", mode="bm25"),
               beir.BeirConfig(name="dense@hash", mode="dense", embedder="hash", parent="bm25"),
               beir.BeirConfig(name="hybrid@hash", mode="hybrid", embedder="hash", parent="dense@hash")]
    first = beir.run(tmp_path, configs, device="cpu", n_boot=50, download=False, root=tmp_path, trec_dir=tmp_path / "trec")
    bm25 = first["results"][0]
    assert bm25["ranking_sha256"] == bm25["trec"]["run_sha256"]  # the hash is the written run file's
    assert bm25["ranking_sha256"] == hashlib.sha256((tmp_path / "trec" / "bm25.run").read_bytes()).hexdigest()
    committed = tmp_path / "bench" / "results" / "committed.json"
    committed.parent.mkdir(parents=True)
    first["results"][1]["retrieval"]["ndcg@10"] = -1.0  # the committed dense row differs
    committed.write_text(json.dumps(first))
    blob = beir.run(tmp_path, configs, device="cpu", n_boot=50, download=False, root=tmp_path,
                    reproduce=("bench/results/committed.json", ("bm25", "dense@hash", "not-run")),
                    comparisons=[("hybrid@hash", "dense@hash")], header={"models_2026": {"set": "test"}})
    rep = blob["reproduces"]
    assert rep["rows"]["bm25"] == {"equal": True, "ranking_equal": True}
    assert rep["rows"]["dense@hash"]["equal"] is False and rep["rows"]["dense@hash"]["differs"]["ndcg@10"][0] == -1.0
    assert "not-run" not in rep["rows"] and rep["all"] is False
    cmp = blob["bootstrap"]["comparisons"]
    assert [(c["row"], c["other"]) for c in cmp] == [("hybrid@hash", "dense@hash")] and set(cmp[0]) >= {"ndcg@10", "recall@100"}
    assert blob["models_2026"] == {"set": "test"}
    assert beir.run(tmp_path, configs, device="cpu", n_boot=50, download=False, root=tmp_path, max_queries=1,
                    reproduce=("bench/results/committed.json", ("bm25",))).get("reproduces") is None  # a subset reproduces nothing


def test_a_checked_engine_records_each_model_check_and_stops_on_a_failure(monkeypatch):
    monkeypatch.setattr(beir, "get_cross_encoder", lambda name, **kw: _OverlapCrossEncoder(inverted=name == "inverted/rerank"))
    engine = beir.Engine(beir.chunks_from_corpus({"d1": "a b"}), device="cpu", check_models=True)
    engine.reranker("some/rerank", "fp32", 512)
    check = engine.provenance[engine.reranker_key("some/rerank", "fp32", 512)]["load_and_score"]
    assert check["passed"] is True and check["query"] == models_2026.CHECK_QUERY
    with pytest.raises(models_2026.ModelCheckFailed):
        engine.reranker("inverted/rerank", "fp32", 512)
    unchecked = beir.Engine(beir.chunks_from_corpus({"d1": "a b"}), device="cpu")
    unchecked.reranker("inverted/rerank", "fp32", 512)  # the committed ladders run as before: no check
    assert "load_and_score" not in unchecked.provenance[unchecked.reranker_key("inverted/rerank", "fp32", 512)]


def test_the_cli_runs_a_models_2026_ladder_checked_and_compared_with_the_committed_file(monkeypatch, tmp_path):
    seen = {}

    def fake_run(data_dir, configs, **kwargs):
        seen.update(kwargs, names=[c.name for c in configs])
        return {"dataset": "BEIR/fiqa", "split": "test", "n_docs": 1, "n_queries": 1, "published": None, "results": []}

    monkeypatch.setattr(beir, "run", fake_run)
    assert beir.main(["--ladder", "models-2026", "--dataset", "fiqa", "--data-dir", str(tmp_path), "--out", str(tmp_path / "x.json")]) == 0
    assert seen["check_models"] is True and seen["reproduce"][0] == "bench/results/beir_fiqa.json"
    assert seen["reproduce"][1] == beir.MODELS_2026_PARENTS and seen["header"]["models_2026"]["set"] == "main"
    assert ("hybrid@qwen3-emb-0.6b", "dense@qwen3-emb-0.6b") in seen["comparisons"]
    assert seen["names"] == [c.name for c in beir.models_2026_configs("main")]
    seen.clear()
    beir.main(["--ladder", "default", "--data-dir", str(tmp_path), "--out", str(tmp_path / "y.json")])
    assert "check_models" not in seen and "reproduce" not in seen  # the committed ladders run as before


# ---- the README blocks (app/eval/report.py): models, dtypes and prompts, every row with its interval, the reproduction


def _m41_row(config, parent=None, embedder=None, reranker=None, ndcg=0.5, delta=None, prompt=None, dtype="torch.float32", beir_row=False):
    rr_dtype = "torch.bfloat16" if reranker and "8B" in reranker else "torch.float32"
    row = {"config": config, "wall_ms": 1234.0,
           "ci": {m: {"mean": ndcg, "lo": round(ndcg - 0.05, 4), "hi": round(ndcg + 0.05, 4)} for m in ("ndcg@10", "recall@10", "recall@50", "recall@100")},
           "models": {"embedder": {"neural": bool(embedder), "model": embedder or "hash", "revision": "abcdef1234", "dtype": dtype,
                                   "query_prompt_name": prompt, "doc_embeddings": "computed in this run"},
                      "reranker": {"neural": bool(reranker), "loaded": bool(reranker), "model": reranker or "overlap", "revision": "fedcba9876",
                                   "dtype": rr_dtype, "default_prompt_name": "query" if reranker and "Qwen" in reranker else None}},
           "latency": {"embed": {"p50_ms": 12.5, "p95_ms": 14.0}, "rerank": {"p50_ms": 300.0, "p95_ms": 320.0},
                       "total": {"p50_ms": 40.0, "p95_ms": 55.0}}}
    if not embedder:
        row["models"].pop("embedder") if beir_row else None
    d = delta or {"delta": -0.0123, "lo": -0.0301, "hi": -0.0011}
    if beir_row:
        row.update(parent=parent, settings={"reranker": reranker, "precision": "bf16" if "8B" in (reranker or "") else "fp32", "max_length": 512, "rerank_top": 20},
                   build_ms={"total_ms": 2500.0})
        if parent:
            row["delta_vs_parent"] = {"reference": parent, "ndcg@10": d, "recall@100": {"delta": 0.0, "lo": 0.0, "hi": 0.0}}
    else:
        row["settings"] = {"parent": parent}
        if parent:
            row["delta_vs_previous"] = {"reference": parent, "ndcg@10": d, "recall@50": {"delta": 0.0, "lo": 0.0, "hi": 0.0}}
            row["vs_parent"] = {"parent": parent, "n_questions": 136, "n_top10_differs": 7, "n_list_differs": 9}
    return row


DEVICE = {"device": "cuda", "device_name": "NVIDIA A40", "torch": "2.5.1+cu121", "sentence_transformers": "6.1.0"}


def test_the_fixture_block_names_each_model_its_dtype_and_prompt_and_keeps_the_losers():
    from app.eval.report import models_2026_fixture_table

    blob = {"git_sha": "1234567", "device": DEVICE, "results": [
        _m41_row("bm25_only"),
        _m41_row("hybrid@bge-small", "dense_only@bge-small", embedder="BAAI/bge-small-en-v1.5"),
        _m41_row("hybrid@qwen3-emb-8b", "hybrid@bge-small", embedder="Qwen/Qwen3-Embedding-8B", prompt="query", dtype="torch.bfloat16"),
        _m41_row("hybrid+qwen3-rerank-8b@bge-small", "hybrid+bge-rerank@bge-small", embedder="BAAI/bge-small-en-v1.5", reranker="Qwen/Qwen3-Reranker-8B")],
        "bootstrap": {"comparisons": [{"row": "hybrid@qwen3-emb-8b", "other": "dense_only@qwen3-emb-8b", "ndcg@10": {"delta": 0.02, "lo": -0.01, "hi": 0.05}}]},
        "reproduces": [{"file": "bench/results/neural.json", "git_sha": "32972cf", "device": "Quadro RTX 8000",
                        "rows": {"bm25_only": {"equal": True}, "hybrid+bge-rerank@bge-small": {"equal": False, "differs": {"ndcg@10": [0.8119, 0.8125]}}}}]}
    text = models_2026_fixture_table(blob, None)
    assert "cuda (NVIDIA A40)" in text and "results sha 1234567" in text
    assert "| hybrid@qwen3-emb-8b | hybrid@bge-small | Qwen/Qwen3-Embedding-8B@abcdef1 (bfloat16, query prompt `query`) |" in text
    assert "Qwen/Qwen3-Reranker-8B@fedcba9 (bfloat16, prompt `query`)" in text and "| bm25_only | — | none (BM25 only) |" in text
    assert "-0.012 [-0.030, -0.001]" in text and "| 7 / 136 |" in text  # a loser keeps its row and interval
    assert "`hybrid@qwen3-emb-8b` minus `dense_only@qwen3-emb-8b`: nDCG@10 +0.020 [-0.010, +0.050] (the interval covers zero)." in text
    assert "`bm25_only` reproduces bench/results/neural.json (results sha 32972cf, Quadro RTX 8000) in every field" in text
    assert "`hybrid+bge-rerank@bge-small` does NOT reproduce bench/results/neural.json (results sha 32972cf, Quadro RTX 8000): ndcg@10 0.8119 there, 0.8125 here" in text
    assert "models_2026_harrier.json is missing; run `make bench-models-2026-harrier`" in text
    assert "models_2026.json is missing" in models_2026_fixture_table(None)


def test_the_cpu_block_names_the_cpu_and_its_threads():
    from app.eval.report import models_2026_cpu_table

    blob = {"git_sha": "1234567", "device": {**DEVICE, "device": "cpu", "cpu_model": "Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz", "threads": 8},
            "results": [_m41_row("hybrid+mxbai-rerank-base-v2@bge-small/fp32-512/top20", "hybrid+bge-rerank@bge-small/fp32-512/top20",
                                 embedder="BAAI/bge-small-en-v1.5", reranker="mixedbread-ai/mxbai-rerank-base-v2")],
            "reproduces": []}
    text = models_2026_cpu_table(blob)
    assert "Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz, 8 torch threads" in text
    assert "| 12.5 / 14.0 | 300.0 / 320.0 |" in text and "mixedbread-ai/mxbai-rerank-base-v2@fedcba9 (float32)" in text
    assert "records no reproduction check" in text


def test_the_beir_block_says_how_each_parent_reproduced_and_where_document_embeddings_came_from():
    from app.eval.report import models_2026_beir_table

    blob = {"git_sha": "1234567", "device": DEVICE, "n_docs": 57638, "n_queries": 648, "results": [
        _m41_row("bm25", beir_row=True),
        _m41_row("dense@qwen3-emb-0.6b", "dense@bge-small", embedder="Qwen/Qwen3-Embedding-0.6B", prompt="query", beir_row=True)],
        "reproduces": {"file": "bench/results/beir_fiqa.json", "git_sha": "32972cf", "device": "Quadro RTX 8000",
                       "rows": {"bm25": {"equal": True, "ranking_equal": True}, "dense@bge-small": {"equal": True, "ranking_equal": False}}}}
    text = models_2026_beir_table(blob, None, "bench/results/beir_fiqa_models_2026.json", "bench/results/beir_fiqa_models_2026_harrier.json",
                                  "make bench-beir-fiqa-models-2026", "make bench-beir-fiqa-models-2026-harrier", "FiQA")
    assert "BEIR FiQA (`make bench-beir-fiqa-models-2026`): 57638 documents, 648 queries" in text
    assert "`bm25` reproduces bench/results/beir_fiqa.json" in text and "and `bm25` ranks every query exactly as the committed TREC run does" in text
    assert ("`dense@bge-small` equals bench/results/beir_fiqa.json (results sha 32972cf, Quadro RTX 8000) in every field of the retrieval block "
            "but does NOT rank every query as the committed TREC run does, so it does not reproduce it exactly") in text
    assert "Document embeddings computed in this run: Qwen/Qwen3-Embedding-0.6B." in text
    assert "-0.0123 [-0.0301, -0.0011]" in text and "beir_fiqa_models_2026_harrier.json is missing" in text


def test_evidence_rows_only_where_the_interval_excludes_zero(tmp_path):
    """docs/RESUME_EVIDENCE.md gets an M41 row only where a paired interval excludes zero, losers included."""
    from app.eval.evidence import FILES, _models_2026_claims

    up, tie, down = {"delta": 0.05, "lo": 0.02, "hi": 0.08}, {"delta": 0.01, "lo": -0.01, "hi": 0.03}, {"delta": -0.04, "lo": -0.06, "hi": -0.02}
    blob = {"git_sha": "1234567", "n_queries": 300, "device": DEVICE, "reproduces": {"file": "bench/results/beir_scifact.json", "rows": {"bm25": {"equal": True}}},
            "results": [{"config": "dense@qwen3-emb-8b", "delta_vs_parent": {"ndcg@10": up}},
                        {"config": "hybrid@qwen3-emb-8b", "delta_vs_parent": {"ndcg@10": tie}},
                        {"config": "hybrid+qwen3-rerank-8b@bge-small/bf16-512/top20", "delta_vs_parent": {"ndcg@10": tie}},
                        {"config": "hybrid+mxbai-rerank-large-v2@bge-small/fp32-512/top20", "delta_vs_parent": {"ndcg@10": up}}],
            "bootstrap": {"comparisons": [{"row": "hybrid@qwen3-emb-8b", "other": "dense@qwen3-emb-8b", "ndcg@10": down}]}}
    path = tmp_path / FILES["beir_scifact_models_2026"]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(blob))
    claims = {c["claim"]: c for c in _models_2026_claims(tmp_path)}
    assert set(claims) == {"BEIR SciFact, Qwen3-Embedding-8B vs bge-small-en-v1.5, dense retrieval, paired delta in nDCG@10",
                           "BEIR SciFact, hybrid RRF minus dense-only, both with Qwen3-Embedding-8B, paired delta in nDCG@10",
                           "BEIR SciFact, mxbai-rerank-large-v2 (fp32) vs bge-reranker-v2-m3 (fp32), both over hybrid@bge-small, paired delta in nDCG@10"}
    loser = claims["BEIR SciFact, hybrid RRF minus dense-only, both with Qwen3-Embedding-8B, paired delta in nDCG@10"]
    assert (loser["value"], loser["ci"]) == ("-0.0400", "[-0.0600, -0.0200]")  # a loser with its interval below zero stays
    assert "(the interval covers zero)" in claims["BEIR SciFact, Qwen3-Embedding-8B vs bge-small-en-v1.5, dense retrieval, paired delta in nDCG@10"]["not_shown"]
