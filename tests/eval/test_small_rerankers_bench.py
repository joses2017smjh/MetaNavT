"""M43: the rows of the small rerankers (make bench-small-rerankers and the BEIR `small-rerankers` ladder), their
per-query sidecars, the rescore from those, and the bar read off the three files.

No torch: the cross-encoder is a stand-in scoring by token overlap, the embedder the hash embedder. What is tested is
the wiring: which rows exist and what each is paired with, that the fixture run ranks exactly as the harness does, that
a student is found through its manifest and never named by a path, and that every interval is reproduced from the
sidecar alone.
"""

import copy
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from app.eval import beir
from app.eval import small_rerankers as sr
from app.eval.gold import load_gold
from app.eval.harness import BenchConfig, run_config
from app.eval.rerank_arms import split_questions
from app.retrieval.bm25 import tokenize

ROOT = Path(__file__).resolve().parents[2]
FILES = ROOT / "bench" / "corpus" / "files"


class _Overlap:
    """A stand-in CrossEncoder: the logit is the share of query tokens in the passage, minus a bias."""

    default_prompt_name, prompts, max_seq_length = None, {}, 512

    def __init__(self, bias=0.5):
        self.bias = bias

    def predict(self, pairs, batch_size=32, show_progress_bar=False, activation_fn=None):
        out = []
        for q, d in pairs:
            qt, dt = set(tokenize(q)), set(tokenize(d))
            x = len(qt & dt) / max(len(qt), 1) - self.bias
            out.append(activation_fn(x) if activation_fn else 1 / (1 + math.exp(-x)))
        return out

    def parameters(self):
        yield type("P", (), {"dtype": "torch.float32"})()


TAGS = ["bge-rerank", sr.TEACHER_TAG, "minilm-l6", "student-minilm-l6"]


@pytest.fixture(scope="module")
def fixture_run():
    """The fixture run with the hash embedder and one stand-in for every reranker (so all reranked rows rank alike)."""
    return sr.run_fixture(ROOT, tags=TAGS, embedder="hash", load=lambda tag: (_Overlap(), {"model": tag, "stand_in": True}))


def test_the_fixture_rows_and_what_each_is_paired_with(fixture_run):
    blob, sidecar = fixture_run
    names = [sr.PARENT, *(sr.row_name(t) for t in TAGS)]
    assert blob["small_rerankers"]["rows"] == names and blob["small_rerankers"]["served_defaults_changed"] is False
    for scope, n in (("held_out", 68), ("tune", 68), ("all_questions", 136)):
        rows = {r["config"]: r for r in blob[scope]["results"]}
        assert list(rows) == names and all(r["n"] == n for r in rows.values())
        assert blob[scope]["bootstrap"]["n_queries"] == n and blob[scope]["bootstrap"]["n_boot"] == 2000
        assert "deltas" not in rows[sr.PARENT]
        refs = {name: [d["reference"] for d in row.get("deltas", [])] for name, row in rows.items()}
        teacher = sr.row_name(sr.TEACHER_TAG)
        assert refs[sr.BGE_ROW] == [sr.PARENT, teacher]
        assert refs[teacher] == [sr.PARENT, sr.BGE_ROW]
        assert refs[sr.row_name("minilm-l6")] == [sr.PARENT, sr.BGE_ROW, teacher]
        assert refs[sr.row_name("student-minilm-l6")] == [sr.PARENT, sr.BGE_ROW, teacher, sr.row_name("minilm-l6")]  # and its own init
    gold = load_gold(ROOT / "bench" / "gold" / "questions.jsonl")
    tune, held = split_questions(gold)
    assert blob["split"]["held_out_ids"] == [gold[i].id for i in held] and blob["split"]["tune_ids"] == [gold[i].id for i in tune]
    assert "no choice was made on the tune half" in blob["split"]["use"]
    info = blob["rerankers"]["minilm-l6"]
    assert info["load_and_score"]["passed"] and info["depth"] == 20 and info["max_length"] == 512 and info["precision"] == "fp32"
    assert info["rerank_latency"]["n"] == 136 and "not the server" in info["rerank_latency"]["note"]
    assert "no committed row has the teacher at this shape" in blob["not_reproduced_anywhere"]
    assert sidecar["split"] == {"tune": tune, "held_out": held} and len(sidecar["ids"]) == 136
    assert str(ROOT) not in json.dumps(blob)  # nothing absolute in a results file


def test_the_fixture_run_ranks_as_the_harness_does(fixture_run, monkeypatch):
    blob, _ = fixture_run
    rows = {r["config"]: r for r in blob["all_questions"]["results"]}
    # the unreranked parent with the hash embedder is main.json's `hybrid` row, field for field
    main = {r["config"]: r for r in json.loads((ROOT / "bench" / "results" / "main.json").read_text())["results"]}
    assert rows[sr.PARENT]["retrieval"] == main["hybrid"]["retrieval"]
    # a reranked row is the harness's row with the same model at rerank_depth 20 and max_length 512
    monkeypatch.setattr("app.retrieval.rerank.get_cross_encoder", lambda name, **kw: _Overlap())
    gold = load_gold(ROOT / "bench" / "gold" / "questions.jsonl")
    cfg = BenchConfig(name="x", mode="hybrid", embedder="hash", reranker="stand-in", enable_rerank=True, enable_router=False,
                      staleness_tier1=False, e2e=False, log_triples=False, rerank_depth=20, rerank_max_length=512)
    harness_row = run_config(cfg, gold, FILES, n_files=blob["n_files"], mtimes="manifest")
    assert rows[sr.BGE_ROW]["retrieval"] == harness_row["retrieval"]
    assert rows[sr.BGE_ROW]["retrieval"] != rows[sr.PARENT]["retrieval"]  # the stand-in does reorder the top 20
    assert [b["file"] for b in blob["reproduces"]] == ["bench/results/neural.json", "bench/results/rerank_depth.json"]
    assert blob["reproduces_all"] is False  # recorded whatever the answer: the hash embedder is not bge-small


def test_every_fixture_interval_is_recomputed_from_the_sidecar(fixture_run):
    blob, sidecar = fixture_run
    assert sr.rescore_fixture(blob, sidecar) == []
    moved = copy.deepcopy(blob)
    moved["held_out"]["results"][1]["deltas"][0]["ndcg@10"]["lo"] += 0.01
    assert any("held_out" in line and "deltas" in line for line in sr.rescore_fixture(moved, sidecar))
    moved = copy.deepcopy(blob)
    moved["tune"]["results"][2]["retrieval"]["ndcg@10"] = 0.0
    assert any("tune" in line and "retrieval.ndcg@10" in line for line in sr.rescore_fixture(moved, sidecar))
    other = {**sidecar, "timestamp": "another run"}
    assert sr.rescore_fixture(blob, other) and "not the same run" in sr.rescore_fixture(blob, other)[0]


def test_a_model_that_cannot_rank_stops_the_fixture_run():
    class Inverted(_Overlap):
        def predict(self, pairs, **kw):
            return [-x for x in super().predict(pairs, **kw)]

    from app.eval.models_2026 import ModelCheckFailed

    with pytest.raises(ModelCheckFailed):
        sr.run_fixture(ROOT, tags=["minilm-l6"], embedder="hash", load=lambda tag: (Inverted(), {}))


# ---- the BEIR ladder


def test_the_beir_ladder_pairs_every_reranker_with_the_unreranked_hybrid():
    with_students = beir.small_reranker_configs(with_students=True)
    names = [c.name for c in with_students]
    assert names == [sr.PARENT, *(sr.row_name(t) for t in sr.reranker_tags())]
    assert with_students[0].parent is None and with_students[0].reranker is None
    for cfg in with_students[1:]:
        assert (cfg.parent, cfg.rerank_top, cfg.max_length, cfg.precision, cfg.embedder) == (sr.PARENT, 20, 512, "fp32", beir.BGE_SMALL)
    by = {c.name: c for c in with_students}
    assert by[sr.BGE_ROW].reranker == "BAAI/bge-reranker-v2-m3" and by[sr.BGE_ROW].name == beir.RERANK_FP32_512
    assert by[sr.row_name("student-minilm-l12")].reranker == "student-minilm-l12"  # a tag: the run resolves it to a verified checkpoint
    assert by[sr.row_name("gte-modernbert-base")].reranker == "Alibaba-NLP/gte-reranker-modernbert-base"
    without = [c.name for c in beir.small_reranker_configs(with_students=False)]
    assert without == [n for n in names if "student" not in n]  # the zero-shot rows can be measured before any student exists
    assert beir.SMALL_RERANKERS_LADDER in beir.LADDERS
    pairs = beir.small_reranker_comparisons(with_students)
    teacher = sr.row_name(sr.TEACHER_TAG)
    assert (sr.row_name("minilm-l6"), sr.BGE_ROW) in pairs and (sr.row_name("minilm-l6"), teacher) in pairs
    assert (teacher, sr.BGE_ROW) in pairs and (sr.BGE_ROW, teacher) in pairs
    assert (sr.row_name("student-minilm-l6"), sr.row_name("minilm-l6")) in pairs
    assert not any(sr.PARENT in pair for pair in pairs)  # that delta is each row's parent delta
    assert sr.comparisons(names)[: 2] == [(sr.BGE_ROW, sr.PARENT), (sr.BGE_ROW, teacher)]


def test_the_rows_a_beir_run_compares_itself_with_are_committed():
    for dataset, pairs in beir.SMALL_RERANKERS_REPRODUCE.items():
        for rel, names in pairs:
            rows = {r["config"] for r in json.loads((ROOT / rel).read_text())["results"]}
            assert set(names) <= rows, (dataset, rel)


def _write_dataset(root: Path) -> None:
    d = root / "scifact"
    (d / "qrels").mkdir(parents=True)
    docs = {f"d{i}": (f"Topic {i}", f"protein number {i} folds in cells and tissue sample {i}") for i in range(30)}
    with (d / "corpus.jsonl").open("w") as fh:
        for did, (title, text) in docs.items():
            fh.write(json.dumps({"_id": did, "title": title, "text": text}) + "\n")
    with (d / "queries.jsonl").open("w") as fh:
        for i in range(12):
            fh.write(json.dumps({"_id": f"q{i}", "text": f"protein number {i} in tissue sample {i}"}) + "\n")
    (d / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\n" + "".join(f"q{i}\td{i}\t1\n" for i in range(12)))


def test_a_beir_run_resolves_students_records_them_and_rescores_from_its_sidecar(tmp_path, monkeypatch):
    _write_dataset(tmp_path)
    loaded = []

    def load(source, **kw):
        loaded.append(source)
        return _Overlap()

    monkeypatch.setattr(beir, "get_cross_encoder", load)
    ckpt = tmp_path / "ckpt" / "student-minilm-l6"
    ckpt.mkdir(parents=True)
    (ckpt / sr.WEIGHTS_FILE).write_bytes(b"w")
    sha = hashlib.sha256(b"w").hexdigest()

    def resolve(name):
        if name == "student-minilm-l6":
            return str(ckpt), {"model": name, "weights_sha256": sha, "student_of": "cross-encoder/ms-marco-MiniLM-L6-v2"}, lambda model: {"reference_check": {"lists": 3}}
        return name, {}, None

    configs = [replace(c, embedder="hash") for c in beir.small_reranker_configs(with_students=True)
               if c.name in (sr.PARENT, sr.BGE_ROW, sr.row_name("minilm-l6"), sr.row_name("student-minilm-l6"))]
    blob = beir.run(tmp_path, configs, device="cpu", n_boot=50, download=False, check_models=True, keep_per_query=True,
                    comparisons=beir.small_reranker_comparisons(configs), resolve_reranker=resolve,
                    reproduce=(("bench/results/beir_scifact.json", ("hybrid@bge-small",)), ("bench/results/missing.json", (sr.BGE_ROW,))),
                    header={"small_rerankers": sr.header([c.name for c in configs])})
    per_query = blob.pop("_per_query")
    assert loaded == ["BAAI/bge-reranker-v2-m3", "cross-encoder/ms-marco-MiniLM-L6-v2", str(ckpt)]
    rows = {r["config"]: r for r in blob["results"]}
    student = rows[sr.row_name("student-minilm-l6")]["models"]["reranker"]
    assert student["model"] == "student-minilm-l6" and student["weights_sha256"] == sha and student["reference_check"] == {"lists": 3}
    assert student["load_and_score"]["passed"] and str(tmp_path) not in json.dumps(blob["results"])  # the sha256, never the path
    assert all(rows[n]["delta_vs_parent"]["reference"] == sr.PARENT for n in rows if n != sr.PARENT)
    assert isinstance(blob["reproduces"], list) and [b["file"] for b in blob["reproduces"]] == ["bench/results/beir_scifact.json", "bench/results/missing.json"]
    sidecar = beir.per_query_sidecar(blob, per_query)
    assert sidecar["schema"] == beir.PER_QUERY_SCHEMA and sidecar["ids"] == [f"q{i}" for i in range(12)]
    assert set(sidecar["scores"]) == set(rows) and len(sidecar["scores"][sr.PARENT]["ndcg@10"]) == 12
    assert sr.rescore_beir(blob, sidecar) == []
    moved = copy.deepcopy(blob)
    moved["bootstrap"]["comparisons"][0]["ndcg@10"]["delta"] += 0.5
    moved["results"][1]["delta_vs_parent"]["ndcg@10"]["hi"] += 0.5
    problems = sr.rescore_beir(moved, sidecar)
    assert any("bootstrap.comparisons" in p for p in problems) and any("delta_vs_parent" in p for p in problems)


def test_a_committed_rows_ranking_is_compared_when_it_recorded_one():
    assert beir._same_ranking({"ranking_sha256": "a"}, {"trec": {"run_sha256": "a"}}) == {"ranking_equal": True}
    assert beir._same_ranking({"ranking_sha256": "a"}, {"ranking_sha256": "b"}) == {"ranking_equal": False}  # rows since M41
    assert beir._same_ranking({"ranking_sha256": "a"}, {}) == {}


# ---- finding a model, and the bar


def test_the_resolver_pins_candidates_and_checks_students(tmp_path, monkeypatch):
    monkeypatch.setattr("app.eval.provenance.hf_revision", lambda name: sr.CANDIDATES["minilm-l6"]["revision"])
    resolve = sr.resolver(tmp_path, ROOT)
    source, record, after = resolve("cross-encoder/ms-marco-MiniLM-L6-v2")  # by hub id, as a BEIR config names it
    assert source == "cross-encoder/ms-marco-MiniLM-L6-v2" and record == {"pinned_revision": sr.CANDIDATES["minilm-l6"]["revision"]} and after is None
    with pytest.raises(RuntimeError, match="pinned"):
        resolve("minilm-l12")  # its cached revision is not the pinned one
    assert resolve("bge-rerank")[:2] == ("BAAI/bge-reranker-v2-m3", {})  # a reference is pinned by the committed row it reproduces
    with pytest.raises(sr.CheckpointError):
        resolve("student-minilm-l6")  # no manifest entry, no checkpoint: never a fallback to the init
    with pytest.raises(KeyError):
        resolve("some/other-model")
    assert sr.tag_of("mixedbread-ai/mxbai-rerank-large-v2") == sr.TEACHER_TAG and sr.tag_of("student-minilm-l12") == "student-minilm-l12"


def _delta(lo):
    return {"delta": lo + 0.02, "lo": lo, "hi": lo + 0.04}


def _fixture_blob(lows):
    rows = [{"config": sr.PARENT, "n": 68, "ci": {"ndcg@10": {"mean": 0.7, "lo": 0.6, "hi": 0.8}}}]
    for tag, lo in lows.items():
        rows.append({"config": sr.row_name(tag), "n": 68, "ci": {"ndcg@10": {"mean": 0.8, "lo": 0.7, "hi": 0.9}},
                     "deltas": [{"reference": sr.PARENT, "ndcg@10": _delta(lo)}, {"reference": sr.BGE_ROW, "ndcg@10": _delta(-0.1)},
                                {"reference": sr.row_name("minilm-l6"), "ndcg@10": _delta(0.05)}]})
    return {"git_sha": "abc1234", "device": {"device_name": "NVIDIA A40"}, "held_out": {"bootstrap": {"n_queries": 68}, "results": rows}}


def _beir_blob(lows, n):
    rows = [{"config": sr.PARENT, "ci": {"ndcg@10": {"mean": 0.7, "lo": 0.6, "hi": 0.8}}, "delta_vs_parent": None}]
    for tag, lo in lows.items():
        rows.append({"config": sr.row_name(tag), "ci": {"ndcg@10": {"mean": 0.8, "lo": 0.7, "hi": 0.9}},
                     "delta_vs_parent": {"reference": sr.PARENT, "ndcg@10": _delta(lo)}})
    comparisons = [{"row": sr.row_name(tag), "other": sr.BGE_ROW, "ndcg@10": _delta(0.01)} for tag in lows if tag != "bge-rerank"]
    return {"git_sha": "abc1234", "n_queries": n, "device": {"device_name": "NVIDIA A40"}, "bootstrap": {"comparisons": comparisons}, "results": rows}


def test_a_row_passes_only_with_all_three_intervals_above_zero(tmp_path):
    results = tmp_path / "bench" / "results"
    results.mkdir(parents=True)
    (results / "small_rerankers.json").write_text(json.dumps(_fixture_blob({"bge-rerank": 0.05, "minilm-l6": 0.01, "student-minilm-l6": 0.02})))
    (results / "beir_scifact_small_rerankers.json").write_text(json.dumps(_beir_blob({"bge-rerank": -0.007, "minilm-l6": 0.01, "student-minilm-l6": 0.02}, 300)))
    table = sr.bar_table(tmp_path)
    by = {r["tag"]: r for r in table["rows"]}
    assert by["bge-rerank"]["passes"] is False  # a tie on SciFact is not a pass, whatever FiQA would say
    assert by["minilm-l6"]["passes"] is None and by["student-minilm-l6"]["passes"] is None  # FiQA not measured: open, never passed
    (results / "beir_fiqa_small_rerankers.json").write_text(json.dumps(_beir_blob({"bge-rerank": 0.04, "minilm-l6": 0.0, "student-minilm-l6": 0.001}, 648)))
    table = sr.bar_table(tmp_path)
    by = {r["tag"]: r for r in table["rows"]}
    assert by["minilm-l6"]["passes"] is False  # a lower end of exactly zero is not above zero
    assert by["student-minilm-l6"]["passes"] is True
    cell = by["student-minilm-l6"]["datasets"]
    assert cell["fixture"]["vs_init"]["lo"] == 0.05 and cell["fixture"]["vs_bge"]["lo"] == -0.1 and cell["scifact"]["vs_bge"]["lo"] == 0.01
    assert cell["fiqa"]["vs_parent"]["lo"] == 0.001 and cell["fiqa"]["above_zero"] is True
    assert table["datasets"]["fixture"]["n"] == 68 and table["datasets"]["fiqa"]["n"] == 648
    assert [r["tag"] for r in table["rows"]] == ["bge-rerank", "minilm-l6", "student-minilm-l6"]  # table order; rows no file has are left out
    assert sr.bar_table(tmp_path / "empty") == {"datasets": {}, "rows": []}


def test_the_cli_modes(tmp_path, capsys):
    assert sr.main(["--bar"]) == 0
    assert sr.main([]) == 2
