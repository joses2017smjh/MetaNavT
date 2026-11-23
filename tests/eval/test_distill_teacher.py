"""M43 (app/eval/distill_teacher.py): the candidates a served reranker sees, scored by the teacher, shard by shard.

No torch: the teacher is a stand-in whose logit is the token overlap, the first stage uses the hash embedder.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from app.eval import distill_teacher as dt
from app.eval.distill_queries import write_jsonl_gz
from app.retrieval.bm25 import tokenize
from app.retrieval.rerank import cross_encoder_logits

ROOT = Path(__file__).resolve().parents[2]


class _Teacher:
    """A stand-in CrossEncoder: the raw logit is the number of query tokens in the text, minus 2."""

    def __init__(self):
        self.calls = 0

    def predict(self, pairs, batch_size=32, show_progress_bar=False, activation_fn=None):
        self.calls += 1
        raw = [float(len(set(tokenize(q)) & set(tokenize(t))) - 2) for q, t in pairs]
        return [activation_fn(x) for x in raw] if activation_fn else [0.5 for _ in raw]


class _Stage:
    texts = {f"d{i}": f"alpha beta doc{i}" for i in range(30)}

    def top(self, query, n):
        return [f"d{i}" for i in range(n)]


def test_raw_logits_come_back_without_the_sigmoid():
    assert cross_encoder_logits(_Teacher(), [("alpha beta gamma", "alpha beta"), ("x", "y")]) == [0.0, -2.0]
    assert cross_encoder_logits(_Teacher(), []) == []

    class NoOverride:
        def predict(self, pairs, batch_size=32, show_progress_bar=False):
            return [0.5] * len(pairs)

    with pytest.raises(TypeError, match="raw logits cannot be read"):  # never a probability under the name of a logit
        cross_encoder_logits(NoOverride(), [("q", "t")])


def test_the_source_document_is_appended_when_the_first_stage_missed_it():
    stage = _Stage()
    ids, rank = dt.candidate_list(stage, "alpha", "d3", top=20)
    assert ids == [f"d{i}" for i in range(20)] and rank == 4
    ids, rank = dt.candidate_list(stage, "alpha", "d25", top=20)
    assert ids == [f"d{i}" for i in range(20)] + ["d25"] and rank is None
    with pytest.raises(KeyError):
        dt.candidate_list(stage, "alpha", "not-a-document", top=20)


def test_shards_partition_the_queries():
    queries = [{"id": f"q{i:03d}"} for i in range(11)]
    parts = [dt.shard_of(list(reversed(queries)), (i, 4)) for i in range(4)]
    assert sorted(q["id"] for part in parts for q in part) == [q["id"] for q in queries]
    assert [q["id"] for q in parts[1]] == ["q001", "q005", "q009"]
    with pytest.raises(ValueError):
        dt.shard_of(queries, (4, 4))


def test_a_shard_scores_every_list_once_and_resumes(tmp_path):
    queries = [{"id": f"fixture:d{i}:q1", "corpus": "fixture", "doc_id": f"d{i}", "query": f"alpha doc{i}"} for i in range(25)]
    out = tmp_path / "teacher.shard-0-of-1.jsonl"
    teacher = _Teacher()
    counts = dt.score_shard(queries[:7], {"fixture": _Stage()}, teacher, out, block=3)
    assert counts == {"already_done": 0, "scored": 7, "pairs": 7 * 20} and teacher.calls == 3  # blocks of 3 lists
    counts = dt.score_shard(queries, {"fixture": _Stage()}, teacher, out, block=10)
    assert counts["already_done"] == 7 and counts["scored"] == 18
    rows = dt.read_jsonl(out)
    assert [r["id"] for r in rows] == [q["id"] for q in queries]
    first = rows[0]
    assert first["candidates"][:3] == ["d0", "d1", "d2"] and first["source_rank"] == 1 and len(first["scores"]) == 20
    assert first["scores"][0] == 0.0 and first["scores"][1] == -1.0  # "alpha doc0" against its own text, then another's
    appended = rows[24]
    assert appended["source_rank"] is None and appended["candidates"][-1] == "d24" and len(appended["scores"]) == 21
    summary = dt.summarise(rows)["fixture"]
    assert summary["lists"] == 25 and summary["pairs"] == 20 * 20 + 5 * 21
    assert summary["source_in_fused_top"] == 20 and summary["source_first_in_fused"] == 1
    assert summary["source_first_with_teacher"] == 25  # the stand-in scores the source document highest in every list


def test_a_score_that_is_not_a_number_stops_the_shard():
    class Broken(_Teacher):
        def predict(self, pairs, **kw):
            return [float("nan")] * len(pairs)

    with pytest.raises(RuntimeError, match="not finite"):
        dt.score_block(Broken(), [("q", ["a", "b"])])


def test_the_fixture_first_stage_is_the_benchmarks_hybrid_list(monkeypatch):
    monkeypatch.setattr(dt, "BGE_SMALL", "hash")  # no torch here; the committed run uses bge-small
    stage = dt.FixtureFirstStage(ROOT)
    assert len(stage.texts) == 71 and stage.provenance["unit"] == "chunk"
    top = stage.top("what encoder does the trellis model use", 20)
    hits = stage.index.retrieve("what encoder does the trellis model use", k=dt.FIXTURE_RETRIEVE_K).hits
    assert top == [h.chunk.chunk_id for h in hits[:20]] and all(h.rerank is None for h in hits)
    assert dt.corpus_texts("fixture", ROOT, ROOT) == stage.texts


def test_the_beir_first_stage_needs_the_corpus_alone(tmp_path, monkeypatch):
    from app.eval import beir

    root = tmp_path / "scifact"
    root.mkdir()
    with (root / "corpus.jsonl").open("w") as fh:
        for i in range(40):
            fh.write(json.dumps({"_id": str(i), "title": f"Title {i}", "text": f"protein folding study number {i} in cells"}) + "\n")
    hybrid = next(c for c in beir.DEFAULT_CONFIGS if c.name == "hybrid@bge-small")
    monkeypatch.setattr(beir, "DEFAULT_CONFIGS", [replace(hybrid, embedder="hash")])
    stage = dt.BeirFirstStage(tmp_path, "scifact", device="cpu")  # no queries.jsonl and no qrels exist here
    top = stage.top("protein folding study number 7", 20)
    assert len(top) == 20 and "7" in top[:3] and set(top) <= set(stage.texts)
    assert stage.provenance["analyzer"] == "beir" and stage.provenance["n_units"] == 40
    assert dt.corpus_texts("scifact", ROOT, tmp_path) == stage.texts


def _shard(work, i, n, rows, *, head="a" * 40, dirty=False, queries_sha="s"):
    path, meta = dt._shard_paths(work, (i, n))
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    run = {"git_head": head, "working_tree_dirty": dirty, "timestamp": f"2026-11-22T0{i}:00:00+00:00", "device": "NVIDIA A40",
           "already_done": 0, "scored": len(rows), "pairs": sum(len(r["scores"]) for r in rows), "build_seconds": 1.0, "score_seconds": 2.0}
    meta.write_text(json.dumps({"git_sha": head[:7], "git_head": head, "working_tree_dirty": dirty, "source_sha256": {}, "timestamp": run["timestamp"],
                                "command": f"python -m app.eval.distill_teacher score --shard {i}/{n}", "device": {"device_name": "NVIDIA A40"},
                                "shard": [i, n], "queries_sha256": queries_sha, "n_queries": len(rows), "runs": [run],
                                "teacher": {"model": "teacher"}, "first_stage": {"config": "hybrid@bge-small", "corpora": {"fixture": {"n_units": 71}}}}))


def test_merge_requires_every_query_once_at_one_clean_commit(tmp_path):
    import hashlib

    queries = [{"id": f"fixture:d{i}:q1", "corpus": "fixture", "doc_id": f"d{i}", "query": "q", "form": "questions", "split": "train"} for i in range(6)]
    qfile = tmp_path / "queries.jsonl.gz"
    write_jsonl_gz(qfile, queries)
    sha = hashlib.sha256(qfile.read_bytes()).hexdigest()
    row = lambda q: {"id": q["id"], "corpus": "fixture", "candidates": [q["doc_id"], "x"], "source_rank": 1, "scores": [2.0, -1.0]}  # noqa: E731
    work, out = tmp_path / "work", tmp_path / "out"
    work.mkdir()
    args = ["merge", "--work", str(work), "--queries", str(qfile), "--out-dir", str(out)]
    with pytest.raises(SystemExit, match="no shard"):
        dt.main(args)
    _shard(work, 0, 2, [row(q) for q in queries[0::2]], queries_sha=sha)
    with pytest.raises(SystemExit, match="are not 0..1 of 2"):
        dt.main(args)
    _shard(work, 1, 2, [row(q) for q in queries[1::2]][:-1], queries_sha=sha)
    with pytest.raises(SystemExit, match="do not cover every query exactly once"):
        dt.main(args)
    _shard(work, 1, 2, [row(q) for q in queries[1::2]], queries_sha=sha, head="b" * 40)
    with pytest.raises(SystemExit, match="not all at one commit"):
        dt.main(args)
    _shard(work, 1, 2, [row(q) for q in queries[1::2]], queries_sha=sha, dirty=True)
    with pytest.raises(SystemExit, match="uncommitted changes"):
        dt.main(args)
    _shard(work, 1, 2, [row(q) for q in queries[1::2]], queries_sha="another file")
    with pytest.raises(SystemExit, match="not all scored for this queries file"):
        dt.main(args)
    assert not out.exists()
    _shard(work, 1, 2, [row(q) for q in queries[1::2]], queries_sha=sha)
    assert dt.main(args) == 0
    merged = dt.read_jsonl(out / dt.SCORES_FILE)
    assert [r["id"] for r in merged] == [q["id"] for q in queries]
    meta = json.loads((out / dt.SCORES_META).read_text())
    assert meta["file"]["n"] == 6 and meta["queries"]["sha256"] == sha and meta["working_tree_dirty"] is False
    assert "--shard <i>/2" in meta["command"] and [s["lists"] for s in meta["shards"]] == [3, 3]
    assert meta["summary"]["fixture"]["source_first_with_teacher"] == 6
    assert "never a relevance judgment" in meta["labels"]


def test_the_teacher_is_the_one_m41_measured_at_the_served_shape():
    assert dt.CANDIDATES_TOP == 20 and dt.FIRST_STAGE == "hybrid@bge-small"
    from app.eval.small_rerankers import SHAPE, TEACHER_MODEL

    assert TEACHER_MODEL == "mixedbread-ai/mxbai-rerank-large-v2" and SHAPE["precision"] == "fp32" and SHAPE["max_length"] == 512
