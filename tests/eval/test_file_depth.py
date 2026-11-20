"""M37: the file-depth cut declared before any run, and the rows recorded under it beside the committed chunk-cut rows."""

import json
import sys
from pathlib import Path

import pytest

from app.eval import file_depth, per_question
from app.eval.gold import load_gold
from app.eval.harness import CI_METRICS
from app.eval.index_loader import build_index
from app.eval.rrf_sweep import files_of

ROOT = Path(__file__).resolve().parents[2]


def _path(cid: str) -> str:
    return cid.split("#")[0]


def test_the_rule_is_declared_as_the_owner_worded_it():
    assert file_depth.FILE_DEPTH_RULE == (
        "file-depth cut: every list (the BM25 leg, the dense leg and the fused list) keeps chunks in rank order until it "
        "holds retrieve_k distinct files (or its candidates run out); metrics are unchanged.")


def test_the_cut_keeps_chunks_in_rank_order_until_the_list_holds_retrieve_k_files():
    ranked = ["a#0", "a#1", "b#0", "c#0", "b#1", "d#0", "a#2"]
    assert file_depth.file_depth_cut(ranked, _path, 3) == ["a#0", "a#1", "b#0", "c#0"]  # stops at the third file
    assert file_depth.file_depth_cut(ranked, _path, 1) == ["a#0"]
    assert file_depth.file_depth_cut(ranked, _path, 4) == ["a#0", "a#1", "b#0", "c#0", "b#1", "d#0"]


def test_a_list_with_fewer_files_than_retrieve_k_keeps_every_candidate():
    ranked = ["a#0", "b#0", "a#1"]
    assert file_depth.file_depth_cut(ranked, _path, 50) == ranked
    assert file_depth.file_depth_cut([], _path, 50) == []


def test_a_chunk_cut_list_can_hold_fewer_files_than_the_file_depth_list():
    # the reason for the rule: 4 chunks hold 2 files under a 4-chunk cut, while the file-depth list reaches 4 files
    ranked = ["a#0", "a#1", "b#0", "b#1", "c#0", "d#0"]
    chunk_cut = ranked[:4]
    assert len({_path(c) for c in chunk_cut}) == 2
    assert len({_path(c) for c in file_depth.file_depth_cut(ranked, _path, 4)}) == 4


@pytest.fixture(scope="module")
def gold():
    return load_gold(ROOT / "bench" / "gold" / "questions.jsonl")


@pytest.fixture(scope="module")
def hash_index():
    return build_index(ROOT / "bench" / "corpus" / "files", embedder_name="hash", retrieve_k=50, enable_router=False,
                       enable_rerank=False, mtimes="manifest")


@pytest.fixture(scope="module")
def hash_run(tmp_path_factory):
    """A CPU check of the harness with the hash embedder (its chunk-cut parents are main.json's rows), written as main()
    writes it."""
    out = tmp_path_factory.mktemp("file_depth") / "file_depth.json"
    argv = sys.argv
    sys.argv = [str(ROOT / "app" / "eval" / "file_depth.py"), "--embedders", "hash", "--device", "cpu", "--out", str(out)]
    try:
        code = file_depth.main()
    finally:
        sys.argv = argv
    return code, out, json.loads(out.read_text())


def test_the_chunk_cut_parents_are_what_the_index_retrieves(hash_index, gold):
    """The parent rows are the default path: per gold question, the committed cut rebuilt from the whole leg rankings
    equals InMemoryHybridIndex.retrieve's files (router and reranker off) and the two legs' top 50 chunks."""
    legs = file_depth.legs_in_full(hash_index, [q.question for q in gold])
    for q, (bm25, dense) in zip(gold, legs):
        cut = file_depth.chunk_lists(bm25, dense, 50)
        served = [h.chunk.chunk_id for h in hash_index.retrieve(q.question).hits]
        assert files_of(cut["hybrid"], hash_index) == files_of(served, hash_index)
        assert cut["bm25"] == [c.chunk_id for c, _ in hash_index.search_bm25(q.question, k=50)]
        assert cut["dense"] == [c.chunk_id for c, _ in hash_index.search_dense(q.question, k=50)]


def test_every_file_depth_list_holds_retrieve_k_files_unless_its_candidates_run_out(hash_index, gold):
    legs = file_depth.legs_in_full(hash_index, [q.question for q in gold])
    for bm25, dense in legs:
        depth = file_depth.depth_lists(hash_index, bm25, dense, 50)
        for leg, whole in (("bm25", bm25), ("dense", dense)):
            n = len(files_of(depth[leg], hash_index))
            assert n == min(50, len(files_of(whole, hash_index)))
            assert depth[leg] == whole[:len(depth[leg])]  # a prefix of the leg's ranking: rank order kept
        assert len(files_of(depth["hybrid"], hash_index)) == min(50, len(files_of(depth["bm25"] + depth["dense"], hash_index)))
        # a file-depth list never holds fewer files than the chunk-cut list of the same leg
        cut = file_depth.chunk_lists(bm25, dense, 50)
        for leg in ("bm25", "dense", "hybrid"):
            assert len(files_of(depth[leg], hash_index)) >= len(files_of(cut[leg], hash_index))


def test_the_hash_check_reproduces_main_json_and_records_every_pair(hash_run, gold):
    code, out, blob = hash_run
    assert code == 0 and blob["reproduces_all"] is True
    e = blob["embedders"]["hash"]
    assert e["reproduces"]["file"] == "bench/results/main.json" and e["reproduces"]["all"] is True
    assert e["chunk_legs_are_prefixes_of_the_whole_rankings"] is True
    assert blob["rule"] == file_depth.FILE_DEPTH_RULE and "unchanged" in blob["default_path"]
    by = {r["config"]: r for r in blob["results"]}
    assert set(by) == {"bm25_only", "bm25_only/file-depth", "dense_only@hash", "dense_only@hash/file-depth", "hybrid@hash",
                       "hybrid@hash/file-depth"}
    refs = {(name, r["delta_vs_previous"]["reference"]) for name, r in by.items() if r.get("delta_vs_previous")}
    refs |= {(c["row"], c["other"]) for c in blob["bootstrap"]["comparisons"]}
    assert refs == {("dense_only@hash", "bm25_only"), ("hybrid@hash", "dense_only@hash"), ("bm25_only/file-depth", "bm25_only"),
                    ("dense_only@hash/file-depth", "dense_only@hash"), ("hybrid@hash/file-depth", "hybrid@hash"),
                    ("hybrid@hash/file-depth", "dense_only@hash/file-depth")}
    assert all(set(r["ci"]) == set(CI_METRICS) for r in blob["results"]) and blob["n_gold"] == len(gold)
    # the per-question sidecar holds every row's scores, and its means are the rows' means
    sidecar = json.loads(per_question.sidecar_path(out).read_text())
    assert sidecar["ids"] == [q.id for q in gold] and set(sidecar["scores"]) == set(by)
    for name, row in by.items():
        scores = sidecar["scores"][name]["recall@50"]
        assert round(sum(scores) / len(scores), 4) == row["ci"]["recall@50"]["mean"]
    # the repository's sidecar contract: every interval and paired delta in the file is recomputed from the sidecar
    assert per_question.differences(blob, sidecar) == []
    assert str(ROOT) not in out.read_text()


def test_the_default_path_does_not_know_the_rule():
    """The rule lives in this module: no retrieval, engine or API code, and none of the harnesses that write the
    committed rows, imports it (only the report and evidence pages read its results file)."""
    serving = [p for p in (ROOT / "app").rglob("*.py") if p.parts[len(ROOT.parts) + 1] != "eval"]
    writers = [ROOT / "app" / "eval" / name for name in ("harness.py", "rrf_sweep.py", "index_loader.py", "hybrid_drops.py")]
    assert [p.relative_to(ROOT).as_posix() for p in serving + writers if "file_depth" in p.read_text()] == []
