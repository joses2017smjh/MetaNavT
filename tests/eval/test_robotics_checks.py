"""The chunking and multi-query checks on the robotics snapshot (app/eval/robotics_checks.py).

Every check here runs on a throwaway snapshot frozen from a temporary git
repository; no test scores the committed snapshot's questions.
"""

import hashlib
import inspect
import json
import math
import platform
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.eval import robotics_checks as rc
from app.eval.robotics import freeze, group_bootstrap, sha256
from app.retrieval.bm25 import BM25Index
from app.retrieval.embedders import HashEmbedder
from app.retrieval.hybrid import InMemoryHybridIndex

WORDS = "gait torque yaw pitch roll contact slip stride terrain ramp step stair grip camera lidar".split()


def _long_summary(i: int) -> bytes:
    """A summary longer than one fixed window, so the two chunkers cut it differently."""
    notes = " ".join(WORDS[(i + j) % len(WORDS)] for j in range(120))
    return json.dumps({"success": bool(i % 2), "steps": 100 + i, "verdict": "PASS", "notes": notes}, indent=2).encode()


def _families(names: list[str]) -> tuple[int, int]:
    """Questions per split when every family has three questions: all of them are kept."""
    dev = sum(int(sha256(name.encode())[:8], 16) % 3 == 0 for name in names)
    return 3 * dev, 3 * (len(names) - dev)


def _snapshot(tmp_path, families: list[str]):
    repo = tmp_path / "source"
    repo.mkdir()
    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.name", "Test fixture")
    git("config", "user.email", "fixture@example.invalid")
    git("remote", "add", "origin", "https://example.invalid/robot.git")
    for i, family in enumerate(families):
        target = repo / family / "summary.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(_long_summary(i))
    git("add", "results")
    git("commit", "-qm", "fixture")
    dev_n, test_n = _families(families)
    destination = tmp_path / "snapshot"
    freeze(repo, destination, n_files=len(families), dev_n=dev_n, test_n=test_n)
    return destination


CAMPAIGNS = [f"results/campaign-{i}" for i in range(12)]
# sha256 of json.dumps(DECLARATION, sort_keys=True) as declared before the test split was scored.
DECLARATION_SHA256 = "a7fac0fe5a26f22643ed1c23cbacca6dabb4733822276458704e173dff2d9221"


def test_declaration_is_pinned():
    """What was declared before the test split was scored. Changing a setting here changes the declaration."""
    assert rc.CHECKS == {"chunking": ("auto", "fixed"), "multiquery": ("hybrid", "hybrid+multiquery")}
    assert (rc.N_PATHS, rc.PRIMARY_METRIC, rc.MQ_QUERIES, rc.EMBEDDER) == (50, "ndcg@10", 3, "hash")
    assert rc.METRICS == ("ndcg@10", "mrr@10", "recall@5", "recall@10", "recall@50")
    assert rc.FIXED_WINDOW == {"size": 512, "overlap": 50}
    # The settings the declaration states in words.
    assert HashEmbedder().dim == 256
    assert (BM25Index().k1, BM25Index().b, BM25Index().analyzer) == (1.5, 0.75, "default")
    assert inspect.signature(InMemoryHybridIndex).parameters["rrf_k"].default == 60
    assert inspect.signature(group_bootstrap).parameters["n_boot"].default == 2000
    assert "random.Random(20261007)" in inspect.getsource(group_bootstrap)
    text = json.dumps(rc.DECLARATION, sort_keys=True)
    assert hashlib.sha256(text.encode()).hexdigest() == DECLARATION_SHA256, text


def test_path_list_cuts_the_fused_ranking_at_distinct_paths(monkeypatch):
    monkeypatch.setattr(rc, "N_PATHS", 12)
    hits = [SimpleNamespace(chunk=SimpleNamespace(path=f"f{n}")) for n in (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 13)]
    paths, tenth = rc.path_list(hits)
    assert paths == [f"f{n}" for n in range(12)]
    assert tenth == 13  # the chunk at rank 13 brought the tenth distinct path, f9


def test_chunking_check_scores_both_strategies_on_the_same_paths(tmp_path):
    snapshot = _snapshot(tmp_path, CAMPAIGNS)
    result = rc.run_check("chunking", snapshot)
    assert (result["parent"], result["arm"]) == ("auto", "fixed")
    auto, fixed = result["ingest"]["auto"], result["ingest"]["fixed"]
    assert auto["files_seen"] == fixed["files_seen"] == 12 and auto["files_rejected"] == fixed["files_rejected"] == 0
    assert fixed["chunks_embedded"] > auto["chunks_embedded"] == 12  # one JSON value each against several windows
    assert fixed["window"] == {"size": 512, "overlap": 50} and auto["chunker_version"]
    assert result["snapshot"]["tracked_blob_mismatches"] == {"n_files": 0, "paths": []}
    assert result["declaration"] == rc.DECLARATION
    for split in ("dev", "test"):
        block = result["summary"][split]
        assert block["n_questions"] == sum(r["split"] == split for r in result["per_query"])
        assert block["questions_whose_relevant_file_is_longer_than_one_window"] == block["n_questions"]
        delta = block["paired_delta"]["ndcg@10"]
        assert (delta["of"], delta["minus"], delta["resampling_unit"]) == ("fixed", "auto", "experiment_family")
        assert delta["mean"] == pytest.approx(block["fixed"]["ndcg@10"]["mean"] - block["auto"]["ndcg@10"]["mean"])
        assert block["verdict"] in {"above zero", "covers zero", "below zero", "identical on every question"}
    for row in result["per_query"]:
        for arm in ("auto", "fixed"):
            scored = row["arms"][arm]
            assert len(scored["top10"]) == 10 and scored["relevant_rank"] is not None  # 12 files: every path is listed
            rank = scored["relevant_rank"]
            assert scored["ndcg@10"] == pytest.approx(1 / math.log2(rank + 1) if rank <= 10 else 0.0)


def test_multiquery_check_counts_firing_and_keeps_the_parent_list_where_no_rule_fires(tmp_path):
    families = CAMPAIGNS + ["results/run_47"]  # "run 47" matches a rewrite rule; the campaign questions match none
    snapshot = _snapshot(tmp_path, families)
    result = rc.run_check("multiquery", snapshot)
    run_47_split = "dev" if int(sha256(b"results/run_47")[:8], 16) % 3 == 0 else "test"
    for split in ("dev", "test"):
        block = result["summary"][split]
        expected = 3 if split == run_47_split else 0
        assert block["fired"] == expected
        if not expected:
            assert block["verdict"] == "rules did not fire"
    for row in result["per_query"]:
        assert row["fired"] == (row["family"] == "results/run_47")
        if not row["fired"]:
            assert row["arms"]["hybrid+multiquery"] == row["arms"]["hybrid"] and row["queries"] is None
        else:
            assert "run 47" in row["queries"]


def test_multiquery_check_stops_when_an_unfired_question_changes_its_list(tmp_path, monkeypatch):
    snapshot = _snapshot(tmp_path, CAMPAIGNS)
    real = rc.multi_query_retrieve

    def reversed_lists(index, query, **kwargs):
        result = real(index, query, **kwargs)
        result.hits.reverse()
        return result

    monkeypatch.setattr(rc, "multi_query_retrieve", reversed_lists)
    with pytest.raises(RuntimeError, match="no rewrite fired"):
        rc.run_check("multiquery", snapshot)


def test_every_retrieval_asks_for_every_chunk(tmp_path, monkeypatch):
    """The declared depth: each search of either check (both chunking arms, the multi-query parent and every rewrite
    inside multi_query_retrieve) asks the index for all its chunks, not the index's default depth."""
    snapshot = _snapshot(tmp_path, CAMPAIGNS)
    real = InMemoryHybridIndex.retrieve
    signature = inspect.signature(real)
    depths = []

    def recording(self, *args, **kwargs):
        depths.append((signature.bind(self, *args, **kwargs).arguments.get("k"), len(self.chunks)))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(InMemoryHybridIndex, "retrieve", recording)
    for check in ("chunking", "multiquery"):
        depths.clear()
        result = rc.run_check(check, snapshot, splits=("dev",))
        searches_per_question = 2 if check == "chunking" else 1 + rc.MQ_QUERIES
        assert len(depths) == searches_per_question * len(result["per_query"]) > 0
        assert all(k == n_chunks for k, n_chunks in depths), (check, depths[:4])


def test_dev_only_run_scores_no_test_question(tmp_path):
    snapshot = _snapshot(tmp_path, CAMPAIGNS)
    result = rc.run_check("chunking", snapshot, splits=("dev",))
    assert list(result["summary"]) == ["dev"] and {r["split"] for r in result["per_query"]} == {"dev"}
    with pytest.raises(ValueError, match="splits"):
        rc.run_check("chunking", snapshot, splits=("train",))


def test_results_file_records_no_host_user_or_absolute_path(tmp_path, monkeypatch):
    from app.eval.provenance import user_names

    snapshot = _snapshot(tmp_path, CAMPAIGNS)
    out = tmp_path / "out" / "robotics_chunking.json"
    monkeypatch.setattr("sys.argv", ["robotics_checks", "chunking", "--snapshot", str(snapshot), "--out", str(out), "--splits", "dev"])
    rc.main()
    text = out.read_text()
    assert json.loads(text)["snapshot"]["path"] == "<path>/snapshot"
    for name in {platform.node(), socket.gethostname(), *user_names()}:
        if len(name) >= 3:
            assert name not in text
    assert str(tmp_path) not in text and "/nfs/" not in text and "/home/" not in text


ROOT = Path(__file__).resolve().parents[2]


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_the_docs_state_the_recorded_robotics_outcomes():
    """The typed outcome sentences of README.md and the engineering reference agree with the committed results files
    they cite (a re-recording with another outcome must rewrite them). Reads committed output and scores nothing."""
    chunking = json.loads((ROOT / "bench" / "results" / "robotics_chunking.json").read_text())
    multiquery = json.loads((ROOT / "bench" / "results" / "robotics_multiquery.json").read_text())
    for blob in (chunking, multiquery):
        assert blob["working_tree_dirty"] is False and blob["declaration"] == rc.DECLARATION
        assert blob["splits"] == ["dev", "test"] and blob["snapshot"]["path"] == rc.DEFAULT_SNAPSHOT
    reference = (ROOT / "doc" / "engineering-reference.md").read_text()
    para = _flat(next(p for p in reference.split("\n\n") if p.startswith("[Chunking results](../bench/results/robotics_chunking.json)")))
    assert "on both splits the paired interval of nDCG@10, fixed window minus structure-aware chunker, is above zero" in para
    for split in ("dev", "test"):
        delta = chunking["summary"][split]["paired_delta"]["ndcg@10"]
        assert chunking["summary"][split]["verdict"] == "above zero" and (delta["of"], delta["minus"]) == ("fixed", "auto")
    dev, test = multiquery["summary"]["dev"], multiquery["summary"]["test"]
    assert 'the development split\'s verdict is "rules did not fire"' in para
    assert dev["verdict"] == "rules did not fire" and dev["fired"] == 0
    assert ("the rules fire only through a false trigger (the rule that looks for `lr` matches inside the word "
            "`fullroute`), which changes no question\'s score") in para
    assert test["fired"] > 0 and test["questions_with_a_different_score"] == 0
    fired = [r for r in multiquery["per_query"] if r["fired"]]
    assert {r["split"] for r in fired} == {"test"} and len(fired) == test["fired"]
    for row in fired:
        question = row["queries"][0].lower()
        assert row["queries"][1] == "current " + row["queries"][0]  # the "current" rule, not an identifier rewrite
        assert "fullroute" in question and "lr" not in question.replace("fullroute", "")
        assert not any(word in question for word in ("learning rate", "draft", "config"))
    # The prototype table rows and the README sentence.
    flat = _flat(reference)
    assert "the fixed window is above the structure-aware chunker on both splits" in flat
    assert "the rules almost never fire, and where they do it is a false trigger" in flat
    assert "where those rules almost never fire, so that check does not exercise multi-query" in _flat((ROOT / "README.md").read_text())
    assert (dev["fired"] + test["fired"]) / (dev["n_questions"] + test["n_questions"]) < 0.05  # "almost never"
