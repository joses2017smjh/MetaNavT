"""A42 (= D05): one helper, app.eval.provenance.run_provenance, gives the benchmark writers the full commit, whether the
working tree had uncommitted changes, and the sha256 of the source files that ran; report.py says "with uncommitted
changes" next to the commit of a file whose flag is true. (Which writers are not converted yet: the CPU benchmark
section of doc/engineering-reference.md.)

Each benchmark writer runs here in a temporary git repository that holds a copy of fixture v1 and an uncommitted change.
"""

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import app.eval.provenance as prov
from app.eval.provenance import REPO_ROOT, git_state, loaded_sources, run_provenance, working_tree_dirty

ROOT = REPO_ROOT
CORPUS = ROOT / "bench" / "corpus"
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _git(cwd: Path, *args: str) -> str:
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false"]
    return subprocess.run(["git", *identity, *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _repo(base: Path, *, fixture: bool = False) -> tuple[Path, str]:
    """A git repository at `base` with one commit (a notes file, plus a copy of fixture v1 when `fixture`) and its HEAD."""
    base.mkdir(parents=True)
    (base / "notes.md").write_text("committed\n")
    if fixture:
        shutil.copytree(CORPUS / "files", base / "bench" / "corpus" / "files")
        shutil.copy2(CORPUS / "MANIFEST.json", base / "bench" / "corpus" / "MANIFEST.json")
        shutil.copytree(ROOT / "bench" / "gold", base / "bench" / "gold")
    _git(base, "init", "-q")
    _git(base, "add", "-A")
    _git(base, "commit", "-q", "-m", "fixture v1 copy")
    return base, _git(base, "rev-parse", "HEAD")


def _sha256(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


@pytest.fixture
def dirty_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))  # git never answers for a directory above tmp_path
    repo, head = _repo(tmp_path / "repo", fixture=True)
    (repo / "notes.md").write_text("an uncommitted change\n")
    return repo, head


def _assert_dirty_header(blob: dict, head: str, module: str) -> None:
    assert blob["git_head"] == head and head.startswith(blob["git_sha"])
    assert blob["working_tree_dirty"] is True
    assert blob["source_sha256"][module] == _sha256(module)
    assert list(blob["source_sha256"]) == sorted(blob["source_sha256"])


# ---- the helper

def test_the_helper_reads_the_commit_and_whether_the_tree_changed(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    repo, head = _repo(tmp_path / "repo")
    assert len(head) == 40 and git_state(repo) == {"git_head": head, "working_tree_dirty": False}
    (repo / "notes.md").write_text("edited, not committed\n")
    assert git_state(repo) == {"git_head": head, "working_tree_dirty": True}
    _git(repo, "checkout", "-q", "--", "notes.md")
    (repo / "scratch.txt").write_text("untracked\n")
    assert working_tree_dirty(repo) is True  # an untracked file counts, as in the writers that recorded it before
    assert working_tree_dirty(repo, untracked=False) is False
    assert working_tree_dirty(repo, exclude=[repo / "scratch.txt"]) is False
    (repo / ".gitignore").write_text("scratch.txt\n.gitignore\n")
    assert working_tree_dirty(repo) is False  # ignored files never count
    plain = tmp_path / "plain"
    plain.mkdir()
    assert git_state(plain) == {"git_head": None, "working_tree_dirty": None}


def test_source_hashes_default_to_the_app_modules_this_process_imported():
    import app.eval.harness  # noqa: F401 - puts the harness and what it imports in sys.modules, which loaded_sources() reads

    names = loaded_sources()
    assert {"app/eval/provenance.py", "app/eval/harness.py", "app/retrieval/hybrid.py"} <= set(names)
    assert names == sorted(names) and all(n.startswith("app/") and n.endswith(".py") for n in names)
    assert not any(n.startswith("tests/") for n in names)
    block = run_provenance(started={"git_head": "f" * 40, "working_tree_dirty": False})
    assert list(block) == ["git_head", "working_tree_dirty", "source_sha256"]
    assert block["git_head"] == "f" * 40 and block["working_tree_dirty"] is False
    assert block["source_sha256"]["app/eval/harness.py"] == _sha256("app/eval/harness.py")
    named = run_provenance(sources=["app/eval/stats.py"], started={"git_head": None, "working_tree_dirty": None})
    assert named["source_sha256"] == {"app/eval/stats.py": _sha256("app/eval/stats.py")}


# ---- each benchmark writer, in a temporary repository with an uncommitted change

def test_the_bench_header_records_the_commit_a_dirty_tree_and_the_modules(dirty_repo, tmp_path):
    from app.eval.harness import DEFAULT_CONFIGS, run_bench

    repo, head = dirty_repo
    blob = run_bench(root=repo, configs=[replace(DEFAULT_CONFIGS[1])], out=tmp_path / "bm25.json")
    _assert_dirty_header(blob, head, "app/eval/harness.py")
    assert json.loads((tmp_path / "bm25.json").read_text())["working_tree_dirty"] is True


def test_the_bench_reads_the_tree_when_it_starts_not_when_it_writes(tmp_path, monkeypatch):
    """The run's own outputs (here an untracked file written while the rows run) must not mark its tree dirty."""
    from app.eval import harness

    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    repo, head = _repo(tmp_path / "repo", fixture=True)
    real = harness.run_config

    def run_config_that_writes(*args, **kwargs):
        (repo / "bench" / "results").mkdir(parents=True, exist_ok=True)
        (repo / "bench" / "results" / "partial.json").write_text("{}\n")
        return real(*args, **kwargs)

    monkeypatch.setattr(harness, "run_config", run_config_that_writes)
    blob = harness.run_bench(root=repo, configs=[replace(harness.DEFAULT_CONFIGS[1])], out=tmp_path / "bm25.json")
    assert blob["git_head"] == head and blob["working_tree_dirty"] is False and working_tree_dirty(repo) is True


def test_the_parity_header_records_the_commit_a_dirty_tree_and_the_modules(dirty_repo, monkeypatch):
    from app.eval import parity
    from app.eval.metrics import aggregate_retrieval

    def fake_run_config(cfg, gold, files_root, n_files=None, **kwargs):
        per_query = [{"id": q.id, "retrieved": q.relevant_ids(), "relevant": q.relevant_ids(), "category": q.category} for q in gold]
        retrieval = aggregate_retrieval(per_query, n_files=n_files)
        return {"config": cfg.name, "settings": {"name": cfg.name}, "retrieval": retrieval.as_dict(), "latency": {},
                "_per_query_scores": retrieval.per_query}

    repo, head = dirty_repo
    monkeypatch.setattr(parity, "wait_for_health", lambda url: {"status": "ok"})
    monkeypatch.setattr(parity, "run_config", fake_run_config)
    monkeypatch.setattr(parity, "post_retrieve", lambda url, query, k=50, timeout=60.0: {"route": "semantic", "hits": [{"path": "README.md"}]})
    _assert_dirty_header(parity.run("http://127.0.0.1:9", root=repo), head, "app/eval/parity.py")


def test_the_beir_header_records_the_commit_a_dirty_tree_and_the_modules(dirty_repo, tmp_path):
    pytest.importorskip("snowballstemmer")
    from app.eval import beir

    data = tmp_path / "data" / "scifact"
    (data / "qrels").mkdir(parents=True)
    docs = {"d1": ("Cell division", "Cells divide during growth."), "d2": ("Vaccines", "Vaccination reduces measles.")}
    (data / "corpus.jsonl").write_text("".join(json.dumps({"_id": k, "title": t, "text": x}) + "\n" for k, (t, x) in docs.items()))
    (data / "queries.jsonl").write_text(json.dumps({"_id": "q1", "text": "do cells divide"}) + "\n")
    (data / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n")
    repo, head = dirty_repo
    blob = beir.run(tmp_path / "data", [beir.BeirConfig(name="bm25", mode="bm25")], device="cpu", n_boot=20, download=False, root=repo)
    _assert_dirty_header(blob, head, "app/eval/beir.py")


def test_the_rerank_arms_header_records_the_commit_a_dirty_tree_and_the_modules(dirty_repo, monkeypatch):
    """The cross-encoder replaced by the overlap reranker and bge-small by the hash embedder, as in test_frontier_eval."""
    import app.eval.rerank_arms as arms
    from app.retrieval.rerank import OverlapReranker

    class StandInTeacher:
        def __init__(self, model_name, **kwargs):
            self.model = object()

        def __call__(self, query, pairs):
            return OverlapReranker()(query, pairs)

    real_build = arms.build_index
    monkeypatch.setattr(arms, "build_index", lambda root, **kw: real_build(root, **{**kw, "embedder_name": "hash"}))
    monkeypatch.setattr(arms, "CrossEncoderReranker", StandInTeacher)
    monkeypatch.setitem(sys.modules, "lightgbm", None)  # the LightGBM student is recorded as not evaluated
    repo, head = dirty_repo
    _assert_dirty_header(arms.run(root=repo), head, "app/eval/rerank_arms.py")


# ---- the writers that recorded the flag their own way now take it from the helper, with the same fields

def test_the_writers_with_their_own_block_take_it_from_the_helper(monkeypatch):
    from app.engine import ingest_scale
    from app.eval import companion_live, m4_provenance, robotics_checks

    monkeypatch.setattr(prov, "git_head", lambda root=None: "e" * 40)
    monkeypatch.setattr(prov, "working_tree_dirty", lambda root=None, **kwargs: True)
    for block in (m4_provenance.provenance(), companion_live.run_provenance()):
        assert list(block) == ["run_at_utc", "git_head", "working_tree_dirty", "source_sha256"]
        assert block["git_head"] == "e" * 40 and block["working_tree_dirty"] is True
        assert all(block["source_sha256"][rel] == _sha256(rel) for rel in block["source_sha256"])
    assert set(m4_provenance.provenance()["source_sha256"]) == set(m4_provenance.SOURCES)
    assert set(companion_live.run_provenance()["source_sha256"]) == set(companion_live.SOURCES)
    assert robotics_checks._working_tree_dirty() is True
    assert robotics_checks._code_sha256() == {rel: _sha256(rel) for rel in robotics_checks.CODE_PATHS}
    assert ingest_scale.worktree_clean(ingest_scale.OUT) is False
    monkeypatch.setattr(prov, "working_tree_dirty", lambda root=None, **kwargs: None)  # git cannot say: not clean
    assert ingest_scale.worktree_clean(ingest_scale.OUT) is False
    monkeypatch.setattr(prov, "working_tree_dirty", lambda root=None, **kwargs: False)
    assert ingest_scale.worktree_clean(ingest_scale.OUT) is True


def test_the_other_benchmark_writers_record_the_block():
    """The writers that recorded a short commit only. Their headers are built inline, so this reads their source."""
    for rel in ("app/eval/rrf_sweep.py", "app/eval/hybrid_drops.py", "app/eval/sweeps.py", "app/eval/frontier_llm.py",
                "app/engine/ingest_runs.py"):
        assert "**run_provenance(" in (ROOT / rel).read_text(), rel


# ---- report.py

def test_the_report_says_when_a_results_file_came_from_an_uncommitted_tree():
    from app.eval import report

    def load(rel: str) -> dict:
        return json.loads((ROOT / rel).read_text())

    main, neural = load("bench/results/main.json"), load(report.NEURAL_FILE)
    frontier, arms = load(report.FRONTIER_FILE), load(report.RERANK_ARMS_FILE)
    parity, beir = load(report.PARITY_FILE), load(report.BEIR_FILE)
    renders = [lambda: report.table(main), lambda: report.neural_table(neural), lambda: report.frontier_table(frontier),
               lambda: report.rerank_arms_table(arms), lambda: report.parity_table(parity), lambda: report.beir_table(beir)]
    for blob in (main, neural, frontier, arms, parity, beir):
        blob["working_tree_dirty"] = False
    assert not any("with uncommitted changes" in render() for render in renders)
    for blob in (main, neural, frontier, arms, parity, beir):
        blob["working_tree_dirty"] = True
    assert f"sha {main['git_sha']} with uncommitted changes" in report.table(main)
    for blob, render in zip((main, neural, frontier, arms, parity, beir), renders):
        assert f"{blob['git_sha']} with uncommitted changes" in render(), render
    # the served-scale files (M12) record the flag as git_dirty
    assert report.scale_sha({"git_sha": "abc1234", "git_dirty": False}) == "results sha abc1234"
    assert report.scale_sha({"git_sha": "abc1234", "git_dirty": True}) == "results sha abc1234 with uncommitted changes"
