"""A42 (= D05): one helper, app.eval.provenance.run_provenance, gives the benchmark writers the full commit, whether the
working tree had uncommitted changes, and the sha256 of the source files that ran; report.py says "with uncommitted
changes" next to the commit of a file whose flag is true. (Which writers are not converted yet: the CPU benchmark
section of doc/engineering-reference.md.)

Each writer runs here in a temporary git repository with an uncommitted change: the benchmark writers on a copy of
fixture v1 held in it, the HTTP load and smoke latency writers against a stand-in API, the robotics lookup evaluation on
a snapshot of a throwaway repository.
"""

import hashlib
import json
import re
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


# ---- the load, latency, answer-evaluation and robotics writers (M10c)

LOAD_HEALTH = {"status": "ok", "n_nodes": 8, "degraded": [], "devices": {"embedding": "cpu"}, "reranker": {"loaded": True, "device": "cpu"}}
LOAD_BODY = {"hits": [{"path": "README.md"}], "latency_ms": {"total": 2.0, "embed": 1.0}, "degraded": [], "reranker": {"loaded": True}}


def _load_transport(seen: list | None = None, out: Path | None = None):
    """A stand-in retrieval API for app/eval/http_load; with `out`, each POST notes whether the file the load is
    writing already starts with the provenance block (an interrupted run keeps it)."""
    import httpx

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json=LOAD_HEALTH)
        if out is not None and seen is not None:
            seen.append(list(json.loads(out.read_text()))[:4])
        return httpx.Response(200, json=LOAD_BODY)

    return httpx.MockTransport(handler)


def _script(name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_http_load_file_records_the_commit_a_dirty_tree_and_the_modules(dirty_repo, tmp_path):
    pytest.importorskip("httpx")
    from app.eval import http_load

    repo, head = dirty_repo
    out, seen = tmp_path / "load.json", []
    blob = http_load.write_report("http://service", [("q1", "ok")], out, root=repo, transport=_load_transport(seen, out),
                                  concurrency_levels=(1,), requests_per_level=2, warmup_requests=0)
    assert blob["outcome"] == "ok"
    _assert_dirty_header(blob, head, "app/eval/http_load.py")
    assert json.loads(out.read_text()) == blob  # the last checkpoint is the report
    assert seen and all(keys == ["git_sha", "git_head", "working_tree_dirty", "source_sha256"] for keys in seen)


def test_the_http_load_reads_the_tree_before_its_first_checkpoint(tmp_path, monkeypatch):
    """The load file is checkpointed while it runs, often to a tracked bench/results file: its own writes must not
    mark the tree dirty. run_load, whose report the served-scale runner nests in a file with its own flag, records none."""
    pytest.importorskip("httpx")
    import asyncio

    from app.eval import http_load

    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    repo, head = _repo(tmp_path / "repo")
    out = repo / "bench" / "results" / "api_latency_load.json"
    blob = http_load.write_report("http://service", [("q1", "ok")], out, root=repo, transport=_load_transport(),
                                  concurrency_levels=(1,), requests_per_level=1, warmup_requests=0)
    assert blob["git_head"] == head and blob["working_tree_dirty"] is False and working_tree_dirty(repo) is True
    bare = asyncio.run(http_load.run_load("http://service", [("q1", "ok")], transport=_load_transport(), concurrency_levels=(1,),
                                          requests_per_level=1, warmup_requests=0))
    assert not {"git_sha", "git_head", "working_tree_dirty", "source_sha256"} & set(bare)


def test_the_http_load_command_line_writes_through_write_report(dirty_repo, tmp_path, monkeypatch):
    """`python -m app.eval.http_load`, the writer the module docstring documents, records the block: the stand-in API
    is reached through run_load, and write_report's default checkout is the temporary repository."""
    pytest.importorskip("httpx")
    from app.eval import http_load

    repo, head = dirty_repo
    real_run_load = http_load.run_load
    monkeypatch.setattr(http_load, "REPO_ROOT", repo)
    monkeypatch.setattr(http_load, "run_load", lambda *a, **kw: real_run_load(*a, transport=_load_transport(), **kw))
    queries, out = tmp_path / "queries.jsonl", tmp_path / "load.json"
    queries.write_text(json.dumps({"id": "q1", "question": "alpha"}) + "\n")
    argv = ["--url", "http://service", "--queries", str(queries), "--out", str(out), "--concurrency", "1", "--requests", "1", "--warmup", "0"]
    assert http_load.main(argv) == 0
    blob = json.loads(out.read_text())
    _assert_dirty_header(blob, head, "app/eval/http_load.py")
    assert list(blob)[:4] == ["git_sha", "git_head", "working_tree_dirty", "source_sha256"] and blob["outcome"] == "ok"


def test_the_http_load_file_passes_through_the_sanitizer(dirty_repo, tmp_path, monkeypatch):
    """Everything write_report writes is scrubbed, not only the url and health block run_load rewrites itself: here a
    readiness error names the host and the caller's deployment block holds an absolute path and the host."""
    pytest.importorskip("httpx")
    import httpx

    from app.eval import http_load

    host = "node-17.cluster.example.edu"
    monkeypatch.setattr(prov.platform, "node", lambda: host)
    monkeypatch.setattr(prov.socket, "gethostname", lambda: host)

    def unreachable(request):
        raise httpx.ConnectError(f"certificate is not valid for {host}", request=request)

    repo, _ = dirty_repo
    out = tmp_path / "load.json"
    blob = http_load.write_report("http://service", [("q1", "ok")], out, root=repo, transport=httpx.MockTransport(unreachable),
                                  deployment={"data_dir": "/scratch/someone/corpus", "node": host},
                                  concurrency_levels=(1,), requests_per_level=1, warmup_requests=0)
    text = out.read_text()
    assert blob["outcome"] == "readiness_failed" and json.loads(text) == blob
    assert "node-17" not in text and "/scratch/someone" not in text
    assert blob["readiness_errors"] == ["ConnectError: certificate is not valid for <host>"]
    assert blob["deployment"] == {"data_dir": "<path>/corpus", "node": "<host>"}


def test_the_revamp_http_file_records_the_tree_as_it_was_when_the_script_started(dirty_repo, tmp_path, monkeypatch):
    """scripts/revamp_http_service.py wrote every committed http_load.json. The services are stand-ins here; the API's
    start reverts the uncommitted change, so a header read after the services started would say the tree was clean."""
    pytest.importorskip("httpx")
    from app.eval import http_load

    revamp = _script("revamp_http_service")
    repo, head = dirty_repo

    class Postgres:
        def __init__(self, container, path):
            pass

        def start(self):
            return self

        def env(self):
            return {"PG_CONNECTION_STRING": "postgresql://127.0.0.1:9/metanavit"}

        def stop(self):
            pass

    class Api:
        url = "http://127.0.0.1:9"

        def __init__(self, environment, cwd, log_path):
            pass

        def start(self):
            _git(repo, "checkout", "-q", "--", "notes.md")
            return self

        def wait_ready(self, timeout):
            return LOAD_HEALTH, None

        def stop(self):
            pass

    real_run_load = http_load.run_load
    monkeypatch.setattr(revamp, "TaskPostgres", Postgres)
    monkeypatch.setattr(revamp, "ApiServer", Api)
    monkeypatch.setattr(revamp, "migrate", lambda environment, cwd, log_path: None)
    monkeypatch.setattr(http_load, "run_load", lambda *a, **kw: real_run_load(*a, transport=_load_transport(), **kw))
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.md").write_text("alpha\n")
    queries = tmp_path / "queries.jsonl"
    queries.write_text(json.dumps({"id": "q1", "question": "alpha"}) + "\n")
    out_dir = tmp_path / "out"
    argv = ["--container", str(tmp_path / "pg.sif"), "--data", str(data), "--queries", str(queries), "--out-dir", str(out_dir),
            "--device", "cpu", "--requests", "1", "--scratch", str(tmp_path / "scratch")]
    assert revamp.main(argv, root=repo) == 0
    blob = json.loads((out_dir / "http_load.json").read_text())
    _assert_dirty_header(blob, head, "app/eval/http_load.py")
    assert working_tree_dirty(repo) is False  # clean by the time the file was written
    assert set(blob["source_sha256"]) == set(blob["deployment"]["source_sha256"])  # the served code and the client


def test_the_smoke_latency_file_records_the_commit_a_dirty_tree_and_the_script(dirty_repo, tmp_path, monkeypatch):
    """scripts/smoke_retrieve.py --latency (bench/results/api_latency_*.json): the script is among the sources, and the
    url and health block pass through the sanitizer."""
    smoke = _script("smoke_retrieve")
    monkeypatch.setattr(prov.platform, "node", lambda: "node-17.cluster.example.edu")
    monkeypatch.setattr(prov.socket, "gethostname", lambda: "node-17.cluster.example.edu")
    monkeypatch.setattr(smoke, "_post", lambda url, payload: (200, {"hits": [{"path": "README.md"}], "latency_ms": {"total": 2.0, "embed": 1.5}}))
    gold = tmp_path / "questions.jsonl"
    gold.write_text("".join(json.dumps({"id": f"q{i}", "question": f"question {i}"}) + "\n" for i in range(3)))
    health = {"embed_model": "/scratch/someone/models/bge-small", "n_nodes": 71, "degraded": []}
    repo, head = dirty_repo
    blob = smoke.replay_latency("http://node-17:8000", str(gold), health, root=repo)
    _assert_dirty_header(blob, head, "scripts/smoke_retrieve.py")
    assert "app/eval/provenance.py" in blob["source_sha256"]
    assert blob["url"] == "http://<host>:8000" and blob["health"]["embed_model"] == "<path>/bge-small"
    assert blob["n"] == 3 and blob["server_total_ms"] == {"p50": 2.0, "p95": 2.0}


JURY_CONFIG = dict(name="hybrid+router@hash", mode="hybrid", embedder="hash", enable_rerank=False, enable_router=True, staleness_tier1=True,
                   log_triples=False, e2e=False)


def _fake_model(prompt: str) -> str:
    """Generator and judges with no model: the answer cites the first source; every verdict is partial."""
    if "Rules:" in prompt:
        return "See " + re.search(r"Source 1 (\[[^\]]+\])", prompt).group(1) + "."
    if "Which is better" in prompt:
        return "TIE"
    return json.dumps({"score": 0.5, "label": "partial"})


def _jury(repo: Path) -> dict:
    from app.eval import llm_eval
    from app.eval.harness import BenchConfig

    return llm_eval.run(root=repo, config=BenchConfig(**JURY_CONFIG), generator_model="fake-gen", judge_models=["fake-a"], limit=2,
                        complete_fn=_fake_model)


def test_the_jury_header_records_the_commit_a_dirty_tree_and_the_modules(dirty_repo):
    repo, head = dirty_repo
    _assert_dirty_header(_jury(repo), head, "app/eval/llm_eval.py")


def test_a_jury_rescore_records_its_own_block_and_carries_the_answers_block(tmp_path, monkeypatch):
    """The header describes the run that last wrote the file (for a rescore, the one that recomputed the checks); the
    rescore block keeps the run that generated the answers. Here the answers come from a clean tree, the first rescore
    runs on a dirty one and the second on a clean one: the answers' flag must stay False through both."""
    from app.eval import llm_eval

    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    repo, head = _repo(tmp_path / "repo", fixture=True)
    blob = _jury(repo)
    assert blob["git_head"] == head and blob["working_tree_dirty"] is False
    path = tmp_path / "jury.json"
    path.write_text(json.dumps(blob))
    (repo / "notes.md").write_text("an uncommitted change\n")
    once = llm_eval.rescore(path, root=repo)
    assert list(once)[:4] == ["git_sha", "git_head", "working_tree_dirty", "source_sha256"]
    assert once["git_head"] == head and once["working_tree_dirty"] is True and once["git_sha"] == once["rescore"]["checks_git_sha"]
    answers = {"rows_git_head": head, "rows_working_tree_dirty": False, "rows_source_sha256": blob["source_sha256"]}
    assert {key: once["rescore"][key] for key in answers} == answers
    path.write_text(json.dumps(once))
    _git(repo, "checkout", "-q", "--", "notes.md")
    twice = llm_eval.rescore(path, root=repo)
    assert twice["working_tree_dirty"] is False and {key: twice["rescore"][key] for key in answers} == answers
    path.write_text(json.dumps({k: v for k, v in blob.items() if k not in ("git_head", "working_tree_dirty", "source_sha256")}))
    earlier = llm_eval.rescore(path, root=repo)  # a file written before the jury writer recorded the block
    assert {key: earlier["rescore"][key] for key in answers} == dict.fromkeys(answers)
    assert list(earlier)[:4] == ["git_sha", "git_head", "working_tree_dirty", "source_sha256"]


def _robotics_snapshot(tmp_path: Path) -> Path:
    """A frozen snapshot of a throwaway robotics repository (twelve campaigns, one summary.json each), as
    tests/eval/test_robotics.py builds one."""
    from app.eval.robotics import freeze

    source = tmp_path / "robot"
    for i in range(12):
        path = source / "results" / f"campaign-{i}" / "summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"success": bool(i % 2), "steps": 100 + i, "verdict": "PASS", "seed": i}))
    _git(source, "init", "-q")
    _git(source, "remote", "add", "origin", "https://example.invalid/robot.git")
    _git(source, "add", "results")
    _git(source, "commit", "-q", "-m", "campaigns")
    freeze(source, tmp_path / "snapshot", n_files=12, dev_n=4, test_n=8)
    return tmp_path / "snapshot"


def test_the_robotics_lookup_file_records_the_commit_a_dirty_tree_and_the_modules(dirty_repo, tmp_path):
    from app.eval.robotics import evaluate

    snapshot = _robotics_snapshot(tmp_path)
    repo, head = dirty_repo
    out = tmp_path / "lookup.json"
    blob = evaluate(snapshot, out, root=repo)
    _assert_dirty_header(blob, head, "app/eval/robotics.py")
    assert json.loads(out.read_text()) == blob
    assert blob["code_sha256"]["eval/robotics.py"] == _sha256("app/eval/robotics.py")  # the mixed-run check keeps its list

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
    # a rescored answer-evaluation file names two runs: the one that generated the answers and the one that rescored them
    for rel in (report.JURY_FILE, report.EARLIER_JURY_FILE):
        jury = load(rel)
        rows, checks = jury["rescore"]["rows_git_sha"], jury["rescore"]["checks_git_sha"]
        assert "with uncommitted changes" not in report.jury_table(jury), rel
        jury["rescore"]["rows_working_tree_dirty"] = True
        text = report.jury_table(jury)
        assert f"generated at sha {rows} with uncommitted changes;" in text and f"at sha {checks} with uncommitted changes" not in text, rel
        jury["rescore"]["rows_working_tree_dirty"], jury["working_tree_dirty"] = False, True
        text = report.jury_table(jury)
        assert f"generated at sha {rows};" in text and f"at sha {checks} with uncommitted changes" in text, rel
