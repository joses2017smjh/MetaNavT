"""The index step runs on its own, before the API, in Compose and in CI; the API start does not index. No database.

Compose and CI can only run in CI (no Docker here), so the order of their steps is checked here from the files
themselves; tests/database/test_index_step.py runs the command and the API against a live Postgres.
"""

import json
import shlex
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _compose(name="docker-compose.yml") -> dict:
    return yaml.safe_load((ROOT / name).read_text())


def test_the_compose_api_container_migrates_then_indexes_then_serves_without_indexing():
    api = _compose()["services"]["api"]
    assert api["environment"]["INDEX_ON_START"] == "false"  # the server process does not index
    shell, flag, script = api["command"]
    assert (shell, flag) == ("sh", "-c")
    steps = [shlex.split(step) for step in script.split("&&")]  # && : a step that fails stops the container
    assert steps[0] == ["python", "-m", "app.database.migrate"]
    timeout, seconds, *index = steps[1]
    assert timeout == "timeout" and int(seconds) > 0 and index == ["python", "-m", "app.engine.ingest"]
    assert steps[2][:2] == ["exec", "uvicorn"] and len(steps) == 3
    for override in ("docker-compose.ml.yml", "docker-compose.gpu.yml"):  # they change the embedder, not the sequence
        service = _compose(override)["services"]["api"]
        assert "command" not in service and "INDEX_ON_START" not in (service.get("environment") or {})


def test_the_ci_parity_job_runs_the_index_step_between_the_migrations_and_the_api():
    job = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())["jobs"]["parity"]
    assert job["env"]["INDEX_ON_START"] == "false"
    runs = [step.get("run", "") for step in job["steps"]]
    migrate = runs.index("make migrate")
    index = runs.index("make index")
    api = next(i for i, run in enumerate(runs) if "uvicorn main:app" in run)
    assert migrate < index < api
    assert job["steps"][index]["timeout-minutes"] > 0


def test_the_index_command_prints_the_report_and_exits_0(monkeypatch, capsys):
    from app.engine import ingest

    monkeypatch.setattr(ingest, "run_index_step", lambda: {"files_indexed": 61, "chunks_embedded": 71})
    assert ingest.main([]) == 0
    out = capsys.readouterr()
    assert json.loads(out.out.strip().splitlines()[-1]) == {"files_indexed": 61, "chunks_embedded": 71}


@pytest.mark.parametrize("error", ["lock", "pending", "rejected"])
def test_a_failed_index_step_exits_1_and_says_why(monkeypatch, capsys, error):
    from app.database.migrate import MigrationsPending
    from app.database.vector_store_manager import IngestLockTimeout
    from app.engine import ingest
    from app.eval.ingest import IngestRejected

    exc = {
        "lock": IngestLockTimeout("another process held the ingest lock"),
        "pending": MigrationsPending("1 schema migration(s) pending"),
        "rejected": IngestRejected({"files_rejected": 2, "files_seen": 61, "max_rejected": 0, "rejected_by_reason": {"empty": 2}}),
    }[error]

    def fail():
        raise exc

    monkeypatch.setattr(ingest, "run_index_step", fail)
    assert ingest.main([]) == 1
    out = capsys.readouterr()
    assert out.out == "" and out.err.strip().startswith(f"indexing failed: {type(exc).__name__}: ")


def test_every_indexing_path_makes_the_same_call(monkeypatch, tmp_path):
    """DATA_DIR, the configured embedder and the label the manifest records: the same for start-up, the step and generate."""
    from llama_index.core.settings import Settings

    from app.engine import ingest
    from app.engine.embeddings import HashEmbedding

    calls = []
    monkeypatch.setattr(ingest, "index_directory", lambda vsm, data_dir, **kw: calls.append((vsm, data_dir, kw)) or {"ok": True})
    monkeypatch.setattr(Settings, "_embed_model", HashEmbedding(dim=8))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INGEST_MAX_REJECTED", "0")

    assert ingest.index_data_dir("vsm") == {"ok": True}
    ((vsm, data_dir, kw),) = calls
    assert (vsm, data_dir, kw["embedder"], kw["max_rejected"]) == ("vsm", str(tmp_path), "hash-8", 0)
    assert kw["embed_fn"](["a b"]) == Settings.embed_model.get_text_embedding_batch(["a b"])


@pytest.mark.parametrize("raw,expected", [(None, 600.0), ("", 600.0), ("2.5", 2.5), ("0", None), ("-1", 600.0), ("soon", 600.0), ("inf", 600.0)])
def test_the_lock_wait_has_a_limit_unless_it_is_set_to_zero(monkeypatch, raw, expected):
    from app.engine.ingest import DEFAULT_INGEST_LOCK_TIMEOUT_SECONDS, ingest_lock_timeout_from_env

    assert DEFAULT_INGEST_LOCK_TIMEOUT_SECONDS == 600.0
    if raw is None:
        monkeypatch.delenv("INGEST_LOCK_TIMEOUT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("INGEST_LOCK_TIMEOUT_SECONDS", raw)
    assert ingest_lock_timeout_from_env() == expected


def test_a_start_without_index_on_start_reports_what_the_manifest_lists(monkeypatch):
    """/health keeps naming the chunker (app/eval/report.py reads it from a recorded parity run) and the counts."""
    from app.engine import bootstrap

    class Manager:
        hybrid_backend = "ts_rank_cd"

        def check_migrations(self):
            pass

        def manifest_summary(self):
            return {"files": 61, "chunks": 71, "pipelines": ["app.chunking/auto/v2|hash-256"]}

        def search_settings(self, depth):
            return {}

    class Retriever:
        mode, staleness_enabled, depth = "sql", True, 50

    monkeypatch.delenv("INDEX_ON_START", raising=False)
    monkeypatch.setattr(bootstrap, "get_vector_store_manager", Manager)
    monkeypatch.setattr(bootstrap, "index_data_dir", lambda vsm: pytest.fail("the start indexed"))
    monkeypatch.setattr(bootstrap, "build_clusters", lambda vsm: {})
    monkeypatch.setattr(bootstrap, "get_index", lambda: None)
    monkeypatch.setattr(bootstrap, "create_hybrid_retriever", lambda index, vsm, clusters: Retriever())

    state = bootstrap.build_retrieval_state()

    assert state.indexing == {
        "indexed": False,
        "reason": "INDEX_ON_START not set",
        "chunker": "app/chunking (structure-aware)",
        "manifest": {"files": 61, "chunks": 71, "pipelines": ["app.chunking/auto/v2|hash-256"]},
    }


def test_compose_ci_and_the_env_example_set_no_chunk_size_nothing_reads():
    """CHUNK_SIZE / CHUNK_OVERLAP sized the SentenceSplitter of the old generate CLI; the index step cuts with
    app/chunking (CHUNKER_VERSION names its boundaries), so the settings fed nothing that runs."""
    for name in ("docker-compose.yml", "docker-compose.ml.yml", "docker-compose.gpu.yml", ".github/workflows/ci.yml", ".env.example"):
        text = (ROOT / name).read_text()
        assert "CHUNK_SIZE" not in text and "CHUNK_OVERLAP" not in text, name


def test_compose_keeps_uploads_on_a_volume_inside_data_dir():
    """Uploads are indexed with DATA_DIR, so they live in DATA_DIR/uploads (a link out of DATA_DIR would be rejected as
    outside_root): a named volume mounted there, in the container that indexes and serves."""
    compose = _compose()
    api = compose["services"]["api"]
    data_dir = api["environment"]["DATA_DIR"]
    mounts = dict(volume.split(":", 1) for volume in api["volumes"])
    assert mounts == {"metanavit_uploads": f"/app/{data_dir}/uploads"}  # WORKDIR /app (Dockerfile)
    assert "metanavit_uploads" in compose["volumes"]
