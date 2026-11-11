"""The launcher of the agent-path runs (app/eval/served_stack.py): the default Compose stack's environment,
code execution and the API key off whatever the caller's environment says, and a task-local fixture copy."""

import os
from pathlib import Path

import yaml

from app.eval import served_stack as ss
from app.eval.corpus import load_manifest, mtime_mismatches, verify_manifest

ROOT = Path(__file__).resolve().parents[2]
DSN = "postgresql://someone:secret@127.0.0.1:5432/metanavit"


def test_environment_is_the_compose_api_service_with_the_run_set_variables_left_out():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]["api"]["environment"]
    assert {k: str(v) for k, v in compose.items() if k not in ss.RUN_SET} == ss.COMPOSE_API_ENV
    assert set(ss.RUN_SET) <= set(compose)


def test_code_execution_key_and_budget_stay_off_whatever_the_parent_sets(tmp_path):
    parent = {"PATH": "/usr/bin", "HOME": "/home/someone", "ENABLE_PYTHON_EXEC": "true",
              "METANAVIT_API_KEY": "parent-key", "RATE_LIMIT_PER_MINUTE": "5", "PG_TEST_DSN": DSN,
              "AWS_SECRET_ACCESS_KEY": "x", "FRONTEND_ENDPOINT": "http://localhost:3000"}
    env = ss.api_env(DSN, "mnv_api_1", tmp_path, model="qwen2.5:7b", ollama_url="http://127.0.0.1:11434",
                     parent=parent, extra={"ENABLE_PYTHON_EXEC": "true"})
    assert env["ENABLE_PYTHON_EXEC"] == "false" and env["METANAVIT_API_KEY"] == "" and env["RATE_LIMIT_PER_MINUTE"] == ""
    assert "AWS_SECRET_ACCESS_KEY" not in env and "PG_TEST_DSN" not in env and "FRONTEND_ENDPOINT" not in env
    assert env["PG_CONNECTION_STRING"] == "postgresql://someone:secret@127.0.0.1:5432/mnv_api_1"
    assert env["MODEL"] == "qwen2.5:7b" and env["EMBEDDING_PROVIDER"] == "hash"
    recorded = ss.public_env(env)
    assert "secret" not in str(recorded) and "PATH" not in recorded and str(tmp_path) not in str(recorded)
    assert recorded["ENABLE_PYTHON_EXEC"] == "false"


def test_python_exec_stays_off_in_the_built_environment(monkeypatch, tmp_path):
    from app.engine.tools.python_exec import python_exec_enabled

    env = ss.api_env(DSN, "mnv_api_1", tmp_path, model="m", ollama_url="http://127.0.0.1:1",
                     parent={"ENABLE_PYTHON_EXEC": "true", "METANAVIT_API_KEY": "k"})
    for name in ("ENABLE_PYTHON_EXEC", "METANAVIT_API_KEY"):
        monkeypatch.setenv(name, env[name])
    assert python_exec_enabled() is False


def test_fixture_copy_matches_the_manifest_with_its_mtimes(tmp_path):
    copy = ss.copy_fixture(tmp_path / "files")
    manifest = load_manifest(ROOT / "bench" / "corpus" / "MANIFEST.json")
    assert verify_manifest(copy, manifest) == [] and mtime_mismatches(copy, manifest) == []
    assert ss.tree_sha256(copy) == ss.tree_sha256(ROOT / "bench" / "corpus" / "files")


def test_planting_appends_and_keeps_the_mtime(tmp_path):
    copy = ss.copy_fixture(tmp_path / "files")
    target = copy / "configs" / "run_042.yaml"
    before, mtime = target.read_text(), target.stat().st_mtime
    ss.plant(copy, {"configs/run_042.yaml": "# planted\n"})
    assert target.read_text() == before + "# planted\n"
    assert target.stat().st_mtime == mtime
    assert os.path.getsize(ROOT / "bench" / "corpus" / "files" / "configs" / "run_042.yaml") == len(before.encode())


def test_database_dsn_replaces_only_the_database():
    assert ss.database_dsn("postgresql://u@127.0.0.1:5/postgres", "mnv_x") == "postgresql://u@127.0.0.1:5/mnv_x"
