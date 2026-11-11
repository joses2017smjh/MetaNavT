"""Task-local Postgres and API launcher (app/eval/service.py) and the HTTP service script that uses it.

No Apptainer, Postgres or model here: subprocess is replaced, so these tests check the commands
that would run and what the script writes, not the services themselves.
"""

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from app.eval import service

ROOT = Path(__file__).resolve().parents[2]


def load_script():
    spec = importlib.util.spec_from_file_location("revamp_http_service", ROOT / "scripts" / "revamp_http_service.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bind_of(command: list[str]) -> str:
    assert command[:3] == ["apptainer", "exec", "--cleanenv"], command
    return command[command.index("--bind") + 1]


def test_apptainer_binds_only_the_run_directory(tmp_path):
    pg = service.TaskPostgres(Path("/images/pgvector.sif"), tmp_path / "pg", port=5555)
    command = pg.command("initdb", "-D", str(pg.data))
    assert bind_of(command) == str((tmp_path / "pg").resolve())
    assert command[5:] == [str(Path("/images/pgvector.sif").resolve()), f"{service.POSTGRES_BIN}/initdb", "-D", str(pg.data)]
    assert service.apptainer_exec(Path("/images/pgvector.sif"), []) == ["apptainer", "exec", "--cleanenv", "/images/pgvector.sif"]


def test_connection_strings_name_no_user(tmp_path):
    pg = service.TaskPostgres(Path("/images/pgvector.sif"), tmp_path, port=5555)
    assert pg.env() == {
        "PG_CONNECTION_STRING": "postgresql://127.0.0.1:5555/metanavit",
        "PSYCOPG2_CONNECTION_STRING": "dbname=metanavit host=127.0.0.1 port=5555",
    }


def test_base_environment_drops_connection_and_indexing_variables(monkeypatch):
    monkeypatch.setenv("PG_CONNECTION_STRING", "postgresql://elsewhere/db")
    monkeypatch.setenv("INDEX_ON_START", "true")
    env = service.base_environment({"RETRIEVE_K": "50"})
    assert "PG_CONNECTION_STRING" not in env and "INDEX_ON_START" not in env
    assert env["RETRIEVE_K"] == "50"


def test_task_postgres_starts_once_initialised_and_stops(tmp_path, monkeypatch):
    calls = []

    class FakePopen:
        def __init__(self, command, **kwargs):
            calls.append(("popen", command))
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

    def fake_run(command, **kwargs):
        calls.append(("run", command))
        if command[6].endswith("/initdb"):
            Path(command[8]).mkdir(parents=True)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(service.subprocess, "run", fake_run)
    monkeypatch.setattr(service.subprocess, "Popen", FakePopen)
    pg = service.TaskPostgres(Path("/images/pgvector.sif"), tmp_path / "pg", port=5555)
    with pg:
        assert pg.initialized
    programs = [Path(c[6]).name for _, c in calls]
    assert programs == ["initdb", "postgres", "pg_isready", "createdb", "pg_ctl"]
    assert {bind_of(c) for _, c in calls} == {str((tmp_path / "pg").resolve())}
    calls.clear()
    with service.TaskPostgres(Path("/images/pgvector.sif"), tmp_path / "pg", port=5556) as again:  # existing data: no initdb, no createdb
        assert not again.initialized
    assert [Path(c[6]).name for _, c in calls] == ["postgres", "pg_isready", "pg_ctl"]


def test_http_service_script_binds_its_scratch_directory_and_scrubs_its_failure_file(tmp_path, monkeypatch):
    """The script once bound a hard-coded home path into Apptainer and wrote its failure file unsanitized."""
    data = tmp_path / "corpus"
    data.mkdir()
    (data / "a.md").write_text("alpha\n")
    queries = tmp_path / "q.jsonl"
    queries.write_text(json.dumps({"id": "q1", "question": "alpha?"}) + "\n")
    out = tmp_path / "out"
    scratch_parent = tmp_path / "tmp"
    scratch_parent.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch_parent))  # where the script's scratch directory goes
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        raise subprocess.CalledProcessError(1, command)  # initdb fails: the error quotes the whole command

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("nothing may start after initdb failed"))
    module = load_script()
    monkeypatch.setattr(sys, "argv", ["revamp_http_service.py", "--container", str(tmp_path / "images" / "pg.sif"), "--data", str(data),
                                      "--queries", str(queries), "--out-dir", str(out), "--device", "cpu"])
    with pytest.raises(subprocess.CalledProcessError):
        module.main()
    assert commands, "initdb was not attempted"
    for command in commands:
        bind = Path(bind_of(command))
        assert bind.parent.parent == scratch_parent.resolve(), bind  # <TMPDIR>/<the script's temporary directory>/postgres
    failure = (out / "setup_failure.json").read_text()
    assert str(tmp_path) not in failure and "/usr/lib/postgresql" not in failure
    assert "<path>/initdb" in failure  # the command is still named, as a placeholder path
    assert json.loads(failure)["error"].startswith("CalledProcessError")
