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


class FakePostgresProcess:
    def __init__(self, command, **kwargs):
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode


def test_task_postgres_needs_a_password_of_its_own_that_no_command_line_carries(tmp_path, monkeypatch):
    """initdb used to set trust authentication: anyone on a shared node could connect to 127.0.0.1:<port> as the superuser."""
    import stat

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[6].endswith("/initdb"):
            pwfile = Path(next(a for a in command if a.startswith("--pwfile=")).split("=", 1)[1])
            assert stat.S_IMODE(pwfile.stat().st_mode) == 0o600
            calls.append(("password given to initdb", pwfile.read_text().strip()))
            Path(command[8]).mkdir(parents=True)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(service.subprocess, "run", fake_run)
    monkeypatch.setattr(service.subprocess, "Popen", FakePostgresProcess)
    pg = service.TaskPostgres(Path("/images/pgvector.sif"), tmp_path / "pg", port=5555)
    with pg:
        password = pg.env()["PGPASSWORD"]
    assert len(password) >= 32 and ("password given to initdb", password) in calls
    commands = [c for c in calls if isinstance(c[0], list)]
    initdb = next(c for c, _ in commands if c[6].endswith("/initdb"))
    assert "--auth=scram-sha-256" in initdb and not any("trust" in a for a in initdb)
    assert all(password not in " ".join(c) for c, _ in commands)  # never on a command line (ps shows those to every user)
    assert password not in pg.env()["PG_CONNECTION_STRING"] and password not in pg.env()["PSYCOPG2_CONNECTION_STRING"]
    createdb = next(kw for c, kw in commands if c[6].endswith("/createdb"))
    assert createdb["env"]["APPTAINERENV_PGPASSWORD"] == password  # --cleanenv passes only APPTAINERENV_ variables
    # the password lives inside the data directory (initdb makes it 0700), so a restored snapshot still opens
    stored = tmp_path / "pg" / "pgdata" / service.PASSWORD_FILE
    assert stored.read_text().strip() == password and stat.S_IMODE(stored.stat().st_mode) == 0o600
    assert not (tmp_path / "pg" / "initdb.pw").exists()
    with service.TaskPostgres(Path("/images/pgvector.sif"), tmp_path / "pg", port=5556) as again:  # existing data: no initdb
        assert again.env()["PGPASSWORD"] == password
    other = service.TaskPostgres(Path("/images/pgvector.sif"), tmp_path / "other", port=5557)
    with other:
        assert other.env()["PGPASSWORD"] != password  # one password per data directory


def test_a_restored_data_directory_is_made_private_again(tmp_path, monkeypatch):
    """tarfile's data filter ignores a directory's stored mode: the restored data directory took the umask and the parent's
    setgid bit (2770 on the login node's scratch), and postgres refuses to start on a directory with group write access."""
    import stat

    monkeypatch.setattr(service.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0))
    monkeypatch.setattr(service.subprocess, "Popen", FakePostgresProcess)
    data = tmp_path / "pg" / "pgdata"
    data.mkdir(parents=True)
    data.chmod(0o2770)
    with service.TaskPostgres(Path("/images/pgvector.sif"), tmp_path / "pg", port=5555) as pg:
        assert not pg.initialized and pg.password is None  # no password file: a data directory from before passwords were set
    assert stat.S_IMODE(data.stat().st_mode) == 0o700


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
        if command[0] != "git":  # git: the commit and working tree the results header records, read before anything starts
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


def test_migrate_runs_the_deploy_step_against_the_given_environment(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)) or subprocess.CompletedProcess(command, 0))
    seconds = service.migrate({"PG_CONNECTION_STRING": "postgresql://127.0.0.1:5/x", "EMBEDDING_DIM": "384"}, cwd=tmp_path, log_path=tmp_path / "logs" / "migrate.log")
    ((command, kwargs),) = calls
    assert command[1:] == ["-m", "app.database.migrate"] and kwargs["cwd"] == tmp_path and kwargs["check"] is True
    assert kwargs["env"] == {"PG_CONNECTION_STRING": "postgresql://127.0.0.1:5/x", "EMBEDDING_DIM": "384"} and seconds >= 0.0
    assert (tmp_path / "logs" / "migrate.log").exists()


def test_index_step_runs_the_index_command_and_returns_its_report(tmp_path, monkeypatch):
    """The deploy order's index step, `python -m app.engine.ingest`: its report is the last line it prints (a line printed
    to stdout before it, by a library for example, is not the report)."""
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stderr"].write("Indexing report: ...\n")
        return subprocess.CompletedProcess(command, 0, stdout='a line a library printed\n{"chunks_embedded": 0, "files_unchanged": 3}\n')

    monkeypatch.setattr(subprocess, "run", fake_run)
    env = {"PG_CONNECTION_STRING": "postgresql://127.0.0.1:5/x", "DATA_DIR": str(tmp_path)}
    report, seconds = service.index_step(env, cwd=tmp_path, log_path=tmp_path / "logs" / "index.log")
    ((command, kwargs),) = calls
    assert command[1:] == ["-m", "app.engine.ingest"] and kwargs["cwd"] == tmp_path and kwargs["check"] is True and kwargs["env"] == env
    assert report == {"chunks_embedded": 0, "files_unchanged": 3} and seconds >= 0.0
    assert "Indexing report" in (tmp_path / "logs" / "index.log").read_text()


def test_index_step_raises_when_the_command_fails(tmp_path, monkeypatch):
    def fake_run(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        service.index_step({}, cwd=tmp_path, log_path=tmp_path / "index.log")


def test_http_service_script_applies_the_migrations_before_the_api_starts(tmp_path, monkeypatch):
    """The deploy step comes first: the API refuses to start while a schema migration is pending."""
    data = tmp_path / "corpus"
    data.mkdir()
    (data / "a.md").write_text("alpha\n")
    queries = tmp_path / "q.jsonl"
    queries.write_text(json.dumps({"id": "q1", "question": "alpha?"}) + "\n")
    order = []
    monkeypatch.setattr(service.TaskPostgres, "start", lambda self: order.append("postgres") or self)
    monkeypatch.setattr(service.TaskPostgres, "stop", lambda self: None)

    def fake_migrate(env, *, cwd, log_path=None, python=None):
        order.append(("migrate", env["PG_CONNECTION_STRING"].startswith("postgresql://127.0.0.1:"), env["EMBEDDING_DIM"], cwd == ROOT))
        return 0.5

    def fake_api_start(self):
        order.append("api")
        raise RuntimeError("stop here")

    module = load_script()
    monkeypatch.setattr(module, "migrate", fake_migrate)
    monkeypatch.setattr(module.ApiServer, "start", fake_api_start)
    monkeypatch.setattr(sys, "argv", ["revamp_http_service.py", "--container", str(tmp_path / "pg.sif"), "--data", str(data), "--queries", str(queries),
                                      "--out-dir", str(tmp_path / "out"), "--device", "cpu", "--scratch", str(tmp_path / "scratch")])
    with pytest.raises(RuntimeError, match="stop here"):
        module.main()
    assert order == ["postgres", ("migrate", True, "384", True), "api"]
