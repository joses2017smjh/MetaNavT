"""The vector store manager must never write the database password to a log.

No database: the pool, the extension checks and PGVectorStore are replaced by
stubs so the test exercises exactly the code that used to log both SQLAlchemy
connection strings (password included) at INFO on every process start.
"""

import logging
from contextlib import contextmanager

import pytest

SECRET = "s3cr3t-Pa55w0rd-DO-NOT-LOG"
DSN = f"postgresql://metanavit_user:{SECRET}@db.internal.example:6543/metanavit_logs"


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(self.format(record))


class _FakeStore:
    """Stands in for PGVectorStore: records what it was built with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _ProbeConnection:
    """Answers the read-only start-up probe: the table exists, 8 dimensions, every privilege."""

    def cursor(self):
        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, sql, params=None):
                self.row = (8, True, True, True) if "pg_attribute" in sql else (True,)

            def fetchone(self):
                return self.row

        return Cursor()


@pytest.fixture
def captured(monkeypatch):
    from app.database import db_base_manager, vector_store_manager as vsm_module

    monkeypatch.setenv("PG_CONNECTION_STRING", DSN)
    monkeypatch.setenv("PSYCOPG2_CONNECTION_STRING", DSN)
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    monkeypatch.setattr(db_base_manager.DatabaseManager, "_ensure_database_exists", lambda self: None)
    monkeypatch.setattr(db_base_manager.DatabaseManager, "_initialize_connection_pool", lambda self, *a, **k: None)
    monkeypatch.setattr(vsm_module.VectorStoreManager, "_detect_vector_dimensions", lambda self: None)

    @contextmanager
    def no_connection(self):
        yield _ProbeConnection()

    monkeypatch.setattr(vsm_module.VectorStoreManager, "get_connection", no_connection)
    monkeypatch.setattr(vsm_module, "PGVectorStore", _FakeStore)

    handler = _Capture()
    handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
    loggers = [logging.getLogger(), logging.getLogger("uvicorn"), logging.getLogger(db_base_manager.__name__)]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.addHandler(handler)
        lg.setLevel(logging.DEBUG)
    try:
        yield handler, vsm_module
    finally:
        for lg, level in zip(loggers, levels):
            lg.removeHandler(handler)
            lg.setLevel(level)


def test_connection_password_never_reaches_the_log(captured):
    handler, vsm_module = captured
    manager = vsm_module.VectorStoreManager(conn_string=DSN)
    store = manager.get_vector_store()

    # The store still receives the real connection strings: only the log changed.
    assert SECRET in store.kwargs["connection_string"] and store.kwargs["connection_string"].startswith("postgresql+psycopg2://")
    assert SECRET in store.kwargs["async_connection_string"] and store.kwargs["async_connection_string"].startswith("postgresql+asyncpg://")

    assert handler.lines, "the manager logged nothing: the capture is not attached"
    leaked = [line for line in handler.lines if SECRET in line]
    assert not leaked, leaked
    assert not [line for line in handler.lines if "metanavit_user:" in line]


def test_the_log_still_says_where_the_store_points(captured):
    handler, vsm_module = captured
    vsm_module.VectorStoreManager(conn_string=DSN).get_vector_store()
    text = "\n".join(handler.lines)
    assert "db.internal.example" in text and "metanavit_logs" in text


def test_describe_dsn_drops_credentials():
    from app.database.vector_store_manager import describe_dsn

    assert describe_dsn(DSN) == "host=db.internal.example port=6543 database=metanavit_logs"
    assert SECRET not in describe_dsn(f"dbname=x user=u password={SECRET} host=h port=5")
    assert describe_dsn("") == "host=None port=None database=None"
