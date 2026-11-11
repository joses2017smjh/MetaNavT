"""Start-up against a live Postgres: what it writes to the shared table, and two replicas creating the database.

Needs PG_TEST_DSN (the CI parity job's service, or a local instance); skipped otherwise.
Each test uses a vector table (or a database) of its own and removes it at the end.
"""

import os
import threading
import time
import uuid

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("PG_TEST_DSN"), reason="PG_TEST_DSN not set")


def _plain():
    import psycopg2

    conn = psycopg2.connect(os.environ["PG_TEST_DSN"])
    conn.autocommit = True
    return conn


@pytest.fixture
def migrated(monkeypatch):
    """A migrated vector table holding a few rows, and a manager on it (the deploy step has run)."""
    from llama_index.core.schema import TextNode

    from app.database.migrate import drop_vector_table, migrate_all
    from app.database.vector_store_manager import VectorStoreManager
    from app.retrieval.embedders import HashEmbedder

    dsn = os.environ["PG_TEST_DSN"]
    monkeypatch.setenv("PG_CONNECTION_STRING", dsn)
    monkeypatch.setenv("PSYCOPG2_CONNECTION_STRING", dsn)
    monkeypatch.setenv("PGVECTOR_TABLE", "probetest_" + uuid.uuid4().hex[:8])
    monkeypatch.setenv("EMBEDDING_DIM", "256")
    monkeypatch.delenv("LEXICAL_BACKEND", raising=False)
    vsm = VectorStoreManager(conn_string=dsn)
    with vsm.get_connection() as conn:
        migrate_all(conn, vsm.vector_table)
    emb = HashEmbedder(256)
    texts = {"a": "learning rate 3e-4", "b": "run 47 log", "c": "paper draft"}
    vsm.get_vector_store().add([TextNode(id_=k, text=t, embedding=emb.encode([t])[0].tolist(), metadata={"path": f"{k}.txt"})
                                for k, t in texts.items()])
    yield vsm
    with vsm.get_connection() as conn:
        drop_vector_table(conn, vsm.vector_table)
    vsm.close()


def test_get_vector_store_never_shows_a_probe_row_to_a_concurrent_reader(migrated):
    """Every replica start used to add a zero-vector test_node_* row to the shared table and delete it
    again: a reader in between saw it. A row trigger logs every write; a reader polls meanwhile."""
    from app.database.vector_store_manager import VectorStoreManager

    vsm = migrated
    table, log = f'"{vsm.schema_name}"."{vsm.data_table}"', f"probe_log_{uuid.uuid4().hex[:8]}"
    admin = _plain()
    with admin.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table};")
        rows_before = cur.fetchone()[0]
        cur.execute(f"CREATE TABLE {log} (op text, node_id text);")
        cur.execute(f"CREATE FUNCTION {log}_fn() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
                    f"INSERT INTO {log} VALUES (TG_OP, COALESCE(NEW.node_id, OLD.node_id)); RETURN NULL; END $$;")
        cur.execute(f"CREATE TRIGGER {log}_trg AFTER INSERT OR UPDATE OR DELETE ON {table} FOR EACH ROW EXECUTE FUNCTION {log}_fn();")
    seen, stop = [], threading.Event()

    def read():
        reader = _plain()
        try:
            with reader.cursor() as cur:
                while not stop.is_set():
                    cur.execute(f"SELECT count(*) FROM {table} WHERE node_id LIKE 'test_node_%%';")
                    seen.append(cur.fetchone()[0])
        finally:
            reader.close()

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    try:
        for _ in range(5):  # five replica starts
            replica = VectorStoreManager(conn_string=os.environ["PG_TEST_DSN"])
            try:
                replica.get_vector_store()
            finally:
                replica.close()
        time.sleep(0.2)
    finally:
        stop.set()
        thread.join(timeout=10)
        with admin.cursor() as cur:
            cur.execute(f"SELECT op, node_id FROM {log};")
            writes = cur.fetchall()
            cur.execute(f"SELECT count(*) FROM {table};")
            rows_after = cur.fetchone()[0]
            cur.execute(f"DROP TRIGGER IF EXISTS {log}_trg ON {table};")
            cur.execute(f"DROP FUNCTION IF EXISTS {log}_fn();")
            cur.execute(f"DROP TABLE IF EXISTS {log};")
        admin.close()
    assert seen and max(seen) == 0, f"a reader saw a probe row: {max(seen)}"
    assert writes == [], writes  # nothing was inserted, updated or deleted, not even briefly
    assert rows_after == rows_before


def test_the_probe_refuses_a_missing_table_and_keeps_no_store(monkeypatch):
    """Before the deploy step there is no table: the probe says so (it no longer creates one by writing a
    row), and the manager keeps no store, so the next start-up attempt probes again."""
    from app.database.vector_store_manager import VectorStoreManager

    dsn = os.environ["PG_TEST_DSN"]
    monkeypatch.setenv("PG_CONNECTION_STRING", dsn)
    monkeypatch.setenv("PGVECTOR_TABLE", "probemissing_" + uuid.uuid4().hex[:8])
    monkeypatch.setenv("EMBEDDING_DIM", "256")
    vsm = VectorStoreManager(conn_string=dsn)
    try:
        with pytest.raises(RuntimeError, match="does not exist: apply the schema migrations"):
            vsm.get_vector_store()
        assert vsm.vector_store is None
        with vsm.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass(%s) IS NULL;", (f'"{vsm.schema_name}"."{vsm.data_table}"',))
                assert cur.fetchone()[0]  # still no table: the probe wrote nothing
    finally:
        vsm.close()


def _fresh_database(monkeypatch, prefix):
    """(name, URL) of a database that does not exist yet, set as DB_NAME and PG_CONNECTION_STRING: the
    admin connection is PG_CONNECTION_STRING with /postgres, the name it creates is DB_NAME, and the pool
    connects to the URL passed as conn_string."""
    from urllib.parse import urlparse, urlunparse

    name = prefix + uuid.uuid4().hex[:8]
    fresh = urlunparse(urlparse(os.environ["PG_TEST_DSN"])._replace(path="/" + name))
    monkeypatch.setenv("PG_CONNECTION_STRING", fresh)
    monkeypatch.setenv("DB_NAME", name)
    monkeypatch.setenv("PGVECTOR_TABLE", "createtest")
    monkeypatch.setenv("EMBEDDING_DIM", "256")
    monkeypatch.delenv("LEXICAL_BACKEND", raising=False)
    return name, fresh


def _drop_database(name):
    admin = _plain()
    with admin.cursor() as cur:
        cur.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE);')
    admin.close()


def test_a_manager_creates_its_missing_database(monkeypatch):
    """The create-if-missing branch had never run: `with psycopg2.connect(...)` opens a transaction even in
    autocommit mode since psycopg2 2.9, and CREATE DATABASE cannot run inside one (ActiveSqlTransaction).
    Compose, CI and the local instances all create the database before the API starts."""
    from app.database.vector_store_manager import VectorStoreManager

    name, fresh = _fresh_database(monkeypatch, "createtest_")
    try:
        manager = VectorStoreManager(conn_string=fresh)
        try:
            with manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT current_database();")
                    assert cur.fetchone()[0] == name
        finally:
            manager.close()
    finally:
        _drop_database(name)


@pytest.mark.parametrize("interleaving", ["together", "one_after_the_other"])
def test_two_replicas_creating_a_fresh_database_at_once_both_start(monkeypatch, interleaving):
    """C37: DatabaseManager (the base of the API's VectorStoreManager and the deploy step's manager) checks
    pg_database and then runs CREATE DATABASE. Two replicas that start together can both find nothing; the
    loser's CREATE used to raise and that replica failed to start. A barrier in the admin connection's cursor
    holds each CREATE until both existence checks have found nothing, so the race happens every time. Sent
    together, both CREATEs pass Postgres's own name check and the loser fails on pg_database's unique index
    (UniqueViolation); sent one after the other, the second finds the first's database (DuplicateDatabase)."""
    import psycopg2
    import psycopg2.extensions

    from app.database.vector_store_manager import VectorStoreManager

    name, fresh = _fresh_database(monkeypatch, "racetest_")
    barrier, first_done, creates, real_connect = threading.Barrier(2, timeout=30), threading.Event(), [], psycopg2.connect

    class RacingCursor(psycopg2.extensions.cursor):
        def execute(self, query, vars=None):
            text = query.as_string(self) if hasattr(query, "as_string") else str(query)
            if not text.lstrip().upper().startswith("CREATE DATABASE"):
                return super().execute(query, vars)
            creates.append(text)
            turn = barrier.wait()  # both existence checks have found nothing: the two CREATEs race
            if interleaving == "together":
                return super().execute(query, vars)
            if turn == 1:  # the second CREATE waits until the first one has finished
                first_done.wait(30)
                return super().execute(query, vars)
            try:
                return super().execute(query, vars)
            finally:
                first_done.set()

    def connect(*args, **kwargs):
        if kwargs.get("database") == "postgres":  # DatabaseManager's admin connection
            kwargs["cursor_factory"] = RacingCursor
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(psycopg2, "connect", connect)
    started: dict = {}

    def start(replica):
        try:
            manager = VectorStoreManager(conn_string=fresh)
            try:
                with manager.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT current_database();")
                        started[replica] = cur.fetchone()[0]
            finally:
                manager.close()
        except Exception as exc:  # noqa: BLE001 - reported by the assertion below
            started[replica] = f"{type(exc).__name__}: {exc}".strip()

    threads = [threading.Thread(target=start, args=(replica,)) for replica in (0, 1)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
    finally:
        _drop_database(name)
    assert len(creates) == 2, creates  # both replicas found no database and sent CREATE DATABASE: a real race
    assert started == {0: name, 1: name}, started
