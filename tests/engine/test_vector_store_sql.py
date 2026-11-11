"""The Python side of the served hybrid statement, without a database.

What VectorStoreManager sends along with app/database/sql/hybrid_*.sql: the
per-statement settings, the lexical list's query text and which lexical backend
runs (LEXICAL_BACKEND, never detected). The live behaviour (a forced HNSW scan returning the full
depth) is checked against Postgres in tests/database/test_dense_parity.py. The
committed plans in doc/sql/ must have been made with the current settings.
"""

import hashlib
import re
from contextlib import contextmanager
from pathlib import Path

import pytest

from app.database.vector_store_manager import HNSW_EF_SEARCH_DEFAULT, HNSW_EF_SEARCH_MAX, VectorStoreManager


def manager(pgvector=(0, 8, 6)):
    """A manager that never connects: only the attributes these methods read."""
    vsm = VectorStoreManager.__new__(VectorStoreManager)
    vsm._pgvector_version = pgvector
    return vsm


class _Cursor:
    def __init__(self):
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))


@pytest.mark.parametrize(
    "k, ef_search",
    [(50, 100), (10, HNSW_EF_SEARCH_DEFAULT), (20, HNSW_EF_SEARCH_DEFAULT), (21, 42), (500, HNSW_EF_SEARCH_MAX), (5000, HNSW_EF_SEARCH_MAX)],
)
def test_ef_search_is_twice_the_depth_within_pgvectors_bounds(k, ef_search):
    assert manager().search_settings(k) == {"hnsw.ef_search": ef_search, "hnsw.iterative_scan": "strict_order"}


def test_settings_follow_the_installed_pgvector():
    assert manager((0, 7, 4)).search_settings(50) == {"hnsw.ef_search": 100}  # iterative scans arrived in 0.8.0
    assert manager((0, 4, 4)).search_settings(50) == {}  # no HNSW before 0.5.0
    assert manager(()).search_settings(50) == {}  # extension not installed


def test_settings_are_one_statement_for_the_transaction_only():
    statement, params = manager().settings_sql(50, statement_timeout_ms=1234)
    assert statement == (
        "SELECT set_config(%(setting_0)s, %(setting_value_0)s, true), set_config(%(setting_1)s, %(setting_value_1)s, true), "
        "set_config(%(setting_2)s, %(setting_value_2)s, true);\n"
    )
    assert params == {
        "setting_0": "hnsw.ef_search", "setting_value_0": "100",
        "setting_1": "hnsw.iterative_scan", "setting_value_1": "strict_order",
        "setting_2": "statement_timeout", "setting_value_2": "1234",
    }
    assert manager().settings_sql(50, statement_timeout_ms=0)[1]["setting_value_2"] == "1"  # 0 would mean no limit
    assert manager(()).settings_sql(50) == ("", {})

    cur = _Cursor()
    assert manager().apply_search_settings(cur, 50) == {"hnsw.ef_search": 100, "hnsw.iterative_scan": "strict_order"}
    assert cur.executed == [manager().settings_sql(50)]  # one statement


# ---------------------------------------------------------------- committed plans (doc/sql, `make explain-sql*`)

PLANS = Path(__file__).resolve().parents[2] / "doc" / "sql"


def plan_settings(text: str) -> tuple[tuple[int, ...], int, str]:
    """(pgvector version, k, the settings line) from a committed plan's header."""
    version = tuple(int(p) for p in re.search(r"^-- pgvector (\d+)\.(\d+)\.(\d+);", text, re.M).groups())
    k = int(re.search(r"^-- query: .*, k=(\d+),", text, re.M).group(1))
    line = re.search(r"^-- per-statement settings \(VectorStoreManager.search_settings\): (.*)$", text, re.M).group(1)
    return version, k, line


@pytest.mark.parametrize("name", ["hybrid_explain.txt", "hybrid_explain_synthetic_10k.txt"])
def test_committed_plans_ran_with_the_current_statement_and_settings(name):
    text = (PLANS / name).read_text()
    regenerate = f"regenerate doc/sql/{name} (make explain-sql / explain-sql-synthetic)"
    statement = re.search(r"^-- EXPLAIN \(ANALYZE, BUFFERS\) of (\S+)$", text, re.M).group(1)
    digest = re.search(r"^-- statement sha256 ([0-9a-f]{64})", text, re.M).group(1)
    assert digest == hashlib.sha256((PLANS.parents[1] / statement).read_bytes()).hexdigest(), regenerate
    version, k, line = plan_settings(text)
    assert line == str(manager(version).search_settings(k)), regenerate


def test_synthetic_plan_reads_the_dense_list_from_the_hnsw_index_and_the_lexical_list_from_gin():
    text = (PLANS / "hybrid_explain_synthetic_10k.txt").read_text()
    rows = int(re.search(r"\((\d+) SYNTHETIC rows", text).group(1))
    assert rows >= 10_000
    assert re.search(r"Index Scan using \S+_embedding_hnsw_m\d+_efc\d+ on ", text)
    assert re.search(r"Bitmap Index Scan on \S+_text_tsv_english_gin", text)


@pytest.mark.parametrize("name", ["hybrid_explain.txt", "hybrid_explain_synthetic_10k.txt"])
def test_no_committed_plan_computes_to_tsvector_per_row(name):
    """The lexical list reads the stored text_tsv column (schema migration vector/0006)."""
    plan = "\n".join(line for line in (PLANS / name).read_text().splitlines() if not line.startswith("--"))
    assert "to_tsvector" not in plan and "text_tsv @@" in plan


# ---------------------------------------------------------------- the two statements

SQL_DIR = Path(__file__).resolve().parents[2] / "app" / "database" / "sql"


def dense_cte(name: str) -> str:
    text = (SQL_DIR / name).read_text()
    return text[text.index("dense AS (") : text.index(",\nlexical AS (")]


def test_both_statements_share_the_dense_cte():
    """No CI job runs the ParadeDB variant; its dense list is at least the tested one, verbatim."""
    assert dense_cte("hybrid_paradedb.sql") == dense_cte("hybrid_tsrank.sql")
    assert "LIMIT 2 * %(k)s" in dense_cte("hybrid_tsrank.sql")  # the rows search_settings sizes ef_search for


# ---------------------------------------------------------------- lexical_query (the websearch text of the lexical list)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("Current learning-rate for run_047!", "current or learning or rate or for or run_047"),
        ("run RUN run_047 run", "run or run_047"),  # lowercased, each term once
        ("Café crème brûlée", "café or crème or brûlée"),  # an accent does not split a word
        ("Café au lait", "café or au or lait"),  # decomposed input is composed first (NFC)
        ("Zürich naïve résumé 東京", "zürich or naïve or résumé or 東京"),
        # Combining marks stay inside the word, as Postgres's parser keeps them: Devanagari vowel
        # signs and virama, Arabic short vowels, a diacritic with no precomposed form.
        ("हिन्दी भाषा", "हिन्दी or भाषा"),
        ("كَتَبَ الوَلَدُ", "كَتَبَ or الوَلَدُ"),
        ("n̈ame", "n̈ame"),
        ("I ❤️ cats", "i or cats"),  # a mark with no word before it (an emoji's variation selector) is no term
        ("cats or dogs", "cats or dogs"),  # websearch's operator word is not a term
        ("fusion_off_resnet OR fusion_on_resnet", "fusion_off_resnet or fusion_on_resnet"),
        ("or", ""),
        ("Or OR oR", ""),
        ('"quoted" -minus', "quoted or minus"),  # quotes and a leading "-" are operators too; never kept
        ("", ""),
    ],
)
def test_lexical_query_terms(query, expected):
    assert VectorStoreManager.lexical_query(query) == expected


# ---------------------------------------------------------------- the lexical backend (LEXICAL_BACKEND)


class _Connection:
    """Records every statement; pg_extension lookups answer `pg_search`; ParadeDB's operator fails without it."""

    def __init__(self, pg_search: bool):
        self.pg_search = pg_search
        self.statements: list[str] = []

    def cursor(self):
        return _RecordingCursor(self)


class _RecordingCursor:
    def __init__(self, conn):
        self.conn, self.row = conn, None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, statement, params=None):
        text = statement if isinstance(statement, str) else repr(statement)
        self.conn.statements.append(text)
        if "@@@" in text and not self.conn.pg_search:
            raise RuntimeError("operator does not exist: character varying @@@ unknown")
        self.row = (1,) if "pg_search" in text and self.conn.pg_search else None

    def fetchone(self):
        return self.row

    def fetchall(self):
        return []


def lexical_manager(monkeypatch, setting, pg_search):
    """(manager, connection) with LEXICAL_BACKEND = setting (None: unset) on a database with or without pg_search."""
    if setting is None:
        monkeypatch.delenv("LEXICAL_BACKEND", raising=False)
    else:
        monkeypatch.setenv("LEXICAL_BACKEND", setting)
    vsm = manager()
    vsm.schema_name, vsm.table_name, vsm.data_table = "public", "t", "data_t"
    conn = _Connection(pg_search)

    @contextmanager
    def get_connection():
        yield conn

    vsm.get_connection = get_connection
    return vsm, conn


def test_ts_rank_cd_is_the_default_even_where_pg_search_is_installed(monkeypatch):
    vsm, conn = lexical_manager(monkeypatch, None, pg_search=True)
    assert vsm.hybrid_backend == "ts_rank_cd"
    assert "ts_rank_cd(" in vsm.hybrid_sql() and "paradedb.score(" not in vsm.hybrid_sql()
    assert not any("pg_search" in s for s in conn.statements)  # nothing was detected: the setting decides


def test_paradedb_runs_only_when_configured_and_installed(monkeypatch):
    vsm, _ = lexical_manager(monkeypatch, "paradedb", pg_search=True)
    assert vsm.hybrid_backend == "paradedb" and "paradedb.score(" in vsm.hybrid_sql()

    vsm, _ = lexical_manager(monkeypatch, "ParadeDB ", pg_search=False)
    with pytest.raises(RuntimeError, match="pg_search"):
        vsm.hybrid_backend  # no fallback to ts_rank_cd: a configured backend that cannot run is an error


def test_an_unknown_lexical_backend_is_an_error(monkeypatch):
    from app.database.vector_store_manager import lexical_backend_setting

    monkeypatch.setenv("LEXICAL_BACKEND", "bm25")
    with pytest.raises(ValueError, match="LEXICAL_BACKEND"):
        lexical_backend_setting()
    monkeypatch.setenv("LEXICAL_BACKEND", "")
    assert lexical_backend_setting() == "ts_rank_cd"


def test_search_bm25_tries_paradedb_only_when_configured(monkeypatch):
    vsm, conn = lexical_manager(monkeypatch, None, pg_search=True)
    vsm.search_bm25("learning rate", limit=5)
    assert conn.statements and not any("paradedb" in s or "@@@" in s for s in conn.statements), conn.statements
    assert any("ts_rank_cd" in s for s in conn.statements)

    vsm, conn = lexical_manager(monkeypatch, "paradedb", pg_search=True)
    vsm.search_bm25("learning rate", limit=5)
    assert "SET search_path TO paradedb, public;" in conn.statements


# ---------------------------------------------------------------- start-up issues no DDL (C35: the BM25 index is never dropped)


class _RecordingConnection:
    """Every statement any cursor of this connection executes, in order; pg_search reads as installed and
    the read-only probe finds the table with 8 dimensions and every privilege."""

    def __init__(self):
        self.statements: list[str] = []

    def cursor(self):
        connection = self

        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, sql, params=None):
                connection.statements.append(" ".join(str(sql).split()))

            def fetchone(self):
                return (8, True, True, True) if "pg_attribute" in connection.statements[-1] else (1,)

            def fetchall(self):
                return []

        return Cursor()

    def commit(self):
        pass

    def rollback(self):
        pass


@pytest.mark.parametrize("setting", [None, "ts_rank_cd", "paradedb"])
def test_get_vector_store_issues_no_drop_index_and_no_ddl(monkeypatch, setting):
    """A recording cursor sees every statement the manager's constructor and get_vector_store send. With
    LEXICAL_BACKEND=paradedb they used to drop the BM25 index and build it again at every start; now the
    pg_search objects are schema migrations (migrations/paradedb/) and start-up sends no DDL at all."""
    from app.database import db_base_manager, vector_store_manager as vsm_module

    monkeypatch.setenv("PG_CONNECTION_STRING", "postgresql://u@127.0.0.1:1/db")
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    if setting is None:
        monkeypatch.delenv("LEXICAL_BACKEND", raising=False)
    else:
        monkeypatch.setenv("LEXICAL_BACKEND", setting)
    monkeypatch.setattr(db_base_manager.DatabaseManager, "_ensure_database_exists", lambda self: None)
    monkeypatch.setattr(db_base_manager.DatabaseManager, "_initialize_connection_pool", lambda self, *a, **k: None)
    monkeypatch.setattr(vsm_module.VectorStoreManager, "_detect_vector_dimensions", lambda self: None)
    connection = _RecordingConnection()

    @contextmanager
    def recording(self):
        yield connection

    class _Store:
        writes: list = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def add(self, nodes):
            self.writes.append(("add", [n.node_id for n in nodes]))
            return [n.node_id for n in nodes]

        def delete(self, ref_doc_id):
            self.writes.append(("delete", ref_doc_id))

    monkeypatch.setattr(vsm_module.VectorStoreManager, "get_connection", recording)
    monkeypatch.setattr(vsm_module, "PGVectorStore", _Store)
    store = vsm_module.VectorStoreManager(conn_string="postgresql://u@127.0.0.1:1/db").get_vector_store()

    assert store.kwargs["perform_setup"] is False  # nor does LlamaIndex create anything
    assert not [s for s in connection.statements if "DROP INDEX" in s.upper()], connection.statements
    ddl = re.compile(r"^\s*(CREATE|DROP|ALTER)\b", re.I)
    assert not [s for s in connection.statements if ddl.match(s)], connection.statements
    # C36: the start-up probe only reads; nothing is written to the shared table, through the store or the pool.
    assert store.writes == [] and not [s for s in connection.statements if re.match(r"^\s*(INSERT|UPDATE|DELETE)\b", s, re.I)]
