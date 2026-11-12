"""
Vector Store Manager Module

Manages PostgreSQL vector operations using pgvector extension for document embeddings.
Provides connection pooling and vector store management through a singleton pattern.

Features:
    - Vector store initialization and management
    - Connection pooling via parent DatabaseManager
    - Both sync (psycopg2) and async (asyncpg) support
    - Dimension detection; no DDL: the data table, the ingest manifest, their indexes, the
      vector extension and (LEXICAL_BACKEND=paradedb) pg_search and the BM25 index come from
      the schema migrations (app/database/migrate.py, `make migrate`), which
      check_migrations() requires before the API starts

Configuration (via environment variables):
    PGVECTOR_SCHEMA: Schema name for vector tables (default: "public")
    PGVECTOR_TABLE: Table name for embeddings (default: "llamaindex_embedding")
    EMBEDDING_DIM: Embedding dimension size (default: 1024)
    PG_CONNECTION_STRING: PostgreSQL connection string

Dependencies:
    - PostgreSQL with pgvector extension
    - llama-index for vector store operations
    - psycopg2 for database connections
    - asyncpg for async operations
"""
import hashlib
import os
import logging
import re
from contextlib import contextmanager
from pathlib import Path
from llama_index.vector_stores.postgres import PGVectorStore
from urllib.parse import urlparse
from psycopg2 import sql
from dotenv import load_dotenv
import unicodedata
import time
import psycopg2
import psycopg2.errors

from .db_base_manager import DatabaseManager
from .migrate import VectorTable, require_current

load_dotenv()
logger = logging.getLogger("uvicorn")

# hnsw.ef_search bounds for the served statement (VectorStoreManager.search_settings):
# pgvector's own default, kept as the floor, and the largest value pgvector accepts.
HNSW_EF_SEARCH_DEFAULT = 40
HNSW_EF_SEARCH_MAX = 1000
# Words websearch_to_tsquery reads as an operator rather than a term (lowercase; it ignores case).
WEBSEARCH_OPERATOR_WORDS = frozenset({"or"})
# The lexical list's backend, chosen by LEXICAL_BACKEND and never detected. ts_rank_cd (Postgres
# full text, the default) is what CI and every served-path results file ran; paradedb (ParadeDB
# pg_search BM25, sql/hybrid_paradedb.sql) has not been run by any test or results file.
LEXICAL_BACKENDS = ("ts_rank_cd", "paradedb")


class IngestLockTimeout(TimeoutError):
    """Another process held the ingest lock longer than the wait allows; this run indexed nothing."""


def lexical_backend_setting() -> str:
    """LEXICAL_BACKEND, lowercased; unset or empty is ts_rank_cd, an unknown value is a ValueError."""
    value = os.getenv("LEXICAL_BACKEND", "").strip().lower() or "ts_rank_cd"
    if value not in LEXICAL_BACKENDS:
        raise ValueError(f"LEXICAL_BACKEND must be one of {', '.join(LEXICAL_BACKENDS)}, got {value!r}")
    return value


def word_runs(text: str):
    """Runs of word characters (letters and digits of any script, "_": what the regex \\w matches)
    and of the combining marks that follow them, in order.

    \\w alone splits a word at every mark that is not a letter: Devanagari vowel signs and
    virama, Arabic short vowels, a diacritic with no precomposed form. Postgres's text search
    parser keeps such a word whole, so a term split there could never match it. A mark only
    continues a run: one with no word before it (an emoji's variation selector) starts none.
    """
    run: list[str] = []
    for ch in text:
        if ch.isalnum() or ch == "_" or (run and unicodedata.category(ch).startswith("M")):
            run.append(ch)
        elif run:
            yield "".join(run)
            run = []
    if run:
        yield "".join(run)


def describe_dsn(dsn: str | None) -> str:
    """Host, port and database name of a connection string, for log lines.

    Never returns the user or the password. Accepts the URL form
    (postgresql://user:pw@host:5432/db) and the libpq keyword form
    (dbname=db user=u password=pw host=h port=5432). A string that cannot be
    parsed yields "None" fields: the parser's own error message quotes the
    whole string, so it is deliberately not logged.
    """
    host = port = database = None
    text = dsn or ""
    try:
        fields = psycopg2.extensions.parse_dsn(text)
        host, port, database = fields.get("host"), fields.get("port"), fields.get("dbname")
    except psycopg2.Error:
        if "://" in text:  # SQLAlchemy-style scheme such as postgresql+psycopg2://
            try:
                parsed = urlparse(text)
                host, port, database = parsed.hostname, parsed.port, parsed.path.lstrip("/") or None
            except ValueError:
                pass
    return f"host={host} port={port} database={database}"


class VectorStoreManager(DatabaseManager):
    """Manages vector store operations using pgvector (and pg_search only with LEXICAL_BACKEND=paradedb)."""
    
    def __init__(self, conn_string: str = None):
        """Initialize vector store manager with connection pooling."""
        # Initialize parent DatabaseManager
        super().__init__(conn_string)
        
        self.schema_name = os.getenv("PGVECTOR_SCHEMA", "public")
        self.table_name = os.getenv("PGVECTOR_TABLE", "llamaindex_embedding")
        # PGVectorStore stores rows in "data_<table_name>". Every raw SQL path here
        # (BM25 / full-text search, dimension detection, row counts) targets that
        # table. Before M0 they targeted the bare table_name, which PGVectorStore
        # never writes to, so production BM25 searched an empty table.
        self.data_table = f"data_{self.table_name}"
        # The configured lexical backend (LEXICAL_BACKEND); a bad value fails here, at start-up.
        self.lexical_backend = lexical_backend_setting()
        # Which lexical backend answered the last search_bm25 call:
        # paradedb | ts_rank_cd | ilike | none. Surfaced by /health and /api/retrieve.
        self.last_bm25_backend: str | None = None
        
        # First, detect the actual vector dimensions in the database
        # before setting self.embed_dim
        detected_dim = self._detect_vector_dimensions()
        if detected_dim is not None:
            logger.info(f"Detected existing vector dimension in database: {detected_dim}")
            self.embed_dim = detected_dim
        else:
            # If no table exists yet, use the environment variable
            self.embed_dim = int(os.getenv("EMBEDDING_DIM", 1024))  # BAAI/bge-large-en-v1.5, as in .env.example
            logger.info(f"Using vector dimension from environment: {self.embed_dim}")
        
        self.vector_store = None
        # No DDL here or anywhere in this class: the vector extension, data_<table_name> (in the
        # shape PGVectorStore gives it), the ingest manifest, the GIN / HNSW / path indexes and,
        # with LEXICAL_BACKEND=paradedb, pg_search and its BM25 index all come from the schema
        # migrations (app/database/migrations/vector/ and paradedb/), applied by the deploy step.

    @property
    def vector_table(self) -> VectorTable:
        """This manager's table as the schema migrations name it (app/database/migrate.py)."""
        return VectorTable(schema=self.schema_name, table=self.table_name, dim=int(self.embed_dim), lexical_backend=self.lexical_backend)

    def check_migrations(self) -> None:
        """Raise MigrationsPending unless every migration this table and backend need is applied. Reads only."""
        with self.get_connection() as conn:
            require_current(conn, self.vector_table)

    def probe_data_table(self) -> None:
        """Check, writing nothing, that the data table exists with self.embed_dim dimensions and that
        this role may SELECT, INSERT and DELETE on it; RuntimeError with the reason otherwise.

        When the table already existed as this manager was built, self.embed_dim is the table's own
        dimension (_detect_vector_dimensions, which EMBEDDING_DIM does not override), so the dimension
        check only bites for a table created afterwards, e.g. by the deploy step during start-up retries.
        """
        qualified = f'"{self.schema_name}"."{self.data_table}"'
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass(%s) IS NOT NULL;", (qualified,))
                if not cur.fetchone()[0]:
                    raise RuntimeError(
                        f"data table {self.schema_name}.{self.data_table} does not exist: "
                        "apply the schema migrations (python -m app.database.migrate)"
                    )
                cur.execute(
                    "SELECT a.atttypmod, has_table_privilege(a.attrelid, 'SELECT'), "
                    "has_table_privilege(a.attrelid, 'INSERT'), has_table_privilege(a.attrelid, 'DELETE') "
                    "FROM pg_attribute a WHERE a.attrelid = %s::regclass AND a.attname = 'embedding' AND NOT a.attisdropped;",
                    (qualified,),
                )
                row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"data table {self.schema_name}.{self.data_table} has no embedding column")
        dim, may_select, may_insert, may_delete = row
        if int(dim) != int(self.embed_dim):
            raise RuntimeError(
                f"data table {self.schema_name}.{self.data_table} stores {dim}-dimensional embeddings, "
                f"this process uses {self.embed_dim} (EMBEDDING_DIM)"
            )
        missing = [name for name, granted in (("SELECT", may_select), ("INSERT", may_insert), ("DELETE", may_delete)) if not granted]
        if missing:
            raise RuntimeError(f"this database role lacks {', '.join(missing)} on {self.schema_name}.{self.data_table}")
        logger.info(f"Probe: {self.schema_name}.{self.data_table} readable and writable, {dim} dimensions (read-only check)")

    def get_vector_store(self) -> PGVectorStore:
        """Get or create PGVectorStore instance."""
        if self.vector_store is None:
            conn_string_env = os.getenv("PG_CONNECTION_STRING")
            if not conn_string_env:
                raise ValueError("PG_CONNECTION_STRING environment variable not set")

            original_scheme = urlparse(conn_string_env).scheme + "://"
            psycopg2_conn_str = conn_string_env.replace(
                original_scheme, "postgresql+psycopg2://"
            )
            asyncpg_conn_str = conn_string_env.replace(
                original_scheme, "postgresql+asyncpg://"
            )
            # Host and database only. The connection strings carry the password and
            # must never be formatted into a log line (tests/engine/test_log_redaction.py).
            logger.info(f"PGVectorStore will connect to {describe_dsn(conn_string_env)} (psycopg2 and asyncpg drivers)")
            
            # Log the database name expected by DatabaseManager's pool
            # Assuming self.DATABASE_NAME is set in DatabaseManager's __init__
            # from the same PG_CONNECTION_STRING
            if hasattr(self, 'DATABASE_NAME'):
                logger.info(f"DatabaseManager's pool is configured for database: '{self.DATABASE_NAME}' (derived from PG_CONNECTION_STRING)")
            else: # Fallback if DATABASE_NAME isn't on self for some reason
                parsed_main_conn = urlparse(self.conn_string if self.conn_string else conn_string_env)
                logger.info(f"DatabaseManager's pool is configured for database: '{parsed_main_conn.path.lstrip('/')}' (derived from PG_CONNECTION_STRING)")


            logger.info(f"Attempting to initialize PGVectorStore for table '{self.schema_name}.{self.data_table}' (created by the schema migrations)...")
            store = PGVectorStore(
                connection_string=psycopg2_conn_str,
                async_connection_string=asyncpg_conn_str,
                schema_name=self.schema_name,
                table_name=self.table_name,
                embed_dim=self.embed_dim,
                perform_setup=False,  # no CREATE SCHEMA / EXTENSION / TABLE from LlamaIndex: vector/0001 owns them
            )
            logger.info(f"PGVectorStore Python object initialized for table '{self.schema_name}.{self.data_table}'")

            # Read-only probe: the table the migrations built is there, with the embedding dimension
            # this manager uses (see probe_data_table), and this role may read, insert and delete its rows. It used to add a
            # zero-vector test_node_* row to the shared table and delete it again on every start,
            # which every concurrent reader could see in between. A failed probe keeps no store,
            # so the next call (a start-up retry) probes again.
            self.probe_data_table()
            self.vector_store = store
            # The ParadeDB BM25 index (LEXICAL_BACKEND=paradedb only) is migrations/paradedb/0002,
            # built once; it used to be dropped and built again here at every start.

        return self.vector_store

    def search_bm25(self, query: str, limit: int = 5) -> list[dict]:
        """
        Perform BM25 search using the 'text' column (the python retrieval mode).

        With LEXICAL_BACKEND=paradedb, ParadeDB's BM25 first; otherwise, or if that fails
        or finds nothing, Postgres full text (ts_rank_cd), then ILIKE. last_bm25_backend
        names the one that answered.
        """
        if not query:
            raise ValueError("Search query cannot be empty")

        results = []
        
        # ParadeDB BM25, two query formats: only for LEXICAL_BACKEND=paradedb (never detected).
        if self.hybrid_backend == "paradedb":
            try:
                table_identifier = sql.Identifier(self.schema_name, self.data_table)
            
                # Format 1: Standard BM25 search with @@@ operator
                search_sql = sql.SQL("""
                    SELECT node_id, text, metadata_, paradedb.score(node_id) AS score
                    FROM {table}
                    WHERE text @@@ {query}
                    ORDER BY score DESC
                    LIMIT %s;
                """).format(
                    table=table_identifier,
                    query=sql.Literal(query)
                )
            
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        # Set search_path
                        cur.execute("SET search_path TO paradedb, public;")
                    
                        # Execute search and log the actual SQL
                        logger.debug(f"BM25 search query: {search_sql.as_string(conn)}")
                        cur.execute(search_sql, (limit,))
                        results = cur.fetchall()
            
                if results:
                    logger.info(f"BM25 search successful, found {len(results)} results")
                    self.last_bm25_backend = "paradedb"
                    return self._bm25_rows(results)
            
                # Format 2: Try more flexible tokenized search
                tokenized_query = " OR ".join(query.split())
                search_sql2 = sql.SQL("""
                    SELECT node_id, text, metadata_, paradedb.score(node_id) AS score
                    FROM {table}
                    WHERE text @@@ {query}
                    ORDER BY score DESC
                    LIMIT %s;
                """).format(
                    table=table_identifier,
                    query=sql.Literal(tokenized_query)
                )
            
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        # Set search_path
                        cur.execute("SET search_path TO paradedb, public;")
                    
                        # Execute tokenized search
                        logger.debug(f"BM25 tokenized search query: {search_sql2.as_string(conn)}")
                        cur.execute(search_sql2, (limit,))
                        results = cur.fetchall()
            
                if results:
                    logger.info(f"BM25 tokenized search successful, found {len(results)} results")
                    self.last_bm25_backend = "paradedb"
                    return self._bm25_rows(results)
            
            except Exception as e:
                logger.warning(f"BM25 search failed: {e}")
                # Fall back to basic search
        
        # If BM25 search failed or returned no results, try full-text search fallback
        try:
            # More powerful full-text search as fallback
            fallback_sql = sql.SQL("""
                SELECT node_id, text, metadata_, ts_rank_cd(text_tsv, plainto_tsquery('english', %s)) AS score
                FROM {table}
                WHERE text_tsv @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s;
            """).format(table=sql.Identifier(self.schema_name, self.data_table))
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(fallback_sql, (query, query, limit))
                    results = cur.fetchall()
            
            if results:
                logger.info(f"Fallback full-text search successful, found {len(results)} results")
                self.last_bm25_backend = "ts_rank_cd"
                return self._bm25_rows(results)
            
            # Last resort: ILIKE search
            basic_sql = sql.SQL("""
                SELECT node_id, text, metadata_, 1.0 AS score
                FROM {table}
                WHERE text ILIKE %s
                LIMIT %s;
            """).format(table=sql.Identifier(self.schema_name, self.data_table))
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(basic_sql, (f'%{query}%', limit))
                    results = cur.fetchall()
            
            if results:
                logger.info(f"Basic ILIKE search successful, found {len(results)} results")
                self.last_bm25_backend = "ilike"
                return self._bm25_rows(results)
            else:
                logger.info("No results found in any search")
                self.last_bm25_backend = "none"
                return []
        except Exception as e:
            logger.error(f"All search methods failed: {e}")
            return []

    # ------------------------------------------------------------------ hybrid (one query)
    SQL_DIR = Path(__file__).resolve().parent / "sql"
    _sql_cache: dict[str, str] = {}

    @property
    def hybrid_backend(self) -> str:
        """The lexical backend: LEXICAL_BACKEND (default ts_rank_cd), never detected. Checked once.

        An installed pg_search is not used unless paradedb is configured, and paradedb
        configured without pg_search raises instead of falling back to ts_rank_cd.
        """
        cached = getattr(self, "_hybrid_backend", None)
        if cached:
            return cached
        backend = getattr(self, "lexical_backend", None) or lexical_backend_setting()
        if backend == "paradedb":
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_search';")
                    installed = cur.fetchone() is not None
            if not installed:
                raise RuntimeError("LEXICAL_BACKEND=paradedb, but the pg_search extension is not installed in this database")
        self._hybrid_backend = backend
        return backend

    def hybrid_sql(self, backend: str | None = None) -> str:
        backend = backend or self.hybrid_backend
        name = "hybrid_paradedb.sql" if backend == "paradedb" else "hybrid_tsrank.sql"
        if name not in self._sql_cache:
            self._sql_cache[name] = (self.SQL_DIR / name).read_text()
        return self._sql_cache[name].replace("{table}", f'"{self.schema_name}"."{self.data_table}"')

    @staticmethod
    def lexical_query(query: str) -> str:
        """OR-join the query terms for websearch_to_tsquery, so the lexical list has
        BM25-like OR semantics instead of websearch's default AND. Stopwords are
        dropped by the 'english' configuration on the Postgres side.

        A term is a run of word characters of any script (letters, digits, "_") and the
        combining marks among them, in the NFC form of the query (word_runs), so an accented
        word stays one term ("café", not "caf"), and so does a word written with separate
        vowel signs ("हिन्दी"). Quotes and a leading "-", websearch's other operators, are
        never part of a term; its operator word "or" is dropped (an English stopword as well).
        """
        terms = []
        for tok in word_runs(unicodedata.normalize("NFC", query or "")):
            t = tok.lower()
            if t not in terms and t not in WEBSEARCH_OPERATOR_WORDS:
                terms.append(t)
        return " or ".join(terms)

    @staticmethod
    def vector_literal(embedding) -> str | None:
        if embedding is None:
            return None
        return "[" + ",".join(f"{float(x):.8g}" for x in embedding) + "]"

    @property
    def pgvector_version(self) -> tuple[int, ...] | None:
        """Installed pgvector version, e.g. (0, 8, 6); None when it is not installed or not readable."""
        cached = getattr(self, "_pgvector_version", None)
        if cached is not None:
            return cached or None
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
                    row = cur.fetchone()
        except Exception as e:  # not cached: the next call asks again
            logger.info(f"pgvector_version: could not read it ({e.__class__.__name__})")
            return None
        self._pgvector_version = tuple(int(p) for p in re.findall(r"\d+", row[0])[:3]) if row else ()
        return self._pgvector_version or None

    def search_settings(self, k: int) -> dict:
        """Settings the hybrid statement runs with for a dense list of depth k, by parameter name.

        hnsw.ef_search is the HNSW scan's candidate list: a scan returns at most that many
        rows (pgvector's default, 40, is below RETRIEVE_K, 50), and its first rows are
        approximate. The dense CTE reads twice the depth from the index and orders a tie at
        the cut by node_id among those rows, so ef_search is twice the depth, at least the
        default and at most pgvector's limit. hnsw.iterative_scan = strict_order (pgvector
        0.8 and later) lets a scan that runs out of candidates keep searching instead of
        returning fewer rows, and keeps its rows in distance order, which the statement's
        LIMIT relies on. Both apply only to the statement's own transaction
        (settings_sql).
        """
        version = self.pgvector_version
        settings: dict = {}
        if version and version >= (0, 5, 0):  # the first release with HNSW
            settings["hnsw.ef_search"] = min(HNSW_EF_SEARCH_MAX, max(2 * int(k), HNSW_EF_SEARCH_DEFAULT))
        if version and version >= (0, 8, 0):
            settings["hnsw.iterative_scan"] = "strict_order"
        return settings

    def settings_sql(self, k: int, statement_timeout_ms: int | None = None) -> tuple[str, dict]:
        """One statement that sets search_settings(k), and statement_timeout_ms if given: (sql, its parameters).

        set_config(..., is_local => true) has SET LOCAL semantics: the values end with the
        transaction (the pool's connections are not in autocommit, and get_connection commits
        after its block), so the next statement on the same pooled connection runs with the
        server defaults again; a session-level SET would stay on whichever pooled connection
        ran it. hybrid_search sends this statement and the hybrid statement in one execute: one
        round trip, one transaction. ("", {}) when there is nothing to set.
        """
        settings = dict(self.search_settings(k))
        if statement_timeout_ms is not None:  # 0 would mean no limit in Postgres
            settings["statement_timeout"] = max(1, int(statement_timeout_ms))
        if not settings:
            return "", {}
        calls, params = [], {}
        for i, (name, value) in enumerate(settings.items()):
            calls.append(f"set_config(%(setting_{i})s, %(setting_value_{i})s, true)")
            params[f"setting_{i}"], params[f"setting_value_{i}"] = name, str(value)
        return "SELECT " + ", ".join(calls) + ";\n", params

    def apply_search_settings(self, cur, k: int) -> dict:
        """Set search_settings(k) on the cursor's transaction, for a caller that runs its own statement after."""
        statement, params = self.settings_sql(k)
        if statement:
            cur.execute(statement, params)
        return self.search_settings(k)

    def hybrid_search(
        self, query: str, query_embedding=None, k: int = 50, rrf_k: int = 60, statement_timeout_ms: int | None = None
    ) -> list[dict]:
        """One round trip: dense top-k + lexical top-k + RRF in SQL (see sql/hybrid_*.sql).

        query_embedding=None runs the lexical list only (router skip_embed). Each row:
        node_id, text, metadata, bm25 (score, rank), dense (score, rank), rrf.
        The statement runs in its own transaction with search_settings(k), and with
        statement_timeout_ms as its statement_timeout when given: Postgres cancels it
        there, the transaction is rolled back and this raises TimeoutError.
        """
        backend = self.hybrid_backend
        params = {
            "qvec": self.vector_literal(query_embedding),
            "query": query if backend == "paradedb" else self.lexical_query(query),
            "k": int(k),
            "rrf_k": float(rrf_k),
        }
        settings, settings_params = self.settings_sql(k, statement_timeout_ms)
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                try:  # the settings and the statement in one execute: one round trip, one transaction
                    cur.execute(settings + self.hybrid_sql(backend), {**params, **settings_params})
                except psycopg2.errors.QueryCanceled as exc:
                    raise TimeoutError(f"hybrid statement cancelled by its statement_timeout ({statement_timeout_ms} ms)") from exc
                rows = cur.fetchall()
        self.last_bm25_backend = backend if any(r[3] is not None for r in rows) else "none"
        return [
            {
                "node_id": r[0],
                "text": r[1],
                "metadata": r[2] or {},
                "bm25_score": None if r[3] is None else float(r[3]),
                "bm25_rank": None if r[4] is None else int(r[4]),
                "dense_score": None if r[5] is None else float(r[5]),
                "dense_rank": None if r[6] is None else int(r[6]),
                "rrf_score": float(r[7]),
            }
            for r in rows
        ]

    def explain_hybrid(self, query: str, query_embedding=None, k: int = 50, rrf_k: int = 60) -> str:
        """EXPLAIN (ANALYZE, BUFFERS) of the hybrid query with the settings hybrid_search uses, for doc/sql/."""
        backend = self.hybrid_backend
        params = {
            "qvec": self.vector_literal(query_embedding),
            "query": query if backend == "paradedb" else self.lexical_query(query),
            "k": int(k),
            "rrf_k": float(rrf_k),
        }
        settings, settings_params = self.settings_sql(k)
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(settings + "EXPLAIN (ANALYZE, BUFFERS) " + self.hybrid_sql(backend), {**params, **settings_params})
                return "\n".join(r[0] for r in cur.fetchall())

    def fetch_all_chunks(self) -> list[dict]:
        """node_id, text, metadata for every row (staleness clusters at startup)."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("SELECT node_id, text, metadata_ FROM {table};").format(
                    table=sql.Identifier(self.schema_name, self.data_table)))
                return [{"node_id": r[0], "text": r[1], "metadata": r[2] or {}} for r in cur.fetchall()]

    @staticmethod
    def _bm25_rows(results) -> list[dict]:
        """Rows are (node_id, text, metadata_, score). metadata_ is JSON/JSONB, decoded by psycopg2."""
        return [
            {"node_id": row[0], "text": row[1], "metadata": row[2] or {}, "score": row[3]}
            for row in results
        ]

    def count_nodes(self) -> int:
        """Number of rows in the data table; 0 only when the table does not exist (UndefinedTable).

        Any other error (the database unreachable, a missing privilege) is raised. It used to read
        as an empty table, so /health answered ok with n_nodes 0 while the database was down.
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql.SQL("SELECT count(*) FROM {table};").format(
                        table=sql.Identifier(self.schema_name, self.data_table)))
                    return int(cur.fetchone()[0])
        except psycopg2.errors.UndefinedTable:
            logger.info(f"count_nodes: {self.schema_name}.{self.data_table} does not exist")
            return 0

    # ------------------------------------------------------------------ ingest manifest (incremental indexing)
    # One row per indexed file: path, SHA-256 of its bytes, chunk count, and the
    # pipeline (chunker version + embedder) that produced its rows. It is the
    # source of truth for the diff app/engine/ingest.py runs in every indexing run. A file's
    # manifest row is written only after its chunks are in the data table, and is
    # deleted in the same transaction as those chunks, so the manifest never lists a
    # file whose rows are missing. The table and the path index the per-file delete reads
    # are schema migrations (migrations/vector/0002 and 0003), not created here.

    @property
    def manifest_table(self) -> str:
        return f"ingest_manifest_{self.table_name}"

    @property
    def ingest_lock_key(self) -> int:
        """The pg_advisory_lock key of this table's indexing step (a signed 64-bit integer derived from its name)."""
        name = f"metanavit.ingest:{self.schema_name}.{self.data_table}"
        return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big", signed=True)

    @contextmanager
    def ingest_lock(self, timeout_s: float | None = None):
        """Session-level advisory lock around the diff-and-index step.

        Two processes that index together (the index step, an upload, a replica
        with INDEX_ON_START) serialise here: the second waits, then diffs against
        the manifest the first one wrote and finds nothing to do. It waits at most
        timeout_s seconds (None or 0: no limit), then raises IngestLockTimeout
        having written nothing; the wait used to have no limit, so a stuck holder
        stopped every other start. The lock lives on its own connection, so
        closing it (or the process dying) releases it. The waiter blocks inside
        pg_advisory_lock under lock_timeout, so it shows in pg_locks as an
        advisory lock not granted yet.
        """
        key = self.ingest_lock_key
        conn = psycopg2.connect(self.conn_string)
        conn.autocommit = True
        started = time.perf_counter()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_try_advisory_lock(%s);", (key,))
                waited = not cur.fetchone()[0]
                if waited:
                    logger.info(f"Another process is indexing; waiting {'up to %g s ' % timeout_s if timeout_s else ''}for the ingest lock")
                    if timeout_s:
                        cur.execute("SELECT set_config('lock_timeout', %s, false);", (f"{max(1, int(timeout_s * 1000))}ms",))
                    try:
                        cur.execute("SELECT pg_advisory_lock(%s);", (key,))
                    except psycopg2.errors.LockNotAvailable as exc:
                        raise IngestLockTimeout(
                            f"another process held the ingest lock of {self.schema_name}.{self.data_table} for more than "
                            f"{timeout_s:g} s (INGEST_LOCK_TIMEOUT_SECONDS); nothing was indexed"
                        ) from exc
            yield {"waited": waited, "waited_ms": round((time.perf_counter() - started) * 1000.0, 1)}
        finally:
            conn.close()

    def manifest_summary(self) -> dict:
        """What the manifest lists, without reading a chunk: {"files", "chunks", "pipelines"} (pipelines sorted)."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("SELECT count(*), coalesce(sum(n_chunks), 0), array_agg(DISTINCT pipeline) FROM {manifest};").format(
                    manifest=sql.Identifier(self.schema_name, self.manifest_table)))
                files, chunks, pipelines = cur.fetchone()
        return {"files": int(files), "chunks": int(chunks), "pipelines": sorted(p for p in (pipelines or []) if p is not None)}

    def ingest_manifest(self) -> dict[str, dict]:
        """{path: {"content_hash", "n_chunks", "pipeline"}} for every file the manifest lists."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("SELECT path, content_hash, n_chunks, pipeline FROM {manifest};").format(
                    manifest=sql.Identifier(self.schema_name, self.manifest_table)))
                return {r[0]: {"content_hash": r[1], "n_chunks": int(r[2]), "pipeline": r[3]} for r in cur.fetchall()}

    def forget_file(self, path: str) -> int:
        """Delete a file's chunks (by path metadata) and its manifest row in one transaction; return chunks deleted."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("DELETE FROM {manifest} WHERE path = %s;").format(
                    manifest=sql.Identifier(self.schema_name, self.manifest_table)), (path,))
                cur.execute(sql.SQL("DELETE FROM {table} WHERE metadata_->>'path' = %s;").format(
                    table=sql.Identifier(self.schema_name, self.data_table)), (path,))
                return int(cur.rowcount)

    def record_file(self, path: str, content_hash: str, n_chunks: int, pipeline: str = "") -> None:
        """List a file in the manifest. Call only after its chunks are committed to the data table."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql.SQL(
                    "INSERT INTO {manifest} (path, content_hash, n_chunks, pipeline) VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (path) DO UPDATE SET content_hash = EXCLUDED.content_hash, "
                    "n_chunks = EXCLUDED.n_chunks, pipeline = EXCLUDED.pipeline, indexed_at = now();"
                ).format(manifest=sql.Identifier(self.schema_name, self.manifest_table)), (path, content_hash, int(n_chunks), pipeline))

    def rows_per_path(self) -> dict[str, int]:
        """{path: chunk rows} for every row that carries a path, what the manifest's n_chunks must equal."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql.SQL(
                    "SELECT metadata_->>'path', count(*) FROM {table} WHERE metadata_->>'path' IS NOT NULL GROUP BY 1;"
                ).format(table=sql.Identifier(self.schema_name, self.data_table)))
                return {r[0]: int(r[1]) for r in cur.fetchall()}

    def delete_unlisted_chunks(self) -> int:
        """Delete chunks whose path the manifest does not list (an interrupted run, or rows from before the manifest)."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql.SQL(
                    "DELETE FROM {table} t WHERE t.metadata_->>'path' IS NOT NULL "
                    "AND NOT EXISTS (SELECT 1 FROM {manifest} m WHERE m.path = t.metadata_->>'path');"
                ).format(table=sql.Identifier(self.schema_name, self.data_table),
                         manifest=sql.Identifier(self.schema_name, self.manifest_table)))
                return int(cur.rowcount)

    def close(self) -> None:
        """Close all connections and cleanup resources."""
        if self.vector_store:
            # Add any vector store cleanup here if needed
            self.vector_store = None
        super().close()  # Call parent close method

    def _detect_vector_dimensions(self) -> int:
        """Detect the vector dimensions in the existing table, if any."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Try to query the table definition to get vector dimensions
                    cur.execute(f"""
                        SELECT a.atttypmod  -- Subtracting 4 gets the actual dimensions
                        FROM pg_attribute a
                        JOIN pg_class c ON a.attrelid = c.oid
                        JOIN pg_namespace n ON c.relnamespace = n.oid
                        WHERE n.nspname = '{self.schema_name}'
                        AND c.relname = '{self.data_table}'
                        AND a.attname = 'embedding'
                        AND a.atttypid = (SELECT oid FROM pg_type WHERE typname = 'vector');
                    """)
                    result = cur.fetchone()
                    if result and result[0] > 0:
                        return result[0]  # Return the detected dimension
                    
                    # If that fails, let's try another approach
                    logger.info("Trying alternate method to detect vector dimensions...")
                    cur.execute(f"""
                        SELECT description 
                        FROM pg_description 
                        JOIN pg_class ON pg_description.objoid = pg_class.oid
                        JOIN pg_namespace ON pg_class.relnamespace = pg_namespace.oid
                        WHERE pg_namespace.nspname = '{self.schema_name}'
                        AND pg_class.relname = '{self.data_table}';
                    """)
                    # If no result or can't parse dimension, we'll return None
                    
                    return None
        except Exception as e:
            logger.warning(f"Error detecting vector dimensions: {e}")
            return None


# Global instance - Consider if singleton is truly needed or if explicit instantiation is better
_vector_store_manager = None

def get_vector_store_manager() -> VectorStoreManager: # Renamed for clarity
    """Get global vector store manager instance."""
    global _vector_store_manager
    if _vector_store_manager is None:
        logger.info("Initializing global VectorStoreManager...")
        _vector_store_manager = VectorStoreManager()
    return _vector_store_manager

# Keep original function for compatibility if needed, but point to manager
def get_vector_store() -> PGVectorStore:
    """Get global vector store instance via the manager."""
    manager = get_vector_store_manager()
    return manager.get_vector_store()

# --- New Function: Expose BM25 Search ---
def search_bm25(query: str, limit: int = 5) -> list[dict]:
    """Perform BM25 search using the global VectorStoreManager."""
    manager = get_vector_store_manager()
    return manager.search_bm25(query=query, limit=limit)