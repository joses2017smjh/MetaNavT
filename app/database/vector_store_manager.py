"""
Vector Store Manager Module

Manages PostgreSQL vector operations using pgvector extension for document embeddings.
Provides connection pooling and vector store management through a singleton pattern.

Features:
    - Vector store initialization and management
    - Connection pooling via parent DatabaseManager
    - pgvector extension handling
    - Both sync (psycopg2) and async (asyncpg) support
    - Automatic table creation and dimension handling

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
import os
import logging
import re
from pathlib import Path
from llama_index.vector_stores.postgres import PGVectorStore
from urllib.parse import urlparse
from psycopg2 import sql
from dotenv import load_dotenv
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
import uuid
import time
import psycopg2

from .db_base_manager import DatabaseManager

load_dotenv()
logger = logging.getLogger("uvicorn")

class VectorStoreManager(DatabaseManager):
    """Manages vector store operations using pgvector and pg_search extensions."""
    
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
        
        # Ensure required extensions exist
        self._ensure_vector_extension()
        self._ensure_pg_search_extension()
        
        # PGVectorStore creates data_<table_name> itself on first use (see get_vector_store).
        
    def _ensure_vector_extension(self) -> None:
        """Create pgvector extension if it doesn't exist."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    logger.info("Ensuring 'vector' extension exists...")
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    logger.info("'vector' extension check complete.")
                except Exception as e:
                    logger.error(f"Error creating vector extension: {e}")
                    # Rollback might not be needed if autocommit is true, but good practice
                    conn.rollback()
                    raise

    def _ensure_pg_search_extension(self):
        """Ensure the pg_search extension exists."""
        logger.info("Ensuring 'pg_search' extension (in 'paradedb' schema) exists...")
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Check for and create schema if it doesn't exist
                    cur.execute("CREATE SCHEMA IF NOT EXISTS paradedb;")
                    
                    # Check if pg_search is already installed
                    cur.execute("""
                        SELECT 1 FROM pg_extension WHERE extname = 'pg_search';
                    """)
                    if cur.fetchone() is None:
                        # Try to create it
                        try:
                            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_search;")
                        except Exception as e:
                            logger.warning(f"Could not create pg_search extension: {e}")
                    
                    # The BM25 index is created in get_vector_store(), once the data table exists.
            logger.info("'pg_search' extension check complete (expected in 'paradedb' schema).")
        except Exception as e:
            logger.warning(f"Unable to verify pg_search extension: {e}")
        
    def _create_bm25_index(self, conn):
        """Create or recreate BM25 index for text column."""
        logger.info(f"Proceeding to ensure BM25 index exists for {self.schema_name}.{self.data_table}...")
        try:
            with conn.cursor() as cur:
                # Drop existing index if it exists to force refresh
                index_name = f"{self.table_name}_bm25_idx_on_text"
                cur.execute(f"DROP INDEX IF EXISTS {index_name};")
                
                # Create BM25 index with proper ParadeDB syntax based on docs
                sql_stmt = f"""
                CREATE INDEX IF NOT EXISTS {index_name} ON {self.schema_name}.{self.data_table}
                USING bm25 (node_id, text)
                WITH (
                    key_field='node_id',
                    text_fields='{{
                        "text": {{"tokenizer": {{"type": "default", "stemmer": "English"}}}}
                    }}'
                );
                """
                logger.info(f"Creating BM25 index '{index_name}' using ParadeDB documentation syntax")
                logger.debug(f"Executing SQL:\n{sql_stmt}\n")
                cur.execute(sql_stmt)
                
                # Analyze the table to refresh statistics
                cur.execute(f"ANALYZE {self.schema_name}.{self.data_table};")
                
                logger.info("BM25 index created successfully")
            logger.info("BM25 index creation/verification successful")
            return True
        except Exception as e:
            logger.warning(f"Error creating BM25 index: {e}")
            return False

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
            logger.info(f"PGVectorStore will use psycopg2_conn_str: {psycopg2_conn_str}")
            logger.info(f"PGVectorStore will use asyncpg_conn_str: {asyncpg_conn_str}")
            
            # Log the database name expected by DatabaseManager's pool
            # Assuming self.DATABASE_NAME is set in DatabaseManager's __init__
            # from the same PG_CONNECTION_STRING
            if hasattr(self, 'DATABASE_NAME'):
                logger.info(f"DatabaseManager's pool is configured for database: '{self.DATABASE_NAME}' (derived from PG_CONNECTION_STRING)")
            else: # Fallback if DATABASE_NAME isn't on self for some reason
                parsed_main_conn = urlparse(self.conn_string if self.conn_string else conn_string_env)
                logger.info(f"DatabaseManager's pool is configured for database: '{parsed_main_conn.path.lstrip('/')}' (derived from PG_CONNECTION_STRING)")


            logger.info(f"Attempting to initialize PGVectorStore for table '{self.schema_name}.{self.data_table}' (PGVectorStore creates data_<table> on first use)...")
            self.vector_store = PGVectorStore(
                connection_string=psycopg2_conn_str,
                async_connection_string=asyncpg_conn_str,
                schema_name=self.schema_name,
                table_name=self.table_name,
                embed_dim=self.embed_dim,
            )
            logger.info(f"PGVectorStore Python object initialized for table '{self.schema_name}.{self.data_table}'")

            # Probe (Optional but good for sanity check - can be simplified now)
            dummy_node_id_for_probe = f"test_node_{uuid.uuid4()}"
            try:
                logger.info(f"Probe: Attempting to ADD dummy node '{dummy_node_id_for_probe}' to pre-created table...")
                logger.info(f"Using embedding dimension: {self.embed_dim}")
                
                # Create a zero vector with the CORRECT dimension
                zero_embedding = [0.0] * self.embed_dim
                
                # PGVectorStore.delete() removes rows by ref_doc_id (stored as doc_id in
                # metadata_), so the probe node must declare itself as its own source or
                # the delete below is a no-op and the dummy row stays retrievable.
                dummy_node = TextNode(
                    id_=dummy_node_id_for_probe,
                    text="dummy_text_content_for_probe_in_precreated_table",
                    embedding=zero_embedding,  # This will now have the correct dimensions
                    metadata={"text": "dummy_metadata_text_for_probe_in_precreated_table"},
                    relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=dummy_node_id_for_probe)},
                )
                self.vector_store.add([dummy_node])
                logger.info(f"Probe: Successfully EXECUTED add for dummy node '{dummy_node_id_for_probe}'.")
                # Now, try to delete it immediately using PGVectorStore to ensure it can write and delete
                self.vector_store.delete(ref_doc_id=dummy_node_id_for_probe)
                logger.info(f"Probe: Successfully deleted dummy node '{dummy_node_id_for_probe}' using PGVectorStore.")
            except Exception as e:
                logger.error(f"Probe FAILED during PGVectorStore.add/delete on pre-created table: {e}", exc_info=True)
                # Provide more helpful error for dimension mismatch
                if "expected" in str(e) and "dimensions" in str(e):
                    actual_dim = None
                    try:
                        # Try to extract the expected dimension from the error message
                        import re
                        match = re.search(r'expected (\d+) dimensions', str(e))
                        if match:
                            actual_dim = int(match.group(1))
                            logger.error(f"DIMENSION MISMATCH ERROR: Database expects {actual_dim} dimensions, but code is using {self.embed_dim}")
                            logger.error(f"To fix: Set EMBEDDING_DIM={actual_dim} in your environment variables")
                    except:
                        pass
                
                raise RuntimeError(
                    f"PGVectorStore failed .add/delete on data table '{self.schema_name}.{self.data_table}'."
                ) from e

            # BM25 Index Creation - attempt it but don't fail if it doesn't work
            logger.info(f"Proceeding to ensure BM25 index exists for {self.schema_name}.{self.data_table}...")
            try:
                with self.get_connection() as conn:
                    bm25_created = self._create_bm25_index(conn)
                if bm25_created:
                    logger.info("BM25 index creation/verification successful")
                else:
                     logger.warning("BM25 index creation failed, but continuing anyway")  
            except Exception as e:
                logger.warning(f"BM25 index creation failed: {e}", exc_info=True)
                logger.warning("Continuing without BM25 search capability")
                # We'll continue without BM25 functionality

        return self.vector_store

    def search_bm25(self, query: str, limit: int = 5) -> list[dict]:
        """
        Perform BM25 search using the 'text' column.
        
        If BM25 search fails, falls back to a basic text search.
        """
        if not query:
            raise ValueError("Search query cannot be empty")

        results = []
        
        # Try multiple BM25 query formats
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
                SELECT node_id, text, metadata_, ts_rank_cd(to_tsvector('english', text), plainto_tsquery('english', %s)) AS score
                FROM {table}
                WHERE to_tsvector('english', text) @@ plainto_tsquery('english', %s)
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
        """'paradedb' when pg_search is installed, else 'ts_rank_cd'. Checked once."""
        cached = getattr(self, "_hybrid_backend", None)
        if cached:
            return cached
        backend = "ts_rank_cd"
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_search';")
                    if cur.fetchone() is not None:
                        backend = "paradedb"
        except Exception as e:
            logger.info(f"hybrid_backend: could not check pg_search ({e.__class__.__name__}); using ts_rank_cd")
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
        dropped by the 'english' configuration on the Postgres side."""
        terms = []
        for tok in re.findall(r"[A-Za-z0-9_]+", query or ""):
            t = tok.lower()
            if t not in terms:
                terms.append(t)
        return " or ".join(terms)

    @staticmethod
    def vector_literal(embedding) -> str | None:
        if embedding is None:
            return None
        return "[" + ",".join(f"{float(x):.8g}" for x in embedding) + "]"

    def hybrid_search(self, query: str, query_embedding=None, k: int = 50, rrf_k: int = 60) -> list[dict]:
        """One round trip: dense top-k + lexical top-k + RRF in SQL (see sql/hybrid_*.sql).

        query_embedding=None runs the lexical list only (router skip_embed). Each row:
        node_id, text, metadata, bm25 (score, rank), dense (score, rank), rrf.
        """
        backend = self.hybrid_backend
        params = {
            "qvec": self.vector_literal(query_embedding),
            "query": query if backend == "paradedb" else self.lexical_query(query),
            "k": int(k),
            "rrf_k": float(rrf_k),
        }
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(self.hybrid_sql(backend), params)
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
        """EXPLAIN (ANALYZE, BUFFERS) of the hybrid query, for doc/sql/."""
        backend = self.hybrid_backend
        params = {
            "qvec": self.vector_literal(query_embedding),
            "query": query if backend == "paradedb" else self.lexical_query(query),
            "k": int(k),
            "rrf_k": float(rrf_k),
        }
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("EXPLAIN (ANALYZE, BUFFERS) " + self.hybrid_sql(backend), params)
                return "\n".join(r[0] for r in cur.fetchall())

    def ensure_text_index(self) -> bool:
        """GIN index on to_tsvector('english', text): the expression the lexical CTE filters on."""
        index_name = f"{self.data_table}_text_tsv_gin"
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{self.schema_name}"."{self.data_table}" '
                        f"USING GIN (to_tsvector('english', text));"
                    )
                conn.commit()
            return True
        except Exception as e:
            logger.warning(f"ensure_text_index failed: {e}")
            return False

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
        """Number of rows in the data table (0 when it does not exist yet)."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql.SQL("SELECT count(*) FROM {table};").format(
                        table=sql.Identifier(self.schema_name, self.data_table)))
                    return int(cur.fetchone()[0])
        except Exception as e:
            logger.info(f"count_nodes: data table not readable yet ({e.__class__.__name__})")
            return 0

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

    def ensure_hnsw(self, m: int = 16, ef_construction: int = 64) -> bool:
        """Create an HNSW index on the embedding column. No-op if pgvector lacks HNSW."""
        index_name = f"{self.table_name}_embedding_hnsw"
        stmt = f"""
            CREATE INDEX IF NOT EXISTS {index_name}
            ON {self.schema_name}.{self.data_table}
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = {int(m)}, ef_construction = {int(ef_construction)});
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(stmt)
                conn.commit()
            logger.info(f"HNSW index {index_name} ready (m={m}, ef_construction={ef_construction})")
            return True
        except Exception as e:
            logger.warning(f"HNSW index not created: {e}")
            return False

    def set_ef_search(self, ef_search: int) -> None:
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SET hnsw.ef_search = %s", (int(ef_search),))
                try:
                    cur.execute("SET hnsw.iterative_scan = relaxed_order")
                except Exception:
                    pass

    def ensure_halfvec(self) -> bool:
        """2x storage cut. Requires pgvector with halfvec."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        ALTER TABLE {self.schema_name}.{self.data_table}
                        ADD COLUMN IF NOT EXISTS embedding_half halfvec({self.embed_dim});
                        """
                    )
                    cur.execute(
                        f"""
                        UPDATE {self.schema_name}.{self.data_table}
                        SET embedding_half = embedding::halfvec
                        WHERE embedding_half IS NULL AND embedding IS NOT NULL;
                        """
                    )
                conn.commit()
            return True
        except Exception as e:
            logger.warning(f"halfvec column not created: {e}")
            return False

    def ensure_binary(self) -> bool:
        """1-bit/dim storage + Hamming index. Rescore with float cosine at query time."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        ALTER TABLE {self.schema_name}.{self.data_table}
                        ADD COLUMN IF NOT EXISTS embedding_bit bit({self.embed_dim});
                        """
                    )
                    cur.execute(
                        f"""
                        UPDATE {self.schema_name}.{self.data_table}
                        SET embedding_bit = binary_quantize(embedding)::bit({self.embed_dim})
                        WHERE embedding_bit IS NULL AND embedding IS NOT NULL;
                        """
                    )
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_bit_hnsw
                        ON {self.schema_name}.{self.data_table}
                        USING hnsw (embedding_bit bit_hamming_ops);
                        """
                    )
                conn.commit()
            return True
        except Exception as e:
            logger.warning(f"binary quantization column not created: {e}")
            return False


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