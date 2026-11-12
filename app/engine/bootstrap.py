"""API process startup: build the retrieval singletons, bring the index in line with DATA_DIR.

main.py's lifespan calls build_retrieval_state() off the event loop, and again with backoff
while it fails (startup_retry_policy below; /health is 503 until it succeeds). It first
requires every schema migration this configuration needs (the deploy step,
`python -m app.database.migrate`, applies them; start-up runs no DDL) and raises
MigrationsPending otherwise. Indexing is a step of its own that runs before the API
(`python -m app.engine.ingest`, app/engine/ingest.py; Compose and CI run it after the
migrations), so a start does not index and does not wait for another process that does:
/health reports what the manifest lists. With INDEX_ON_START=true (a single-process
setup) the start runs the same index_data_dir call first: DATA_DIR is scanned, each file's
SHA-256 is compared with the manifest table, and only new or changed files are chunked
(app/chunking, the benchmark's chunker), embedded with Settings.embed_model and written to
pgvector; rows of removed files are deleted. That run's report is kept on the state for /health.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Iterator

from app.database.vector_store import get_vector_store_manager
from app.engine.index import get_index
from app.engine.ingest import index_data_dir
from app.engine.retriever import HybridRetriever, create_hybrid_retriever
from app.eval.ingest import CHUNKER
from app.graph.staleness import cluster_versions
from app.retrieval.types import Chunk

logger = logging.getLogger("uvicorn")

# A start-up attempt that fails (the database not up yet, a migration not applied yet) is retried
# by main.py's lifespan in the background, without a restart: STARTUP_RETRY_SECONDS before the
# first retry, doubling up to STARTUP_RETRY_MAX_SECONDS, at most STARTUP_RETRY_ATTEMPTS attempts
# in all (0 or unset: no limit; 1: never retry, the behaviour before retries existed).
STARTUP_RETRY_DEFAULT_SECONDS = 1.0
STARTUP_RETRY_DEFAULT_MAX_SECONDS = 30.0


@dataclass(frozen=True)
class StartupRetryPolicy:
    initial_s: float
    max_s: float
    attempts: int  # 0: no limit


def _positive_seconds(name: str, default: float) -> float:
    """A positive finite number of seconds from the environment; anything else keeps the default."""
    try:
        value = float(os.getenv(name, "").strip())
    except ValueError:
        return default
    return value if math.isfinite(value) and value > 0 else default


def startup_retry_policy() -> StartupRetryPolicy:
    """STARTUP_RETRY_SECONDS, STARTUP_RETRY_MAX_SECONDS, STARTUP_RETRY_ATTEMPTS (see above)."""
    try:
        attempts = int(os.getenv("STARTUP_RETRY_ATTEMPTS", "").strip())
    except ValueError:
        attempts = 0
    return StartupRetryPolicy(
        initial_s=_positive_seconds("STARTUP_RETRY_SECONDS", STARTUP_RETRY_DEFAULT_SECONDS),
        max_s=_positive_seconds("STARTUP_RETRY_MAX_SECONDS", STARTUP_RETRY_DEFAULT_MAX_SECONDS),
        attempts=max(0, attempts),
    )


def retry_delays(initial_s: float, max_s: float) -> Iterator[float]:
    """The wait before each retry: initial_s, then twice the previous wait, never more than max_s."""
    delay = min(initial_s, max_s)
    while True:
        yield delay
        delay = min(delay * 2, max_s)


@dataclass
class RetrievalState:
    vsm: Any
    index: Any
    retriever: HybridRetriever
    indexing: dict = field(default_factory=dict)


def build_clusters(vsm) -> dict:
    """Staleness Tier 1 version clusters over every indexed chunk (app/graph/staleness, unchanged)."""
    chunks = []
    for row in vsm.fetch_all_chunks():
        meta = row["metadata"] or {}
        chunks.append(
            Chunk(
                chunk_id=row["node_id"],
                path=meta.get("path") or meta.get("file_path") or row["node_id"],
                text=row["text"] or "",
                start_byte=int(meta.get("start_byte") or 0),
                end_byte=int(meta.get("end_byte") or 0),
                mtime=float(meta.get("mtime") or 0.0),
                content_hash=str(meta.get("content_hash") or ""),
            )
        )
    return cluster_versions(chunks)


def build_retrieval_state() -> RetrievalState:
    """Connect once, optionally index, and build the retriever the API serves."""
    vsm = get_vector_store_manager()
    # LEXICAL_BACKEND is checked before anything else: paradedb without pg_search fails the start.
    logger.info(f"Lexical backend: {vsm.hybrid_backend} (LEXICAL_BACKEND)")
    # Then the schema: a pending migration (database, vector table, ParadeDB when selected) fails the start.
    vsm.check_migrations()
    if os.getenv("INDEX_ON_START", "false").strip().lower() in {"1", "true", "yes"}:
        indexing = index_data_dir(vsm)
    else:
        # The index step (python -m app.engine.ingest) indexes DATA_DIR; this start only reports what it listed.
        indexing = {"indexed": False, "reason": "INDEX_ON_START not set", "chunker": CHUNKER, "manifest": vsm.manifest_summary()}
    # The GIN and HNSW indexes the hybrid statement reads are migrations (vector/0004, 0005), not built here.
    enable_staleness = os.getenv("ENABLE_STALENESS", "true").strip().lower() != "false"
    clusters = build_clusters(vsm) if enable_staleness else {}
    logger.info(f"Staleness Tier 1: {len(clusters)} version clusters")
    index = get_index()
    retriever = create_hybrid_retriever(index, vsm, clusters=clusters)
    logger.info(
        f"Retriever mode={retriever.mode} staleness={retriever.staleness_enabled} "
        f"depth={retriever.depth} dense search settings={vsm.search_settings(retriever.depth)}"
    )
    return RetrievalState(vsm=vsm, index=index, retriever=retriever, indexing=indexing)
