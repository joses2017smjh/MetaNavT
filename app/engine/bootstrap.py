"""API process startup: build the retrieval singletons, bring the index in line with DATA_DIR.

main.py's lifespan calls build_retrieval_state() once, off the event loop. With
INDEX_ON_START=true (docker-compose sets it), every start runs
app/engine/ingest.index_directory under a Postgres advisory lock: DATA_DIR is
scanned, each file's SHA-256 is compared with the manifest table, and only new
or changed files are chunked (app/chunking, the benchmark's chunker), embedded
with Settings.embed_model and written to pgvector; rows of removed files are
deleted. An unchanged directory embeds nothing, so `docker compose up` does not
re-embed on every start. The run's report is kept on the state for /health.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from llama_index.core.settings import Settings

from app.database.vector_store import get_vector_store_manager
from app.engine.index import get_index
from app.engine.ingest import index_directory, max_rejected_from_env
from app.engine.retriever import HybridRetriever, create_hybrid_retriever
from app.graph.staleness import cluster_versions
from app.retrieval.types import Chunk

logger = logging.getLogger("uvicorn")


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
    indexing: dict = {"indexed": False, "reason": "INDEX_ON_START not set"}
    if os.getenv("INDEX_ON_START", "false").strip().lower() in {"1", "true", "yes"}:
        indexing = index_directory(
            vsm,
            os.getenv("DATA_DIR", "data"),
            embed_fn=Settings.embed_model.get_text_embedding_batch,
            embedder=str(getattr(Settings.embed_model, "model_name", "") or type(Settings.embed_model).__name__),
            max_rejected=max_rejected_from_env(),
        )
    vsm.ensure_text_index()  # GIN on to_tsvector('english', text) for the lexical CTE
    try:
        vsm.ensure_hnsw()  # HNSW on the embedding column for the dense CTE
    except Exception as exc:  # noqa: BLE001 - index is an optimisation, not a requirement
        logger.warning(f"ensure_hnsw failed: {exc}")
    enable_staleness = os.getenv("ENABLE_STALENESS", "true").strip().lower() != "false"
    clusters = build_clusters(vsm) if enable_staleness else {}
    logger.info(f"Staleness Tier 1: {len(clusters)} version clusters")
    index = get_index()
    retriever = create_hybrid_retriever(index, vsm, clusters=clusters)
    logger.info(f"Retriever mode={retriever.mode} staleness={retriever.staleness_enabled}")
    return RetrievalState(vsm=vsm, index=index, retriever=retriever, indexing=indexing)
