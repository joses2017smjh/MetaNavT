"""API process startup: build the retrieval singletons, index a directory on first start.

main.py's lifespan calls build_retrieval_state() once, off the event loop. With
INDEX_ON_START=true (docker-compose sets it) and an empty vector table, DATA_DIR
is read with SimpleDirectoryReader, split with Settings.chunk_size /
chunk_overlap, embedded with Settings.embed_model and written to pgvector. While
the table has rows the step is a no-op, so `docker compose up` does not
re-embed on every start (drop the volume to re-index).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TransformComponent
from llama_index.core.settings import Settings
from pydantic import Field

from app.database.vector_store import get_vector_store_manager
from app.engine.index import get_index
from app.engine.retriever import HybridRetriever, create_hybrid_retriever
from app.graph.staleness import cluster_versions
from app.retrieval.types import Chunk

OFFSET_KEYS = ("start_byte", "end_byte")


class ByteOffsets(TransformComponent):
    """Store each chunk's byte range in its source file as metadata.

    SentenceSplitter records start_char_idx / end_char_idx; the API cites bytes
    (like the bench's Chunk), so the char offsets are converted against the
    source document's text. The keys are excluded from the embedded text.
    """

    doc_texts: dict = Field(default_factory=dict)

    def __call__(self, nodes, **kwargs):
        for node in nodes:
            src = node.source_node
            text = self.doc_texts.get(src.node_id) if src is not None else None
            if text is None or node.start_char_idx is None or node.end_char_idx is None:
                continue
            node.metadata["start_byte"] = len(text[: node.start_char_idx].encode("utf-8"))
            node.metadata["end_byte"] = len(text[: node.end_char_idx].encode("utf-8"))
            for key in OFFSET_KEYS:
                if key not in node.excluded_embed_metadata_keys:
                    node.excluded_embed_metadata_keys.append(key)
                if key not in node.excluded_llm_metadata_keys:
                    node.excluded_llm_metadata_keys.append(key)
        return nodes

logger = logging.getLogger("uvicorn")


@dataclass
class RetrievalState:
    vsm: Any
    index: Any
    retriever: HybridRetriever
    indexing: dict = field(default_factory=dict)


def load_directory(data_dir: str) -> list:
    """Read every file under data_dir; add a `path` relative to it (the bench's path form)."""
    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"DATA_DIR {data_dir!r} is not a directory")
    documents = SimpleDirectoryReader(input_dir=str(root), recursive=True, filename_as_id=True).load_data()
    for doc in documents:
        abs_path = doc.metadata.get("file_path")
        doc.metadata["path"] = os.path.relpath(abs_path, root) if abs_path else doc.metadata.get("file_name")
        try:
            doc.metadata["mtime"] = os.stat(abs_path).st_mtime if abs_path else 0.0  # staleness clusters
        except OSError:
            doc.metadata["mtime"] = 0.0
        # Embed and rank on the text alone, as the bench does; metadata stays out of the vectors.
        doc.excluded_embed_metadata_keys = list(doc.metadata.keys())
        doc.excluded_llm_metadata_keys = list(doc.metadata.keys())
    return documents


def index_directory_if_empty(vsm, data_dir: str) -> dict:
    existing = vsm.count_nodes()
    if existing > 0:
        logger.info(f"Index already has {existing} nodes; skipping INDEX_ON_START")
        return {"indexed": False, "n_nodes": existing, "reason": "table already populated"}
    documents = load_directory(data_dir)
    logger.info(f"Indexing {len(documents)} documents from {data_dir} with {type(Settings.embed_model).__name__}")
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap),
            ByteOffsets(doc_texts={doc.doc_id: doc.text for doc in documents}),
            Settings.embed_model,
        ],
        vector_store=vsm.get_vector_store(),
    )
    nodes = pipeline.run(documents=documents, show_progress=False)
    logger.info(f"Indexed {len(nodes)} nodes")
    return {"indexed": True, "n_documents": len(documents), "n_nodes": len(nodes), "data_dir": data_dir}


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
            )
        )
    return cluster_versions(chunks)


def build_retrieval_state() -> RetrievalState:
    """Connect once, optionally index, and build the retriever the API serves."""
    vsm = get_vector_store_manager()
    indexing: dict = {"indexed": False, "reason": "INDEX_ON_START not set"}
    if os.getenv("INDEX_ON_START", "false").strip().lower() in {"1", "true", "yes"}:
        indexing = index_directory_if_empty(vsm, os.getenv("DATA_DIR", "data"))
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
