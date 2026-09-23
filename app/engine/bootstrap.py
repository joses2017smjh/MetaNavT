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
from llama_index.core.settings import Settings

from app.database.vector_store import get_vector_store_manager
from app.engine.index import get_index
from app.engine.retriever import HybridRetriever, create_hybrid_retriever

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
            Settings.embed_model,
        ],
        vector_store=vsm.get_vector_store(),
    )
    nodes = pipeline.run(documents=documents, show_progress=False)
    logger.info(f"Indexed {len(nodes)} nodes")
    return {"indexed": True, "n_documents": len(documents), "n_nodes": len(nodes), "data_dir": data_dir}


def build_retrieval_state() -> RetrievalState:
    """Connect once, optionally index, and build the retriever the API serves."""
    vsm = get_vector_store_manager()
    indexing: dict = {"indexed": False, "reason": "INDEX_ON_START not set"}
    if os.getenv("INDEX_ON_START", "false").strip().lower() in {"1", "true", "yes"}:
        indexing = index_directory_if_empty(vsm, os.getenv("DATA_DIR", "data"))
    index = get_index()
    retriever = create_hybrid_retriever(index, vsm)
    return RetrievalState(vsm=vsm, index=index, retriever=retriever, indexing=indexing)
