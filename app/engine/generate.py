"""
Document Ingestion Pipeline Module

`python -m app.engine.generate` (scripts/run.sh generate) is the index step under its old
name: generate_datasource() runs app/engine/ingest.run_index_step, the same call as
`python -m app.engine.ingest` (make index) and as an API start with INDEX_ON_START. DATA_DIR
is chunked with the benchmark's chunker and diffed against the ingest manifest, so a run after
any other indexing run embeds only what changed.

It used to load files with the crawler (app/engine/loaders.get_documents) and split them with
LlamaIndex's SentenceSplitter (run_pipeline below) into the same table, without byte offsets or
the manifest, so a second run added a second copy of every chunk. run_pipeline, get_doc_store and
persist_storage are kept (the owner's decision on team code, OD12) and nothing calls them; neither
does anything call the crawler now.
"""
# flake8: noqa: E402
from dotenv import load_dotenv

# Explicit environment variables win over .env (compose, CI and tests set them on purpose).
load_dotenv()

import json
import logging
import os

from llama_index.core.ingestion import  IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.core.storage import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")

def get_doc_store():
    """
    Initialize document store from storage directory or memory.
    
    Returns:
        SimpleDocumentStore: Initialized document store
        
    Note:
        Falls back to in-memory store if storage directory doesn't exist
    """
    # If the storage directory is there, load the document store from it.
    # If not, set up an in-memory document store since we can't load from a directory that doesn't exist.
    if os.path.exists(STORAGE_DIR):
        return SimpleDocumentStore.from_persist_dir(STORAGE_DIR)
    else:
        return SimpleDocumentStore()


def run_pipeline(docstore, vector_store, documents):
    """
    Run document ingestion pipeline with configured transformations.
    
    Args:
        docstore: Document storage backend
        vector_store: Vector database connection
        documents: List of documents to process
        
    Returns:
        List of processed document nodes
        
    Steps:
        1. Split documents into chunks
        2. Generate embeddings
        3. Store in vector database
    """
    logger.info(f"embedding model {Settings.embed_model}\n\n\n")
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=Settings.chunk_size,
                chunk_overlap=Settings.chunk_overlap,
            ),
            Settings.embed_model,
        ],
        vector_store=vector_store,
    )

    # Run the ingestion pipeline and store the results
    nodes = pipeline.run(show_progress=True, documents=documents)

    return 
    

def persist_storage(docstore, vector_store):
    storage_context = StorageContext.from_defaults(
        docstore=docstore,
        vector_store=vector_store,
    )
    storage_context.persist(STORAGE_DIR)


def generate_datasource() -> dict:
    """Index DATA_DIR into the served table: the index step (see the module docstring). Returns its report."""
    from app.engine.ingest import run_index_step

    logger.info("Generate index for the provided data (the index step, as python -m app.engine.ingest)")
    report = run_index_step()
    logger.info(f"Finished generating the index: {json.dumps(report, sort_keys=True)}")
    return report


def print_all_env_variables():
    for key, value in os.environ.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    from app.engine.ingest import main

    raise SystemExit(main())
