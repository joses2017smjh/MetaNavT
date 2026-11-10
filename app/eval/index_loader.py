"""Index a frozen corpus directory into an InMemoryHybridIndex."""

from __future__ import annotations

from pathlib import Path

from app.eval.hashing import content_hash
from app.eval.ingest import BENCH_POLICY, SKIP_NAMES, file_chunks, ingest_report, iter_corpus_files, plan_ingest, scan_directory
from app.retrieval.embedders import HashEmbedder, TfidfEmbedder
from app.retrieval.hybrid import Chunk, InMemoryHybridIndex
from app.retrieval.rerank import CrossEncoderReranker, OverlapReranker
from app.retrieval.router import QueryRouter
from app.retrieval.context import path_context

__all__ = ["SKIP_NAMES", "iter_corpus_files", "load_chunks", "load_chunks_with_report", "build_index"]


def load_chunks_with_report(root: Path, strategy: str = "auto", *, include_path_context: bool = False) -> tuple[list[Chunk], dict]:
    """Chunks for every accepted file plus the ingestion report (files seen, indexed, rejected by reason).

    Scanning and chunking are app.eval.ingest, shared with the served indexer
    (app/engine/ingest.py). The benchmark policy accepts every decodable,
    non-empty file; an undecodable file is counted, not skipped silently.
    """
    root = Path(root)
    scan = scan_directory(root, BENCH_POLICY)
    chunks: list[Chunk] = []
    for source in scan.files:
        for piece in file_chunks(source.path, source.text, strategy):
            chunks.append(
                Chunk(
                    chunk_id=piece.chunk_id,
                    path=source.path,
                    text=piece.text,
                    start_byte=piece.start_byte,
                    end_byte=piece.end_byte,
                    mtime=source.mtime,
                    content_hash=source.sha256,
                    metadata={"kind": piece.kind, "evidence_raw": piece.raw, "source_span_sha256": content_hash(piece.raw),
                              "text_is_verbatim": piece.text == piece.raw,
                              **({"search_context": path_context(source.path)} if include_path_context else {})},
                )
            )
    # An in-memory build has no previous run: every accepted file is new.
    report = ingest_report(scan, plan_ingest(scan.hashes(), {}))
    report["chunks_embedded"] = len(chunks)
    return chunks, report


def load_chunks(root: Path, strategy: str = "auto", *, include_path_context: bool = False) -> list[Chunk]:
    return load_chunks_with_report(root, strategy, include_path_context=include_path_context)[0]


def build_index(
    root: Path,
    *,
    embedder_name: str = "tfidf",
    retrieve_k: int = 50,
    rerank_n: int = 8,
    enable_router: bool = True,
    enable_rerank: bool = True,
    reranker: str = "overlap",
    chunk_strategy: str = "auto",
    include_path_context: bool = False,
) -> InMemoryHybridIndex:
    chunks, ingest = load_chunks_with_report(root, strategy=chunk_strategy, include_path_context=include_path_context)
    if embedder_name == "hash":
        embedder = HashEmbedder()
    elif embedder_name == "tfidf":
        embedder = TfidfEmbedder()
    elif embedder_name.startswith("st:"):
        from app.retrieval.embedders import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(embedder_name.split(":", 1)[1])
    elif embedder_name == "st":
        from app.retrieval.embedders import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
    else:
        raise ValueError(f"unknown embedder {embedder_name}")

    rerank_fn = None
    reranker_loaded = False
    reranker_fallback = None
    if enable_rerank:
        if reranker == "overlap":
            rerank_fn = OverlapReranker()
            reranker_fallback = None
        elif reranker in {"rankgpt", "listwise"}:
            from app.retrieval.rankgpt import RankGPTReranker

            rerank_fn = RankGPTReranker()
            reranker_loaded = rerank_fn.complete is not None
            reranker_fallback = None if reranker_loaded else "overlap-listwise"
        elif reranker == "none":
            rerank_fn = None
        else:
            rerank_fn = CrossEncoderReranker(model_name=reranker)
            reranker_loaded = rerank_fn.model is not None
            reranker_fallback = None if reranker_loaded else "overlap"

    index = InMemoryHybridIndex(
        chunks,
        embedder=embedder,
        retrieve_k=retrieve_k,
        rerank_n=rerank_n,
        router=QueryRouter(),
        rerank_fn=rerank_fn,
        enable_router=enable_router,
        enable_rerank=enable_rerank and rerank_fn is not None,
    )
    index.reranker_name = reranker
    index.reranker_loaded = reranker_loaded
    index.reranker_fallback = reranker_fallback
    index.ingest_report = ingest
    return index
