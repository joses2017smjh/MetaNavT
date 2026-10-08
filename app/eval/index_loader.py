"""Index a frozen corpus directory into an InMemoryHybridIndex."""

from __future__ import annotations

from pathlib import Path

from app.chunking.structure import chunk_text
from app.eval.hashing import content_hash
from app.retrieval.embedders import HashEmbedder, TfidfEmbedder
from app.retrieval.hybrid import Chunk, InMemoryHybridIndex, chunk_id_for
from app.retrieval.rerank import CrossEncoderReranker, OverlapReranker
from app.retrieval.router import QueryRouter
from app.retrieval.context import path_context


SKIP_NAMES = {".git", "__pycache__", ".pytest_cache"}


def iter_corpus_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_NAMES for part in path.parts):
            continue
        files.append(path)
    return files


def load_chunks(root: Path, strategy: str = "auto", *, include_path_context: bool = False) -> list[Chunk]:
    root = Path(root)
    chunks: list[Chunk] = []
    for path in iter_corpus_files(root):
        rel = str(path.relative_to(root)).replace("\\", "/")
        try:
            # Preserve physical CRLF bytes as well as Unicode for citations.
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            continue
        st = path.stat()
        digest = content_hash(text)
        spans = chunk_text(text, path=rel, strategy=strategy)
        if not spans:
            spans = chunk_text(text, path=rel, strategy="fixed")
        for span in spans:
            # Structure chunkers report character positions; citations use UTF-8 bytes.
            start_byte = len(text[:span.start].encode("utf-8"))
            end_byte = len(text[:span.end].encode("utf-8"))
            raw_span = text[span.start:span.end]
            cid = chunk_id_for(rel, start_byte, end_byte)
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    path=rel,
                    text=span.text,
                    start_byte=start_byte,
                    end_byte=end_byte,
                    mtime=st.st_mtime,
                    content_hash=digest,
                    metadata={"kind": span.kind, "evidence_raw": raw_span, "source_span_sha256": content_hash(raw_span),
                              "text_is_verbatim": span.text == raw_span,
                              **({"search_context": path_context(rel)} if include_path_context else {})},
                )
            )
    return chunks


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
    chunks = load_chunks(root, strategy=chunk_strategy, include_path_context=include_path_context)
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
    return index
