"""Index a frozen corpus directory into an InMemoryHybridIndex."""

from __future__ import annotations

from pathlib import Path

from app.chunking.structure import chunk_text
from app.eval.hashing import content_hash
from app.retrieval.embedders import HashEmbedder, TfidfEmbedder
from app.retrieval.hybrid import Chunk, InMemoryHybridIndex, chunk_id_for
from app.retrieval.rerank import CrossEncoderReranker, OverlapReranker
from app.retrieval.router import QueryRouter


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


def load_chunks(root: Path, strategy: str = "auto") -> list[Chunk]:
    root = Path(root)
    chunks: list[Chunk] = []
    for path in iter_corpus_files(root):
        rel = str(path.relative_to(root)).replace("\\", "/")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        st = path.stat()
        digest = content_hash(text)
        spans = chunk_text(text, path=rel, strategy=strategy)
        if not spans:
            spans = chunk_text(text, path=rel, strategy="fixed")
        for span in spans:
            cid = chunk_id_for(rel, span.start, span.end)
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    path=rel,
                    text=span.text,
                    start_byte=span.start,
                    end_byte=span.end,
                    mtime=st.st_mtime,
                    content_hash=digest,
                    metadata={"kind": span.kind},
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
) -> InMemoryHybridIndex:
    chunks = load_chunks(root, strategy=chunk_strategy)
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
    if enable_rerank:
        if reranker == "overlap":
            rerank_fn = OverlapReranker()
        elif reranker == "none":
            rerank_fn = None
        else:
            rerank_fn = CrossEncoderReranker(model_name=reranker)

    return InMemoryHybridIndex(
        chunks,
        embedder=embedder,
        retrieve_k=retrieve_k,
        rerank_n=rerank_n,
        router=QueryRouter(),
        rerank_fn=rerank_fn,
        enable_router=enable_router,
        enable_rerank=enable_rerank and rerank_fn is not None,
    )
