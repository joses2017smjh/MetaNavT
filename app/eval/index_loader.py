"""Index a frozen corpus directory into an InMemoryHybridIndex."""

from __future__ import annotations

import json
from pathlib import Path

from app.chunking.contextual import prepend_context
from app.eval.hashing import content_hash
from app.eval.ingest import BENCH_POLICY, SKIP_NAMES, file_chunks, ingest_report, iter_corpus_files, plan_ingest, scan_directory
from app.retrieval.embedders import HashEmbedder, TfidfEmbedder
from app.retrieval.hybrid import Chunk, InMemoryHybridIndex
from app.retrieval.rerank import CrossEncoderReranker, OverlapReranker
from app.retrieval.router import QueryRouter
from app.retrieval.context import path_context

__all__ = [
    "SKIP_NAMES", "MTIME_MODES", "iter_corpus_files", "fixture_manifest", "mtime_provenance",
    "load_chunks", "load_chunks_with_report", "build_index",
]

# Where a chunk's mtime comes from. The staleness filter (app/graph/staleness.py) and the file
# graph's co-modified edges compare mtimes, so they decide benchmark rows. A git checkout sets
# every file's mtime to the moment it was written (on some filesystems the same whole second for
# every file), which is why the benchmark reads a frozen fixture's mtimes from its MANIFEST:
#   auto      the manifest when `root` is a frozen fixture (see fixture_manifest), else the disk
#   manifest  the manifest; an error when there is none
#   disk      os.stat st_mtime, as the served indexer (app/engine/ingest.py) always does
MTIME_MODES = ("auto", "manifest", "disk")
MANIFEST_NAME = "MANIFEST.json"


def fixture_manifest(root: Path) -> Path | None:
    """The frozen fixture's MANIFEST.json when `root` is its files directory, else None.

    The layout is bench/corpus/MANIFEST.json next to bench/corpus/files: a
    MANIFEST.json beside `root` whose `root` field names a directory with the
    same name as `root` and that lists every file with its intended mtime
    (written by bench/corpus/build.py). A copy of the fixture made the same way
    (the files directory and the MANIFEST beside it) is a fixture too.
    """
    candidate = Path(root).parent / MANIFEST_NAME
    try:
        manifest = json.loads(candidate.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        return None
    files = manifest["files"]
    if Path(str(manifest.get("root", ""))).name != Path(root).name:
        return None
    if not all(isinstance(e, dict) and "path" in e and "mtime" in e for e in files):
        return None
    return candidate


def _manifest_mtimes(manifest_path: Path) -> dict[str, float]:
    return {str(e["path"]): float(e["mtime"]) for e in json.loads(manifest_path.read_text())["files"]}


def mtime_provenance(root: Path, mtimes: str = "auto", *, base: Path | None = None) -> dict:
    """What a results header records about the mtimes a load of `root` uses.

    The manifest is named relative to `base` (the repository root in the
    harness: bench/corpus/MANIFEST.json), or by its parent directory's name.
    """
    if mtimes not in MTIME_MODES:
        raise ValueError(f"unknown mtimes mode {mtimes!r}; expected one of {MTIME_MODES}")
    manifest = fixture_manifest(root) if mtimes != "disk" else None
    if mtimes == "manifest" and manifest is None:
        raise ValueError(f"mtimes='manifest' but there is no fixture {MANIFEST_NAME} beside {root}")
    if manifest is None:
        return {"source": "disk", "field": "st_mtime"}
    try:
        name = manifest.resolve().relative_to(Path(base).resolve()).as_posix() if base is not None else None
    except ValueError:
        name = None
    return {"source": "manifest", "file": name or f"{manifest.parent.name}/{MANIFEST_NAME}", "field": "files[].mtime"}


def _stored_prefix(prefixes: dict[str, dict], chunk_id: str, raw: str) -> str:
    """The LLM-written prefix stored for this chunk ('' when its call failed: indexed without a prefix)."""
    entry = prefixes.get(chunk_id)
    if entry is None or entry.get("span_sha256") != content_hash(raw):
        raise ValueError(f"no stored prefix for chunk {chunk_id} with these bytes; re-run `make frontier-generations`")
    return entry["text"]


def load_chunks_with_report(
    root: Path, strategy: str = "auto", *, include_path_context: bool = False, contextual: bool = False, mtimes: str = "auto",
    prefixes: dict[str, dict] | None = None,
) -> tuple[list[Chunk], dict]:
    """Chunks for every accepted file plus the ingestion report (files seen, indexed, rejected by reason).

    Scanning and chunking are app.eval.ingest, shared with the served indexer
    (app/engine/ingest.py). The benchmark policy accepts every decodable,
    non-empty file; an undecodable file is counted, not skipped silently.

    `contextual` prepends a situating prefix to each chunk's indexed text
    (app/chunking/contextual.py). By default the prefix is the path-words
    heuristic of app/retrieval/context.py, labelled as such wherever its numbers
    appear. With `prefixes` (chunk id -> stored LLM-written prefix and the hash of
    the chunk's bytes, app.eval.frontier_llm.context_prefixes) the stored prefix is
    used instead; no model is called here. Citations still use `evidence_raw`.

    `mtimes` (MTIME_MODES): by default a frozen fixture's chunks carry the
    mtimes its MANIFEST records, so a fresh checkout scores like the one that
    recorded the results; any other directory keeps the disk's. The choice is
    in the report as `mtimes` (mtime_provenance). A fixture file the manifest
    does not list is an error, not a silent fallback to the disk.
    """
    root = Path(root)
    provenance = mtime_provenance(root, mtimes)
    listed = _manifest_mtimes(fixture_manifest(root)) if provenance["source"] == "manifest" else None
    scan = scan_directory(root, BENCH_POLICY)
    if listed is not None:
        unlisted = [s.path for s in scan.files if s.path not in listed]
        if unlisted:
            raise ValueError(f"{len(unlisted)} files under {root} are not in its {MANIFEST_NAME}: {unlisted[:5]}")
    chunks: list[Chunk] = []
    for source in scan.files:
        mtime = listed[source.path] if listed is not None else source.mtime
        for piece in file_chunks(source.path, source.text, strategy):
            indexed_text = piece.text
            if contextual:
                prefix = _stored_prefix(prefixes, piece.chunk_id, piece.raw) if prefixes is not None else path_context(source.path)
                indexed_text = prepend_context(piece.text, prefix)
            chunks.append(
                Chunk(
                    chunk_id=piece.chunk_id,
                    path=source.path,
                    text=indexed_text,
                    start_byte=piece.start_byte,
                    end_byte=piece.end_byte,
                    mtime=mtime,
                    content_hash=source.sha256,
                    metadata={"kind": piece.kind, "evidence_raw": piece.raw, "source_span_sha256": content_hash(piece.raw),
                              "text_is_verbatim": indexed_text == piece.raw,
                              **({"search_context": path_context(source.path)} if include_path_context else {})},
                )
            )
    # An in-memory build has no previous run: every accepted file is new.
    report = ingest_report(scan, plan_ingest(scan.hashes(), {}))
    report["chunks_embedded"] = len(chunks)
    report["mtimes"] = provenance
    return chunks, report


def load_chunks(
    root: Path, strategy: str = "auto", *, include_path_context: bool = False, contextual: bool = False, mtimes: str = "auto",
    prefixes: dict[str, dict] | None = None,
) -> list[Chunk]:
    return load_chunks_with_report(root, strategy, include_path_context=include_path_context, contextual=contextual, mtimes=mtimes,
                                   prefixes=prefixes)[0]


def build_index(
    root: Path,
    *,
    embedder_name: str = "hash",
    retrieve_k: int = 50,
    rerank_n: int = 8,
    enable_router: bool = True,
    enable_rerank: bool = True,
    reranker: str = "overlap",
    chunk_strategy: str = "auto",
    include_path_context: bool = False,
    contextual: bool = False,
    mtimes: str = "auto",
    prefixes: dict[str, dict] | None = None,
) -> InMemoryHybridIndex:
    chunks, ingest = load_chunks_with_report(
        root, strategy=chunk_strategy, include_path_context=include_path_context, contextual=contextual, mtimes=mtimes,
        prefixes=prefixes,
    )
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
