"""Incremental, accountable indexing of DATA_DIR into pgvector (the served path).

build_retrieval_state() calls index_directory() on every API start when
INDEX_ON_START is set. One run, under a Postgres advisory lock:

  1. reconcile: count the rows per path and compare them with the manifest
     table (one row per indexed file: path, content hash, chunk count, chunker
     version + embedder); a listed file whose rows differ from its chunk count
     (rows deleted or added behind the manifest's back) counts as changed;
  2. scan DATA_DIR and sort every file into accepted or rejected-with-a-reason
     (app/eval/ingest.py, the same scan and chunker the benchmark loader uses);
  3. diff the accepted files' SHA-256 against the manifest;
  4. fail before writing anything if rejections exceed INGEST_MAX_REJECTED;
  5. delete the rows of removed files; for each new or changed file embed its
     chunks, delete its old rows by path, insert the new ones, then list it in
     the manifest;
  6. delete chunks whose path the manifest does not list;
  7. verify: the rows per path must now equal the manifest's chunk counts, or
     the run raises IngestUnreconciled instead of reporting success.

An unchanged re-run embeds nothing. The step can be interrupted anywhere: a
file is listed in the manifest only after its chunks are committed, and a
file's old rows are deleted before its new ones are inserted, so the next run
repeats the unfinished files and leaves one set of rows.

The report (counts of files seen, indexed, unchanged, removed and rejected by
reason, and the reconciliation) is returned, logged, and served by /health
under "indexing". "Failed" is not a counted category: a failure (the embedder
or the database raising, the rejection limit, a verification mismatch) aborts
the run with an exception. main.py's lifespan records it and retries the start
in the background: the process stays up, /health answers 503 with the error and
/api/retrieve answers 503 until a later attempt succeeds. A file the run had
not reached keeps its previous rows and manifest entry, and so does the file in
flight when the embedder raises (its old rows are deleted only after its chunks
are embedded). When the database raises after that file's old rows were
deleted, the file is no longer listed in the manifest, and may have no rows,
until the next run indexes it again as new.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Callable, Optional

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

from app.eval.ingest import (
    CHUNKER_VERSION,
    REJECTED_SAMPLE,
    SERVED_MAX_BYTES,
    SERVED_SUFFIXES,
    IngestPolicy,
    IngestRejected,
    IngestUnreconciled,
    SourceFile,
    file_chunks,
    plan_and_report,
    row_mismatches,
    scan_directory,
)

logger = logging.getLogger("uvicorn")

EmbedFn = Callable[[list], list]


def policy_from_env() -> IngestPolicy:
    """INGEST_SUFFIXES (comma-separated, "*" = every type) and INGEST_MAX_FILE_BYTES override the defaults."""
    raw = os.getenv("INGEST_SUFFIXES", "").strip()
    if raw == "*":
        suffixes = None
    elif raw:
        suffixes = frozenset(s if s.startswith(".") else f".{s}" for s in (part.strip().lower() for part in raw.split(",")) if s)
    else:
        suffixes = SERVED_SUFFIXES
    return IngestPolicy(suffixes=suffixes, max_bytes=int(os.getenv("INGEST_MAX_FILE_BYTES", SERVED_MAX_BYTES)), dedup=True)


def max_rejected_from_env() -> Optional[int]:
    """INGEST_MAX_REJECTED: fail the run when more files than this are rejected. Unset = count only."""
    raw = os.getenv("INGEST_MAX_REJECTED", "").strip()
    return int(raw) if raw else None


def node_id_for(path: str, start_byte: int, end_byte: int) -> str:
    """Deterministic row id: the SHA-1 the benchmark's chunk id truncates, without the path prefix.

    hybrid_*.sql breaks score ties by node_id. An id that starts with the path
    would order ties by directory name (configs/ before logs/), a ranking signal
    nobody chose; the bare digest keeps tie order unrelated to the path, as the
    random ids were before, and the same on every machine.
    """
    return hashlib.sha1(f"{path}:{start_byte}:{end_byte}".encode()).hexdigest()


def build_nodes(source: SourceFile, strategy: str = "auto") -> list[TextNode]:
    """One TextNode per chunk of a file, with the benchmark's chunk texts and byte ranges."""
    nodes = []
    for piece in file_chunks(source.path, source.text, strategy):
        metadata = {
            "path": source.path,  # relative to DATA_DIR, the bench's path form
            "chunk_id": piece.chunk_id,  # the benchmark's id for this chunk: path::<first 12 hex of the node id>
            "file_path": source.abs_path,
            "file_name": Path(source.path).name,
            "mtime": source.mtime,  # staleness clusters
            "content_hash": source.sha256,
            "start_byte": piece.start_byte,
            "end_byte": piece.end_byte,
            "kind": piece.kind,
            "text_is_verbatim": piece.text == piece.raw,
        }
        nodes.append(
            TextNode(
                id_=node_id_for(source.path, piece.start_byte, piece.end_byte),
                text=piece.text,
                metadata=metadata,
                # Embed and rank on the text alone, as the bench does; metadata stays out of the vectors.
                excluded_embed_metadata_keys=list(metadata),
                excluded_llm_metadata_keys=list(metadata),
                relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=source.path)},
            )
        )
    return nodes


def index_directory(
    vsm,
    data_dir: str,
    *,
    embed_fn: EmbedFn,
    policy: Optional[IngestPolicy] = None,
    max_rejected: Optional[int] = None,
    embedder: str = "",
) -> dict:
    """Bring the vector table in line with data_dir and return the run's report.

    embed_fn maps a list of chunk texts to a list of vectors; `embedder` names it.
    A file indexed by another chunker version or embedder counts as changed, and
    so does a listed file whose rows no longer equal its manifest chunk count.
    Raises IngestRejected (nothing written) when more than max_rejected files are
    rejected, and IngestUnreconciled when the rows per path do not equal the
    manifest's chunk counts after the run's writes.
    """
    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"DATA_DIR {data_dir!r} is not a directory")
    policy = policy or policy_from_env()
    pipeline = f"{CHUNKER_VERSION}|{embedder}"
    with vsm.ingest_lock() as lock:
        store = vsm.get_vector_store()  # the data table and the manifest are schema migrations (vector/0001, 0002)
        manifest = vsm.ingest_manifest()
        # Rows of unlisted paths are left to delete_unlisted_chunks below; a listed file whose rows do not
        # match its chunk count is indexed again, whatever its content hash says.
        at_start = [m for m in row_mismatches(_chunk_counts(manifest), vsm.rows_per_path()) if m["path"] in manifest]
        repair = {m["path"] for m in at_start}
        scan = scan_directory(root, policy)
        try:
            known = {p: row["content_hash"] if row["pipeline"] == pipeline and p not in repair else "" for p, row in manifest.items()}
            plan, report = plan_and_report(scan, known, max_rejected=max_rejected)
        except IngestRejected as exc:
            logger.error(f"Indexing refused: {exc}")
            raise
        by_path = {f.path: f for f in scan.files}
        chunks_deleted = sum(vsm.forget_file(path) for path in plan.removed)
        chunks_embedded = 0
        for path in plan.to_index:
            source = by_path[path]
            nodes = build_nodes(source)
            for node, vector in zip(nodes, embed_fn([n.text for n in nodes])):
                node.embedding = list(vector)
            chunks_deleted += vsm.forget_file(path)  # after embedding: an embedder failure leaves the old rows in place
            store.add(nodes)
            vsm.record_file(path, source.sha256, len(nodes), pipeline)
            chunks_embedded += len(nodes)
        unlisted = vsm.delete_unlisted_chunks()
        listed = vsm.ingest_manifest()
        after_run = row_mismatches(_chunk_counts(listed), vsm.rows_per_path())
        report.update(
            indexed=bool(plan.to_index),
            data_dir=data_dir,
            pipeline=pipeline,
            chunks_embedded=chunks_embedded,
            chunks_deleted=chunks_deleted,
            unlisted_chunks_deleted=unlisted,
            n_nodes=sum(row["n_chunks"] for row in listed.values()),
            reconciliation={
                "files_mismatched_at_start": len(at_start),  # each one indexed again (or removed) by this run
                "mismatched_sample": at_start[:REJECTED_SAMPLE],
                "files_mismatched_after_run": len(after_run),  # 0 in every returned report
            },
            lock=lock,
        )
        if after_run:
            logger.error(f"Indexing did not verify: {json.dumps(after_run[:REJECTED_SAMPLE])}")
            raise IngestUnreconciled(after_run, report)
    logger.info(f"Indexing report: {json.dumps(report, sort_keys=True)}")
    return report


def _chunk_counts(manifest: dict[str, dict]) -> dict[str, int]:
    return {path: row["n_chunks"] for path, row in manifest.items()}
