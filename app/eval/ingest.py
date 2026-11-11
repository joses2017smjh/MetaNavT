"""One ingestion path for the benchmark loader and the served API: scan, chunk, diff, account.

Pure functions (stdlib, app.chunking and the chunk-id helper): no database and
no LlamaIndex, so `make test-eval` covers them. Two callers:

  app/eval/index_loader.py   the in-memory benchmark index
  app/engine/ingest.py       the served pgvector index (API start)

Both call file_chunks(), so a chunk boundary the benchmark measures is the
boundary the API serves.

The accounting follows the robotics snapshot freeze (app/eval/robotics.py):
walk one sorted listing, confine every path to the root before reading it,
pass every file through explicit gates, count every rejection by reason, and
identify content by the SHA-256 of its raw bytes. A file is never dropped
without being counted:

  files_seen == files_indexed + files_unchanged + files_rejected

Confinement: a file whose real path (every symlink resolved) is not under the
root, or is under a directory the scan skips (.git and the caches), is
rejected as `outside_root` before its size or bytes are read, so a link to a
file elsewhere on the machine never reaches a chunk, an embedding or a search
result. A link that stays inside the scanned tree is read like any other file.
The walk does not descend into symlinked directories.

`files_removed` counts paths the manifest listed that are no longer accepted
(deleted on disk, or now rejected); their rows are deleted.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from app.chunking.structure import chunk_text
from app.eval.reindex import files_to_reindex, files_to_remove
from app.retrieval.hybrid import chunk_id_for

SKIP_NAMES = {".git", "__pycache__", ".pytest_cache"}
REPORT_SCHEMA = "ingest-report-v2"  # v2: the outside_root reason
CHUNKER = "app/chunking (structure-aware)"
CHUNKER_VERSION = "app.chunking/auto/v1"  # bump when chunk boundaries, texts or ids change: every file is then re-indexed
# Gate order is the order of this tuple; a file is counted under the first gate it fails.
# outside_root comes first: a file outside the root is rejected before its size or bytes are read.
REJECT_REASONS = ("outside_root", "unsupported_type", "too_large", "undecodable", "empty", "duplicate_content")
REJECTED_SAMPLE = 20  # paths listed in a report; the counts always cover every file

# Suffixes the served path indexes: the ones chunk_text routes by structure, plus
# plain-text experiment artifacts that take the fixed window.
SERVED_SUFFIXES = frozenset({
    ".csv", ".jsonl", ".py", ".md", ".rst", ".yaml", ".yml", ".json",
    ".txt", ".out", ".log", ".sbatch", ".sh", ".cfg", ".ini", ".toml", ".tex",
})
SERVED_MAX_BYTES = 1_048_576


@dataclass(frozen=True)
class IngestPolicy:
    """Which files a path accepts. None switches a gate off."""

    suffixes: frozenset | None = None  # lower-case suffixes with the dot; None = every type
    max_bytes: int | None = None  # None = no size cap
    dedup: bool = False  # reject a later file whose bytes equal an accepted one


# The benchmark loader's historical behaviour: every decodable, non-empty file.
BENCH_POLICY = IngestPolicy()
SERVED_POLICY = IngestPolicy(suffixes=SERVED_SUFFIXES, max_bytes=SERVED_MAX_BYTES, dedup=True)


@dataclass(frozen=True)
class SourceFile:
    path: str  # relative to the scanned root, forward slashes
    sha256: str  # of the raw bytes
    text: str
    mtime: float
    size_bytes: int
    abs_path: str


@dataclass(frozen=True)
class FileChunk:
    chunk_id: str
    start_byte: int
    end_byte: int
    text: str  # what is searched and embedded (a CSV row carries its header)
    raw: str  # the exact source slice the byte range addresses
    kind: str


@dataclass
class Scan:
    files: list[SourceFile] = field(default_factory=list)  # accepted, in path order
    rejected: list[dict] = field(default_factory=list)  # {"path", "reason"}, in path order

    @property
    def seen(self) -> int:
        return len(self.files) + len(self.rejected)

    def hashes(self) -> dict[str, str]:
        return {f.path: f.sha256 for f in self.files}


@dataclass(frozen=True)
class IngestPlan:
    new: list[str]
    changed: list[str]
    unchanged: list[str]
    removed: list[str]

    @property
    def to_index(self) -> list[str]:
        return sorted(self.new + self.changed)


class IngestRejected(RuntimeError):
    """More files were rejected than the configured limit allows; nothing was written."""

    def __init__(self, report: dict):
        self.report = report
        super().__init__(
            f"{report['files_rejected']} of {report['files_seen']} files rejected, limit {report['max_rejected']}: "
            f"{ {k: v for k, v in report['rejected_by_reason'].items() if v} }"
        )


def iter_corpus_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_NAMES for part in path.parts):
            continue
        files.append(path)
    return files


def confined_real_path(real_root: Path, path: Path) -> Path | None:
    """`path` with every symlink resolved when that stays inside the scanned tree, else None.

    The second layer of the robotics freeze's check (app/eval/robotics.py
    confined()); the first, no absolute path and no "..", holds by construction
    for a path the listing yielded. `real_root` is the root already resolved,
    so a root reached through a symlinked prefix is not mistaken for an escape.
    """
    try:
        rel = path.resolve().relative_to(real_root)
    except (OSError, RuntimeError, ValueError):  # outside the root, or a symlink loop
        return None
    if any(part in SKIP_NAMES for part in rel.parts):  # e.g. a link to .git/config
        return None
    return real_root / rel


def scan_directory(root: Path, policy: IngestPolicy = BENCH_POLICY) -> Scan:
    """Read every file under root once and sort it into accepted or rejected-with-a-reason."""
    root = Path(root)
    real_root = root.resolve()
    scan = Scan()
    seen_hashes: set[str] = set()
    for path in iter_corpus_files(root):
        rel = str(path.relative_to(root)).replace("\\", "/")
        real = confined_real_path(real_root, path)
        if real is None:  # before its size or bytes are read
            scan.rejected.append({"path": rel, "reason": "outside_root"})
            continue
        st = real.stat()  # stat and read the resolved path: a link retargeted after the check is not followed
        if policy.suffixes is not None and path.suffix.lower() not in policy.suffixes:
            scan.rejected.append({"path": rel, "reason": "unsupported_type"})
            continue
        if policy.max_bytes is not None and st.st_size > policy.max_bytes:
            scan.rejected.append({"path": rel, "reason": "too_large"})
            continue
        data = real.read_bytes()
        try:
            # Preserve physical CRLF bytes as well as Unicode for citations.
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            scan.rejected.append({"path": rel, "reason": "undecodable"})
            continue
        if not text.strip():  # the chunkers drop whitespace-only spans, so this file has no chunk
            scan.rejected.append({"path": rel, "reason": "empty"})
            continue
        digest = hashlib.sha256(data).hexdigest()
        if policy.dedup and digest in seen_hashes:
            scan.rejected.append({"path": rel, "reason": "duplicate_content"})
            continue
        seen_hashes.add(digest)
        scan.files.append(SourceFile(path=rel, sha256=digest, text=text, mtime=st.st_mtime, size_bytes=len(data), abs_path=str(path)))
    return scan


def file_chunks(rel: str, text: str, strategy: str = "auto") -> list[FileChunk]:
    """The one chunking path: app.chunking spans with UTF-8 byte offsets and deterministic ids."""
    spans = chunk_text(text, path=rel, strategy=strategy)
    if not spans:
        spans = chunk_text(text, path=rel, strategy="fixed")
    chunks = []
    for span in spans:
        # Structure chunkers report character positions; citations use UTF-8 bytes.
        start_byte = len(text[:span.start].encode("utf-8"))
        end_byte = len(text[:span.end].encode("utf-8"))
        chunks.append(
            FileChunk(
                chunk_id=chunk_id_for(rel, start_byte, end_byte),
                start_byte=start_byte,
                end_byte=end_byte,
                text=span.text,
                raw=text[span.start:span.end],
                kind=span.kind,
            )
        )
    return chunks


def plan_ingest(current: Mapping[str, str], manifest: Mapping[str, str]) -> IngestPlan:
    """Diff the accepted files ({path: sha256}) against the manifest of what is indexed."""
    stale = files_to_reindex(current, manifest)
    return IngestPlan(
        new=[p for p in stale if p not in manifest],
        changed=[p for p in stale if p in manifest],
        unchanged=sorted(p for p in current if manifest.get(p) == current[p]),
        removed=files_to_remove(current, manifest),
    )


def ingest_report(scan: Scan, plan: IngestPlan, *, max_rejected: int | None = None) -> dict:
    """Counts for one run. The caller adds what only it knows (rows written, data dir, lock)."""
    by_reason = {reason: 0 for reason in REJECT_REASONS}
    for row in scan.rejected:
        by_reason[row["reason"]] += 1
    return {
        "schema": REPORT_SCHEMA,
        "chunker": CHUNKER,
        "files_seen": scan.seen,
        "files_indexed": len(plan.to_index),
        "files_unchanged": len(plan.unchanged),
        "files_removed": len(plan.removed),
        "files_rejected": len(scan.rejected),
        "indexed_new": len(plan.new),
        "indexed_changed": len(plan.changed),
        "rejected_by_reason": by_reason,
        "rejected_sample": scan.rejected[:REJECTED_SAMPLE],
        "max_rejected": max_rejected,
    }


def enforce_rejection_limit(report: dict) -> None:
    """Raise IngestRejected when a limit is set and the run rejected more files than it allows."""
    limit = report.get("max_rejected")
    if limit is not None and report["files_rejected"] > limit:
        raise IngestRejected(report)


def plan_and_report(scan: Scan, manifest: Mapping[str, str], *, max_rejected: int | None = None) -> tuple[IngestPlan, dict]:
    """Decide what one indexing run does. Raises IngestRejected before the caller writes anything."""
    plan = plan_ingest(scan.hashes(), manifest)
    report = ingest_report(scan, plan, max_rejected=max_rejected)
    enforce_rejection_limit(report)
    return plan, report
