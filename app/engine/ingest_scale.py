"""Indexing and start-up cost at 1x, 10x and 100x fixture v1: hash embedder, live Postgres.

    PG_TEST_DSN=postgresql://... python -m app.engine.ingest_scale --label before      (make ingest-scale INGEST_SCALE_LABEL=before)

Each size is a temporary directory holding that many copies of bench/corpus/files. Copy 0 is the
fixture as it is; every later copy adds one line to each file (a comment, a JSON key, a CSV row) so
no two files have the same bytes, because the served scan rejects duplicate content. Each file's
mtime is the MANIFEST's plus its copy number, so the staleness clusters are the same on every run.
bench/corpus/files is only read.

For each size, in a vector table of its own, built by the schema migrations and dropped at the end:

  index_first              app/engine/ingest.index_directory on the empty table: every chunk is embedded
  index_unchanged          the same call again: nothing is embedded
  clusters                 app/engine/bootstrap.build_clusters, the staleness clusters start-up loads
  startup_index_on_start   app/engine/bootstrap.build_retrieval_state with INDEX_ON_START=true
  startup_no_index         the same with INDEX_ON_START unset

Each start-up builds its own VectorStoreManager (connection pool, dimension check), as a new process
would. Wall times are seconds measured in this process on this machine, the median of --repeats runs
(index_first runs once); they include the database round trips, not Python's imports or the HTTP
server. Next to them are counts that do not depend on the machine: files, chunks embedded, rows,
clusters, and the rows of chunk text each step read from the data table, counted on the client by a
cursor that sees every statement the process sends (one more, untimed run of the step).

--label names the rows this run writes: it replaces the rows that carry its label and keeps every
other row as written, so rows recorded before a change and after it sit in one file, each naming
the commit that produced it. The committed file is written only from a tree without uncommitted
changes. Vectors come from the hash embedder (a fallback: deterministic, no model), so the file
records indexing and start-up cost on this corpus, not retrieval quality or a real embedder's cost.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "bench" / "corpus" / "files"
MANIFEST = ROOT / "bench" / "corpus" / "MANIFEST.json"
OUT = ROOT / "bench" / "results" / "ingest_scale.json"
SCHEMA = "ingest-scale-v1"
SIZES = (1, 10, 100)
REPEATS = 3
HASH_DIM = 256
LABEL = re.compile(r"^[a-z0-9_-]{1,40}$")


# ---------------------------------------------------------------- the corpus


def perturb(rel: str, data: bytes, copy: int) -> bytes:
    """The bytes of copy `copy` of a fixture file: copy 0 unchanged, a later copy with one line added.

    The added line keeps the file's format, so each copy is chunked the way the original is, give or
    take that line: a comment in YAML, Slurm scripts and Python (at the top of a Python module, so it
    joins the code before the first definition), a key in a JSON object, a row as wide as the header
    in a CSV, and a plain line in Markdown and logs.
    """
    if copy == 0:
        return data
    text = data.decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    suffix = Path(rel).suffix.lower()
    if suffix == ".py":
        return (f"# copy {copy}{newline}" + text).encode("utf-8")
    if suffix == ".json":
        value = json.loads(text)
        if isinstance(value, dict):
            return (json.dumps({**value, "copy": copy}, indent=2) + "\n").encode("utf-8")
    if suffix == ".csv":
        width = len(text.split(newline, 1)[0].split(","))
        tail = "" if text.endswith(newline) else newline
        return (text + tail + f"copy {copy}" + "," * (width - 1) + newline).encode("utf-8")
    marker = f"copy {copy}" if suffix in {".md", ".out", ".log", ".txt"} else f"# copy {copy}"
    tail = "" if text.endswith(newline) else newline
    return (text + tail + marker + newline).encode("utf-8")


def build_corpus(dest: Path, copies: int) -> dict:
    """Write `copies` copies of the fixture under dest/copy_NNN/; return {"files", "bytes", "sha256"}.

    sha256 is over "path:sha256 of the bytes" of every written file in path order (the MANIFEST's
    aggregate form, of the raw bytes), so two runs can show they measured the same corpus.
    """
    import hashlib

    from app.eval.corpus import aggregate_sha256, load_manifest

    mtimes = {entry["path"]: float(entry["mtime"]) for entry in load_manifest(MANIFEST)["files"]}
    sources = sorted(p for p in FIXTURE.rglob("*") if p.is_file())
    entries, total = [], 0
    for copy in range(copies):
        for source in sources:
            rel = source.relative_to(FIXTURE).as_posix()
            data = perturb(rel, source.read_bytes(), copy)
            target = dest / f"copy_{copy:03d}" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            mtime = mtimes[rel] + copy
            os.utime(target, (mtime, mtime))
            entries.append({"path": target.relative_to(dest).as_posix(), "sha256": hashlib.sha256(data).hexdigest()})
            total += len(data)
    return {"files": len(entries), "bytes": total, "sha256": aggregate_sha256(entries)}


# ---------------------------------------------------------------- what start-up reads


@contextmanager
def count_text_reads(data_table: str) -> Iterator[dict]:
    """Count, on the client, the statements that select the `text` column of `data_table` and the rows they return.

    Every psycopg2 connection opened inside the block (the managers' pools, PGVectorStore's
    SQLAlchemy engine, the ingest lock) gets a cursor class that looks at each statement it runs.
    """
    import psycopg2
    import psycopg2.extensions

    counts = {"statements": 0, "rows": 0}
    reads_text = re.compile(r"\bselect\b(?:(?!\bfrom\b).)*\btext\b(?:(?!\bfrom\b).)*\bfrom\b[^;]*" + re.escape(data_table), re.I | re.S)

    class Recording(psycopg2.extensions.cursor):
        def execute(self, query, vars=None):
            result = super().execute(query, vars)
            statement = query.as_string(self) if hasattr(query, "as_string") else (query.decode() if isinstance(query, bytes) else str(query))
            if reads_text.search(statement):
                counts["statements"] += 1
                counts["rows"] += max(self.rowcount, 0)
            return result

    real = psycopg2.connect

    def connect(*args, **kwargs):
        kwargs.setdefault("cursor_factory", Recording)
        return real(*args, **kwargs)

    psycopg2.connect = connect
    try:
        yield counts
    finally:
        psycopg2.connect = real


# ---------------------------------------------------------------- one size


def _median_seconds(step: Callable[[], object], repeats: int) -> tuple[float, list[float], object]:
    samples, last = [], None
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        last = step()
        samples.append(time.perf_counter() - started)
    return round(statistics.median(samples), 4), [round(s, 4) for s in samples], last


def _versions(vsm) -> dict:
    with vsm.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SHOW server_version;")
            server = cur.fetchone()[0]
    version = vsm.pgvector_version
    return {"postgres": server.split()[0], "pgvector": ".".join(str(p) for p in version) if version else None}


def measure_size(dsn: str, copies: int, repeats: int) -> dict:
    """Index and start up over `copies` copies of the fixture in a fresh table; the row for this size."""
    from llama_index.core.settings import Settings

    from app.database import vector_store
    from app.database.migrate import drop_vector_table, migrate_all
    from app.database.vector_store_manager import VectorStoreManager
    from app.engine.bootstrap import build_clusters, build_retrieval_state
    from app.engine.ingest import index_directory
    from app.settings import init_embed_model

    table = "scale_" + uuid.uuid4().hex[:8]
    with tempfile.TemporaryDirectory(prefix="ingest_scale_") as tmp:
        corpus = Path(tmp) / "files"
        generated = build_corpus(corpus, copies)
        env = {
            "PG_CONNECTION_STRING": dsn, "PSYCOPG2_CONNECTION_STRING": dsn, "PGVECTOR_TABLE": table,
            "EMBEDDING_PROVIDER": "hash", "EMBEDDING_DIM": str(HASH_DIM), "RERANKER_MODEL": "none",
            "RETRIEVAL_MODE": "sql", "DATA_DIR": str(corpus), "INGEST_MAX_REJECTED": "0",
        }
        os.environ.update(env)
        for name in ("PGVECTOR_SCHEMA", "LEXICAL_BACKEND", "INGEST_SUFFIXES", "INGEST_MAX_FILE_BYTES", "ENABLE_STALENESS", "INDEX_ON_START"):
            os.environ.pop(name, None)
        init_embed_model()  # the hash embedder, as the API's start-up sets it
        embedder = str(getattr(Settings.embed_model, "model_name", "") or type(Settings.embed_model).__name__)
        vsm = VectorStoreManager(conn_string=dsn)
        try:
            with vsm.get_connection() as conn:
                migrate_all(conn, vsm.vector_table)  # the deploy step
            versions = _versions(vsm)

            def index(manager=vsm):
                return index_directory(manager, str(corpus), embed_fn=Settings.embed_model.get_text_embedding_batch, embedder=embedder, max_rejected=0)

            def counted(step) -> dict:
                """Run step(manager) once on a manager opened inside the count, so every connection it uses is seen."""
                with count_text_reads(vsm.data_table) as reads:
                    fresh = VectorStoreManager(conn_string=dsn)
                    try:
                        step(fresh)
                    finally:
                        fresh.close()
                return reads

            started = time.perf_counter()
            first = index()
            index_first_s = round(time.perf_counter() - started, 4)
            rows = vsm.count_nodes()
            unchanged_s, unchanged_samples, unchanged = _median_seconds(index, repeats)
            unchanged_reads = counted(index)
            clusters_s, clusters_samples, clusters = _median_seconds(lambda: build_clusters(vsm), repeats)
            cluster_reads = counted(build_clusters)

            def start(index_on_start: bool):
                def run():
                    if index_on_start:
                        os.environ["INDEX_ON_START"] = "true"
                    else:
                        os.environ.pop("INDEX_ON_START", None)
                    vector_store.get_vector_store_manager.cache_clear()
                    try:
                        return build_retrieval_state()
                    finally:
                        if vector_store.get_vector_store_manager.cache_info().currsize:
                            vector_store.get_vector_store_manager().close()
                        vector_store.get_vector_store_manager.cache_clear()
                        os.environ.pop("INDEX_ON_START", None)

                return run

            startup = {}
            for name, flag in (("startup_index_on_start", True), ("startup_no_index", False)):
                seconds, samples, state = _median_seconds(start(flag), repeats)
                with count_text_reads(vsm.data_table) as reads:
                    start(flag)()
                startup[name] = {
                    "seconds": seconds,
                    "samples": samples,
                    "chunks_embedded": (state.indexing or {}).get("chunks_embedded", 0),
                    "text_rows_read": reads["rows"],
                    "text_statements": reads["statements"],
                }
        finally:
            with vsm.get_connection() as conn:
                drop_vector_table(conn, vsm.vector_table)
            vsm.close()
    return {
        "size": f"{copies}x",
        "copies": copies,
        "corpus": generated,
        **versions,
        "files_seen": first["files_seen"],
        "files_rejected": first["files_rejected"],
        "rows": rows,
        "index_first": {"seconds": index_first_s, "files_indexed": first["files_indexed"], "chunks_embedded": first["chunks_embedded"]},
        "index_unchanged": {
            "seconds": unchanged_s,
            "samples": unchanged_samples,
            "files_unchanged": unchanged["files_unchanged"],
            "chunks_embedded": unchanged["chunks_embedded"],
            "text_rows_read": unchanged_reads["rows"],
        },
        "clusters": {
            "seconds": clusters_s,
            "samples": clusters_samples,
            "clusters": len(clusters),
            "members": sum(len(c.members) for c in clusters.values()),
            "text_rows_read": cluster_reads["rows"],
            "text_statements": cluster_reads["statements"],
        },
        **startup,
    }


# ---------------------------------------------------------------- the file


def worktree_clean(exclude: Path | None = None) -> bool:
    """True when git sees no change to a tracked file (the results file being rewritten aside)."""
    command = ["git", "status", "--porcelain", "--untracked-files=no", "--", "."]
    if exclude is not None:
        command.append(f":(exclude){exclude.resolve().relative_to(ROOT).as_posix()}")
    try:
        return subprocess.check_output(command, cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip() == ""
    except (subprocess.CalledProcessError, OSError, ValueError):
        return False


def merge(existing: dict | None, label: str, rows: list[dict]) -> dict:
    """The file with `label`'s rows replaced by `rows`; every other row is kept exactly as it was written."""
    kept = [row for row in (existing or {}).get("rows", []) if row.get("label") != label]
    order = {"before": 0, "after": 1}
    return {
        "schema": SCHEMA,
        "embedder": f"hash-{HASH_DIM} (fallback: deterministic hashing, no model)",
        "fixture_sha256": (existing or {}).get("fixture_sha256") or _fixture_sha256(),
        "corpus": "copy 0 of bench/corpus/files as it is, copies 1..n-1 with one line added to each file; mtimes from the MANIFEST plus the copy number",
        "timings": "seconds on the machine in each row's device block, median of the row's samples; index_first is one run",
        "rows": sorted(kept + rows, key=lambda row: (order.get(row["label"], 2), row["label"], row["copies"])),
    }


def _fixture_sha256() -> str:
    from app.eval.corpus import load_manifest

    return load_manifest(MANIFEST)["aggregate_sha256"]


def record(dsn: str, label: str, sizes=SIZES, repeats: int = REPEATS) -> list[dict]:
    """Measure every size and return the rows, each stamped with the label, commit, command and machine."""
    from app.eval.provenance import command_line, device_info, git_sha, scrub

    rows = []
    for copies in sizes:
        row = measure_size(dsn, int(copies), repeats)
        rows.append(scrub({
            "label": label,
            "git_sha": git_sha(ROOT),
            "worktree_clean": worktree_clean(OUT),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": command_line(),
            "device": device_info("cpu"),
            **row,
        }))
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True, help="names the rows this run writes (before, after); rows with other labels are kept")
    p.add_argument("--sizes", default=",".join(str(s) for s in SIZES), help="copies of the fixture per size, comma-separated")
    p.add_argument("--repeats", type=int, default=REPEATS)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    if not LABEL.match(args.label):
        p.error("--label: lowercase letters, digits, - and _")
    dsn = os.environ.get("PG_TEST_DSN") or os.environ.get("PG_CONNECTION_STRING")
    if not dsn:
        print("set PG_TEST_DSN (or PG_CONNECTION_STRING)")
        return 2
    if args.out.resolve() == OUT.resolve() and not worktree_clean(OUT):
        print("refusing to write bench/results/ingest_scale.json from a tree with uncommitted changes: commit first, so the rows name the code that produced them")
        return 2
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    rows = record(dsn, args.label, sizes, args.repeats)
    existing = json.loads(args.out.read_text()) if args.out.exists() else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merge(existing, args.label, rows), indent=2) + "\n")
    for row in rows:
        print(
            f"{row['label']:7s} {row['size']:>5s}  files {row['files_seen']:5d}  rows {row['rows']:5d}  "
            f"index first {row['index_first']['seconds']:8.3f} s  unchanged {row['index_unchanged']['seconds']:7.3f} s  "
            f"clusters {row['clusters']['seconds']:6.3f} s ({row['clusters']['text_rows_read']} text rows)  "
            f"start-up {row['startup_index_on_start']['seconds']:7.3f} s (INDEX_ON_START) / {row['startup_no_index']['seconds']:6.3f} s"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
