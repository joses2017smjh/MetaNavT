"""Indexing reports for four consecutive runs over a temporary copy of fixture v1.

    PG_TEST_DSN=postgresql://... python -m app.engine.ingest_runs      (make ingest-runs)

  first_start        an empty table: every file is chunked, embedded and written
  unchanged_restart  the same directory again: nothing is embedded
  edit_one_file      one file edited: only its chunks are embedded again
  delete_one_file    one file deleted: its rows are removed

Each run is app/engine/ingest.index_directory, the function the API calls on
start, against a live Postgres and a table of its own that is dropped at the
end. The frozen fixture is copied first; bench/corpus/files is never written.
Vectors come from the hash embedder (a fallback: deterministic, no model), so
the file records what was indexed, not retrieval quality.
tests/database/test_incremental_index.py regenerates the runs and compares
them with the committed file.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bench" / "results" / "ingest_runs.json"
EDITED = "configs/run_047.yaml"
DELETED = "logs/run_046.out"
HASH_DIM = 256


def record_runs(dsn: str) -> list[dict]:
    """Run the four steps and return [{"run", "report", "texts_sent_to_embedder", "rows_in_table"}]."""
    os.environ["PG_CONNECTION_STRING"] = dsn
    os.environ["PSYCOPG2_CONNECTION_STRING"] = dsn
    os.environ["PGVECTOR_TABLE"] = "ingestruns_" + uuid.uuid4().hex[:8]
    os.environ["EMBEDDING_DIM"] = str(HASH_DIM)

    from app.database.vector_store_manager import VectorStoreManager
    from app.engine.ingest import index_directory
    from app.eval.ingest import SERVED_POLICY
    from app.retrieval.embedders import HashEmbedder

    embedder = HashEmbedder(HASH_DIM)
    sent: list[str] = []

    def embed(texts):
        sent.extend(texts)
        return embedder.encode(texts).tolist()

    vsm = VectorStoreManager(conn_string=dsn)
    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "files"
        shutil.copytree(ROOT / "bench" / "corpus" / "files", corpus)

        def step(name: str) -> None:
            sent.clear()
            report = index_directory(vsm, str(corpus), embed_fn=embed, policy=SERVED_POLICY, max_rejected=0, embedder=f"hash-{HASH_DIM}")
            report["data_dir"] = "temporary copy of bench/corpus/files"
            report["lock"] = {"waited": report["lock"]["waited"]}  # the wait time differs on every run
            runs.append({"run": name, "report": report, "texts_sent_to_embedder": len(sent), "rows_in_table": vsm.count_nodes()})

        try:
            step("first_start")
            step("unchanged_restart")
            with (corpus / EDITED).open("a", encoding="utf-8") as handle:
                handle.write("# edited by app.engine.ingest_runs\n")
            step("edit_one_file")
            (corpus / DELETED).unlink()
            step("delete_one_file")
        finally:
            with vsm.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f'DROP TABLE IF EXISTS "{vsm.schema_name}"."{vsm.data_table}";')
                    cur.execute(f'DROP TABLE IF EXISTS "{vsm.schema_name}"."{vsm.manifest_table}";')
            vsm.close()
    return runs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    dsn = os.environ.get("PG_TEST_DSN") or os.environ.get("PG_CONNECTION_STRING")
    if not dsn:
        print("set PG_TEST_DSN (or PG_CONNECTION_STRING)")
        return 2

    from app.eval.corpus import load_manifest
    from app.eval.provenance import git_sha

    runs = record_runs(dsn)
    blob = {
        "git_sha": git_sha(ROOT),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": "python -m app.engine.ingest_runs",
        "corpus_sha256": load_manifest(ROOT / "bench" / "corpus" / "MANIFEST.json")["aggregate_sha256"],
        "embedder": f"hash-{HASH_DIM} (fallback: deterministic hashing, no model)",
        "edited": EDITED,
        "deleted": DELETED,
        "runs": runs,
    }
    args.out.write_text(json.dumps(blob, indent=2) + "\n")
    for row in runs:
        r = row["report"]
        print(
            f"{row['run']:18s} seen {r['files_seen']:3d}  indexed {r['files_indexed']:3d}  unchanged {r['files_unchanged']:3d}  "
            f"removed {r['files_removed']:3d}  rejected {r['files_rejected']:3d}  embedded {row['texts_sent_to_embedder']:3d} chunks  rows {row['rows_in_table']:3d}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
