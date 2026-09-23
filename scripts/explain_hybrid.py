"""EXPLAIN (ANALYZE, BUFFERS) for the hybrid query on the fixture -> doc/sql/hybrid_explain.txt.

    PG_TEST_DSN=postgresql://... python scripts/explain_hybrid.py [--query "..."] [--k 50]

Runs against a database that already holds fixture v1 (docker compose up, or the
CI parity service); the hash embedder produces the query vector, so no model.
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--query", default="current learning rate for run 47")
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--out", type=Path, default=ROOT / "doc" / "sql" / "hybrid_explain.txt")
    args = p.parse_args()
    dsn = os.environ.get("PG_TEST_DSN") or os.environ.get("PG_CONNECTION_STRING")
    if not dsn:
        print("set PG_TEST_DSN (or PG_CONNECTION_STRING)")
        return 2
    os.environ.setdefault("PG_CONNECTION_STRING", dsn)
    os.environ.setdefault("PSYCOPG2_CONNECTION_STRING", dsn)
    os.environ.setdefault("EMBEDDING_DIM", "256")

    from app.database.vector_store_manager import VectorStoreManager
    from app.eval.provenance import git_sha
    from app.retrieval.embedders import HashEmbedder

    vsm = VectorStoreManager(conn_string=dsn)
    n = vsm.count_nodes()
    qvec = HashEmbedder(int(os.environ["EMBEDDING_DIM"])).encode([args.query])[0].tolist()
    vsm.ensure_text_index()
    vsm.ensure_hnsw()
    with vsm.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            row = cur.fetchone()
            pgvector = row[0] if row else "?"
            cur.execute(
                "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = %s AND tablename = %s ORDER BY indexname;",
                (vsm.schema_name, vsm.data_table),
            )
            indexes = cur.fetchall()
    plan = vsm.explain_hybrid(args.query, qvec, k=args.k)
    header = [
        f"-- EXPLAIN (ANALYZE, BUFFERS) of app/database/sql/hybrid_{'paradedb' if vsm.hybrid_backend == 'paradedb' else 'tsrank'}.sql",
        f"-- generated {datetime.now(timezone.utc).isoformat()} at git {git_sha(ROOT)} by scripts/explain_hybrid.py",
        f"-- {version}",
        f"-- pgvector {pgvector}; lexical backend {vsm.hybrid_backend}; table {vsm.schema_name}.{vsm.data_table} ({n} rows, fixture v1)",
        f"-- query: {args.query!r}, k={args.k}, rrf_k=60, hash embedder dim {os.environ['EMBEDDING_DIM']}",
        "-- indexes:",
        *[f"--   {name}: {ddl}" for name, ddl in indexes],
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(header) + plan + "\n")
    print("\n".join(header[:6]))
    print(plan[:1200])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
