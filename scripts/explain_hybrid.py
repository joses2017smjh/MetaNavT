"""EXPLAIN (ANALYZE, BUFFERS) of the served hybrid statement -> doc/sql/.

    PG_TEST_DSN=postgresql://... python scripts/explain_hybrid.py [--query "..."] [--k 50]
    PG_TEST_DSN=postgresql://... python scripts/explain_hybrid.py --synthetic-rows 10000 --out doc/sql/hybrid_explain_synthetic_10k.txt

Default: runs against a database that already holds fixture v1 (docker compose
up, or the CI parity service) in the served table. --synthetic-rows N instead
loads N generated rows into a scratch table (dropped again at the end): seeded
random text, mostly made-up words with a few domain words, embedded with the hash
embedder. They exist so the planner sees a table of that size; they are not data,
and the lexical leg's cost on them says nothing about real text.

Either way the table gets the indexes the API's startup creates (GIN on the text,
HNSW on the embedding) and is ANALYZEd, the query vector comes from the hash
embedder (no model), and the statement runs with the per-transaction settings
hybrid_search applies (VectorStoreManager.search_settings), which the header lists.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SYNTHETIC_TABLE = "explain_synthetic"
DOMAIN_WORDS = (
    "learning rate run epoch loss config encoder checkpoint dataset baseline ablation result "
    "current fusion paper draft gpu batch schedule val_rmse"
).split()


def synthetic_documents(n: int, seed: int = 0) -> list[str]:
    """n seeded documents of 30-60 words: made-up words, each word a domain word with probability 0.02."""
    rng = random.Random(seed)
    syllables = [c + v for c in "bdfgklmnprstvz" for v in "aeiou"]
    vocab = sorted({"".join(rng.choice(syllables) for _ in range(rng.randint(2, 4))) for _ in range(5000)})
    docs = []
    for _ in range(n):
        words = [rng.choice(DOMAIN_WORDS) if rng.random() < 0.02 else rng.choice(vocab) for _ in range(rng.randint(30, 60))]
        if rng.random() < 0.02:
            words += ["run", str(rng.randint(1, 500))]
        docs.append(" ".join(words))
    return docs


def load_synthetic(vsm, n: int, dim: int) -> None:
    """Create the scratch table through PGVectorStore (the served schema) and insert n generated rows."""
    from psycopg2.extras import execute_values

    from app.retrieval.embedders import HashEmbedder

    with vsm.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP TABLE IF EXISTS "{vsm.schema_name}"."{vsm.data_table}";')
    vsm.get_vector_store()  # creates data_<table> with the columns and indexes LlamaIndex gives it
    docs = synthetic_documents(n)
    vectors = HashEmbedder(dim).encode(docs)
    rows = [
        (text, json.dumps({"path": f"synthetic/doc_{i:05d}.txt", "chunk_id": f"synthetic-{i:05d}"}), f"synthetic-{i:05d}", vsm.vector_literal(vec))
        for i, (text, vec) in enumerate(zip(docs, vectors))
    ]
    with vsm.get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                f'INSERT INTO "{vsm.schema_name}"."{vsm.data_table}" (text, metadata_, node_id, embedding) VALUES %s',
                rows,
                template="(%s, %s::json, %s, %s::vector)",
                page_size=500,
            )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--query", default="current learning rate for run 47")
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--synthetic-rows", type=int, default=0, help="load this many generated rows into a scratch table instead")
    p.add_argument("--out", type=Path, default=ROOT / "doc" / "sql" / "hybrid_explain.txt")
    args = p.parse_args()
    dsn = os.environ.get("PG_TEST_DSN") or os.environ.get("PG_CONNECTION_STRING")
    if not dsn:
        print("set PG_TEST_DSN (or PG_CONNECTION_STRING)")
        return 2
    os.environ.setdefault("PG_CONNECTION_STRING", dsn)
    os.environ.setdefault("PSYCOPG2_CONNECTION_STRING", dsn)
    os.environ.setdefault("EMBEDDING_DIM", "256")
    if args.synthetic_rows:
        os.environ["PGVECTOR_TABLE"] = SYNTHETIC_TABLE

    from app.database.vector_store_manager import VectorStoreManager
    from app.eval.provenance import git_sha, scrub
    from app.retrieval.embedders import HashEmbedder

    dim = int(os.environ["EMBEDDING_DIM"])
    vsm = VectorStoreManager(conn_string=dsn)
    if args.synthetic_rows:
        load_synthetic(vsm, args.synthetic_rows, dim)
    n = vsm.count_nodes()
    qvec = HashEmbedder(dim).encode([args.query])[0].tolist()
    vsm.ensure_text_index()
    vsm.ensure_hnsw()
    try:
        with vsm.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f'ANALYZE "{vsm.schema_name}"."{vsm.data_table}";')
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                cur.execute(
                    "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = %s AND tablename = %s ORDER BY indexname;",
                    (vsm.schema_name, vsm.data_table),
                )
                indexes = cur.fetchall()
        plan = vsm.explain_hybrid(args.query, qvec, k=args.k)
        settings = vsm.search_settings(args.k)
    finally:
        if args.synthetic_rows:
            with vsm.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f'DROP TABLE IF EXISTS "{vsm.schema_name}"."{vsm.data_table}";')
    pgvector = ".".join(str(part) for part in vsm.pgvector_version or ()) or "?"
    rows_note = (
        f"{n} SYNTHETIC rows: seeded generated text (scripts/explain_hybrid.py synthetic_documents), hash-embedded; not data"
        if args.synthetic_rows
        else f"{n} rows, fixture v1"
    )
    header = [
        f"-- EXPLAIN (ANALYZE, BUFFERS) of app/database/sql/hybrid_{'paradedb' if vsm.hybrid_backend == 'paradedb' else 'tsrank'}.sql",
        f"-- generated {datetime.now(timezone.utc).isoformat()} at git {git_sha(ROOT)} by scripts/explain_hybrid.py {scrub(' '.join(sys.argv[1:]))}".rstrip(),
        f"-- {scrub(version)}",
        f"-- pgvector {pgvector}; lexical backend {vsm.hybrid_backend}; table {vsm.schema_name}.{vsm.data_table} ({rows_note}); ANALYZE before EXPLAIN",
        f"-- query: {args.query!r}, k={args.k}, rrf_k=60, hash embedder dim {dim}",
        f"-- per-statement settings (VectorStoreManager.search_settings): {settings}",
        "-- indexes:",
        *[f"--   {name}: {ddl}" for name, ddl in indexes],
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(header) + plan + "\n")
    print("\n".join(header[:7]))
    print(plan[:1200])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
