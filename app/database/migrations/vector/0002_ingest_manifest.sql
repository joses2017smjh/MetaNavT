-- vector 0002: the ingest manifest (app/engine/ingest.py): one row per indexed file with its
-- content hash, chunk count and pipeline. VectorStoreManager.ensure_ingest_manifest used to
-- create it inline on every indexing run; the definition is unchanged, so a manifest that
-- code wrote is adopted with its rows.

CREATE TABLE IF NOT EXISTS {manifest} (
    path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    n_chunks INTEGER NOT NULL,
    pipeline TEXT NOT NULL DEFAULT '',
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
