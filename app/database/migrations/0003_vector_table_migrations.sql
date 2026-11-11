-- 0003: the ledger of the per-table sequences (directories under migrations/, such as vector/:
-- the data table, the ingest manifest and their indexes). A vector table's schema, name and
-- embedding dimension are configuration (PGVECTOR_SCHEMA, PGVECTOR_TABLE, EMBEDDING_DIM), so
-- those files are templates, applied once per table. Each row records one applied file of one
-- sequence (its directory name) for one table (vector_table is schema.data_table), with the
-- values it was rendered with (params: the embedding dimension, the vector extension's schema).

CREATE TABLE IF NOT EXISTS vector_table_migrations (
    sequence TEXT NOT NULL,
    vector_table TEXT NOT NULL,
    version INTEGER NOT NULL,
    name TEXT NOT NULL,
    checksum TEXT NOT NULL,
    params JSONB NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (sequence, vector_table, version)
);
