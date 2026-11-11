-- vector 0001: the data table LlamaIndex's PGVectorStore reads and writes, in the shape its
-- own setup gives it (llama-index-vector-stores-postgres 0.5.5, no hybrid column, JSON
-- metadata). The API builds PGVectorStore with perform_setup=False, so this file is where the
-- table, the vector extension and PGVectorStore's ref_doc_id index come from.
-- IF NOT EXISTS throughout: a table PGVectorStore created before migrations existed is
-- adopted with its rows. The block at the end refuses one whose embedding column has another
-- dimension than EMBEDDING_DIM ({dim} here), which used to win silently at start-up.
-- tests/database/test_migrations.py compares this table with the one PGVectorStore creates.

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA {vector_schema};

CREATE TABLE IF NOT EXISTS {data_table} (
    id BIGSERIAL PRIMARY KEY,
    text VARCHAR NOT NULL,
    metadata_ JSON,
    node_id VARCHAR,
    embedding {vector_schema}.vector({dim})
);

CREATE INDEX IF NOT EXISTS "{table}_idx_1" ON {data_table} USING btree ((metadata_ ->> 'ref_doc_id'));

DO $$
DECLARE
    stored integer;
BEGIN
    SELECT a.atttypmod INTO stored FROM pg_attribute a
    WHERE a.attrelid = {data_table_literal}::regclass AND a.attname = 'embedding' AND NOT a.attisdropped;
    IF stored IS DISTINCT FROM {dim} THEN
        RAISE EXCEPTION 'table % has embedding dimension %, EMBEDDING_DIM is {dim}', {data_table_literal}, stored;
    END IF;
END
$$;
