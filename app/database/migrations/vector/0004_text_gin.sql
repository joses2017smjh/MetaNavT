-- migrate: no-transaction
-- vector 0004: GIN on to_tsvector('english', text), the expression the lexical list of
-- app/database/sql/hybrid_tsrank.sql filters on. Built CONCURRENTLY; VectorStoreManager.
-- ensure_text_index created it inline at every start before, under the same name, so a
-- database that has it keeps it. An invalid leftover is dropped first, as in 0003.

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_index WHERE indexrelid = to_regclass('{schema}."data_{table}_text_tsv_gin"') AND NOT indisvalid) THEN
        EXECUTE 'DROP INDEX {schema}."data_{table}_text_tsv_gin"';
    END IF;
END
$$;

CREATE INDEX CONCURRENTLY IF NOT EXISTS "data_{table}_text_tsv_gin" ON {data_table} USING gin (to_tsvector('english', text));
