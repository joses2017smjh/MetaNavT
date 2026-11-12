-- migrate: no-transaction
-- vector 0007: GIN on the stored text_tsv column (0006), the column the lexical list filters
-- on; the name carries the text search configuration it was computed with. Built
-- CONCURRENTLY; an invalid leftover is dropped first, as in 0003. Then the GIN index on the
-- expression (0004), which nothing reads any more, is dropped CONCURRENTLY.

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_index WHERE indexrelid = to_regclass('{schema}."data_{table}_text_tsv_english_gin"') AND NOT indisvalid) THEN
        EXECUTE 'DROP INDEX {schema}."data_{table}_text_tsv_english_gin"';
    END IF;
END
$$;

CREATE INDEX CONCURRENTLY IF NOT EXISTS "data_{table}_text_tsv_english_gin" ON {data_table} USING gin (text_tsv);

DROP INDEX CONCURRENTLY IF EXISTS {schema}."data_{table}_text_tsv_gin";
