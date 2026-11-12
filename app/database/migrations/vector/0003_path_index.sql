-- migrate: no-transaction
-- vector 0003: the index the ingest step's per-file delete reads (metadata_->>'path'),
-- built CONCURRENTLY so writes to the table go on during the build. Created inline by
-- ensure_ingest_manifest before; a database that has it keeps it (IF NOT EXISTS). An index
-- left INVALID by an interrupted CONCURRENTLY build would satisfy IF NOT EXISTS, so the
-- block first drops one that is invalid and the next statement builds it again.

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_index WHERE indexrelid = to_regclass('{schema}."data_{table}_path_idx"') AND NOT indisvalid) THEN
        EXECUTE 'DROP INDEX {schema}."data_{table}_path_idx"';
    END IF;
END
$$;

CREATE INDEX CONCURRENTLY IF NOT EXISTS "data_{table}_path_idx" ON {data_table} ((metadata_->>'path'));
