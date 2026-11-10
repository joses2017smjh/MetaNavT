-- 0001 baseline: the tables the application used to create inline at startup.
--   plan_decisions                                 app/api/routers/plans.py (DecisionLog)
--   indexed_files, directory_processing_results    app/database/index_manager.py (IndexManager)
-- IF NOT EXISTS throughout: a database created by the pre-migration code is adopted
-- unchanged (rows untouched) and recorded as version 1.
-- The vector table (data_<PGVECTOR_TABLE>) is created by PGVectorStore and is not managed here.

CREATE TABLE IF NOT EXISTS plan_decisions (
    id BIGSERIAL PRIMARY KEY,
    plan_id TEXT NOT NULL,
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    action TEXT NOT NULL,
    actor TEXT,
    note TEXT,
    decided_at TIMESTAMPTZ NOT NULL,
    result TEXT NOT NULL,
    payload JSONB
);

CREATE TABLE IF NOT EXISTS indexed_files (
    file_path TEXT NOT NULL,
    process_name TEXT NOT NULL,
    process_version TEXT NOT NULL,
    mtime BIGINT NOT NULL,
    data BYTEA,
    content_hash TEXT,
    PRIMARY KEY (file_path, process_name, process_version)
);

-- content_hash was added after the table first shipped, with an inline ALTER at startup.
-- Keep that step so an indexed_files table from before the column converges to this shape.
ALTER TABLE indexed_files ADD COLUMN IF NOT EXISTS content_hash TEXT;

CREATE TABLE IF NOT EXISTS directory_processing_results (
    dir_path TEXT NOT NULL,
    process_name TEXT NOT NULL,
    process_version TEXT NOT NULL,
    is_applicable BOOLEAN NOT NULL,
    mtime BIGINT NOT NULL,
    PRIMARY KEY (dir_path, process_name, process_version)
);
