-- vector 0006: text_tsv, a stored generated column holding to_tsvector('english', text), the
-- value the lexical list of app/database/sql/hybrid_tsrank.sql ranks with ts_rank_cd. That
-- statement used to compute it for every matching row of every query; it is computed once per
-- row here, at insert, with the same text search configuration, so ranks and scores are the
-- same (tests/database/test_dense_parity.py compares the two on every gold question).
-- Adding a stored generated column rewrites the table under an exclusive lock: on a large
-- table, run it in a maintenance window.

ALTER TABLE {data_table} ADD COLUMN IF NOT EXISTS text_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
