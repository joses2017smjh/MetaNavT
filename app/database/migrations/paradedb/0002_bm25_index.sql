-- paradedb 0002: the BM25 index app/database/sql/hybrid_paradedb.sql searches (text @@@ query,
-- paradedb.score), with the tokenizer settings VectorStoreManager._create_bm25_index used.
-- That method dropped the index and built it again at every start, removing it from under
-- any other replica; it is built once here, IF NOT EXISTS, and never dropped.
-- Not run by any test or results file: no image used here has pg_search (OD07).

CREATE INDEX IF NOT EXISTS "{table}_bm25_idx_on_text" ON {data_table}
USING bm25 (node_id, text)
WITH (
    key_field='node_id',
    text_fields='{"text": {"tokenizer": {"type": "default", "stemmer": "English"}}}'
);
