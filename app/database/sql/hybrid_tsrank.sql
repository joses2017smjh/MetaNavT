-- Hybrid retrieval in one round trip (Postgres + pgvector, no ParadeDB): the statement
-- for LEXICAL_BACKEND=ts_rank_cd, the default.
--
--   dense   : pgvector cosine distance (<=>), top-k. The planner may read it from the
--             HNSW index on the embedding column or with a sequential scan (on fixture
--             v1 it scans: doc/sql/hybrid_explain.txt). An HNSW scan returns at most
--             hnsw.ef_search rows, so the caller sets it for this statement's
--             transaction, in the same execute (VectorStoreManager.settings_sql).
--   lexical : websearch_to_tsquery('english', ...) over to_tsvector('english', text),
--             ranked by ts_rank_cd, with the GIN index on the same expression
--   fused   : reciprocal rank fusion, score = sum 1 / (rrf_k + rank) with 1-based
--             ranks, the same formula as app/retrieval/fuse.py (tests/database/
--             test_hybrid_sql.py checks the two agree on identical input lists)
--
-- Parameters: %(qvec)s  vector literal text, or NULL to run lexical-only
--             %(query)s websearch text (the caller OR-joins the query terms so the
--                       lexical list has BM25-like OR semantics, not AND)
--             %(k)s     depth of each list and of the fused result
--             %(rrf_k)s RRF constant (60)
-- {table} is replaced by the qualified data table (schema.data_<table>).
--
-- The query vector and the tsquery are separate CTEs. params (the vector) is NOT
-- MATERIALIZED: inlined, it is a constant, which an index ordering operator needs on
-- its right; read from a materialized CTE it is a column, and the dense list could
-- never be read from the HNSW index (tests/database/test_dense_parity.py forces the
-- index and checks the plan). lexical_params stays MATERIALIZED, as the tsquery was
-- when both shared one CTE read twice: inlined as a constant, it leads the planner to
-- a sequential scan of fixture v1 that computes to_tsvector for every row, slower
-- than the GIN bitmap scan the lexical list keeps this way.
WITH params AS NOT MATERIALIZED (
    SELECT %(qvec)s::vector AS qvec
),
lexical_params AS MATERIALIZED (
    SELECT websearch_to_tsquery('english', %(query)s) AS tsq
),
dense AS (
    SELECT node_id, dense_score,
           row_number() OVER (ORDER BY dense_score DESC, node_id) AS dense_rank
    FROM (
        -- The cut at k orders equal scores by node_id. That key cannot go into the
        -- index-ordered scan below: an index ordering operator must match every ORDER
        -- BY key, so a second key would leave the HNSW index unused. The scan reads the
        -- 2k nearest rows instead and this level keeps k of them, so a tie at the cut
        -- is decided by node_id unless the tied rows run past the 2k read.
        SELECT node_id, 1 - distance AS dense_score
        FROM (
            SELECT t.node_id, t.embedding <=> p.qvec AS distance
            FROM {table} t, params p
            WHERE p.qvec IS NOT NULL
            ORDER BY t.embedding <=> p.qvec
            LIMIT 2 * %(k)s
        ) nearest
        ORDER BY dense_score DESC, node_id
        LIMIT %(k)s
    ) s
),
lexical AS (
    SELECT node_id, bm25_score,
           row_number() OVER (ORDER BY bm25_score DESC, node_id) AS bm25_rank
    FROM (
        SELECT t.node_id, ts_rank_cd(to_tsvector('english', t.text), p.tsq) AS bm25_score
        FROM {table} t, lexical_params p
        WHERE to_tsvector('english', t.text) @@ p.tsq
        ORDER BY bm25_score DESC, t.node_id
        LIMIT %(k)s
    ) s
),
fused AS (
    SELECT COALESCE(d.node_id, l.node_id) AS node_id,
           d.dense_score, d.dense_rank, l.bm25_score, l.bm25_rank,
           COALESCE(1.0 / (%(rrf_k)s + d.dense_rank), 0)
         + COALESCE(1.0 / (%(rrf_k)s + l.bm25_rank), 0) AS rrf_score
    FROM dense d
    FULL OUTER JOIN lexical l ON d.node_id = l.node_id
)
SELECT f.node_id, t.text, t.metadata_,
       f.bm25_score, f.bm25_rank, f.dense_score, f.dense_rank, f.rrf_score
FROM fused f
JOIN {table} t ON t.node_id = f.node_id
ORDER BY f.rrf_score DESC, f.node_id
LIMIT %(k)s;
