-- Hybrid retrieval in one round trip (Postgres + pgvector, no ParadeDB).
--
--   dense   : pgvector cosine distance (<=>), top-k, served by the HNSW index
--   lexical : websearch_to_tsquery('english', ...) over to_tsvector('english', text),
--             ranked by ts_rank_cd, served by the GIN index on the same expression
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
WITH params AS (
    SELECT %(qvec)s::vector AS qvec,
           websearch_to_tsquery('english', %(query)s) AS tsq
),
dense AS (
    SELECT node_id, dense_score,
           row_number() OVER (ORDER BY dense_score DESC, node_id) AS dense_rank
    FROM (
        SELECT t.node_id, 1 - (t.embedding <=> p.qvec) AS dense_score
        FROM {table} t, params p
        WHERE p.qvec IS NOT NULL
        ORDER BY t.embedding <=> p.qvec
        LIMIT %(k)s
    ) s
),
lexical AS (
    SELECT node_id, bm25_score,
           row_number() OVER (ORDER BY bm25_score DESC, node_id) AS bm25_rank
    FROM (
        SELECT t.node_id, ts_rank_cd(to_tsvector('english', t.text), p.tsq) AS bm25_score
        FROM {table} t, params p
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
