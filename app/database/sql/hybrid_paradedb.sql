-- Same shape as hybrid_tsrank.sql, with ParadeDB pg_search BM25 for the lexical
-- list (text @@@ query, paradedb.score). Selected automatically when the
-- pg_search extension is installed. Not exercised by CI, which runs
-- pgvector/pgvector:pg16 without ParadeDB; the result label says which ran.
WITH params AS (
    SELECT %(qvec)s::vector AS qvec
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
        SELECT t.node_id, paradedb.score(t.node_id) AS bm25_score
        FROM {table} t
        WHERE t.text @@@ %(query)s
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
