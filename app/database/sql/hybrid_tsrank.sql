-- Hybrid retrieval in one round trip (Postgres + pgvector, no ParadeDB): the statement
-- for LEXICAL_BACKEND=ts_rank_cd, the default.
--
--   dense   : pgvector cosine distance (<=>), top-k. The planner may read it from the
--             HNSW index on the embedding column or with a sequential scan (on fixture
--             v1 it scans: doc/sql/hybrid_explain.txt). An HNSW scan returns at most
--             hnsw.ef_search rows, so the caller sets it for this statement's
--             transaction, in the same execute (VectorStoreManager.settings_sql).
--   lexical : websearch_to_tsquery('english', ...) over text_tsv, the stored
--             to_tsvector('english', text) (schema migration vector/0006), ranked by
--             ts_rank_cd, with the GIN index on that column (vector/0007)
--   paths   : on the router's lexical_path route only, the rows whose file name, path
--             or stem occur in the query, scored as InMemoryHybridIndex.search_path
--             scores chunks (the benchmark's path lookup for that route)
--   fused   : reciprocal rank fusion, score = sum 1 / (rrf_k + rank) with 1-based
--             ranks, the same formula as app/retrieval/fuse.py (tests/database/
--             test_hybrid_sql.py checks the two agree on identical input lists)
--
-- Parameters (psycopg2 named parameters; never written with their placeholder syntax in a
-- comment, where psycopg2 substitutes them too and a newline in a value would end the comment):
--             qvec       vector literal text, or NULL to run lexical-only
--             query      websearch text (the caller OR-joins the query terms so the
--                        lexical list has BM25-like OR semantics, not AND)
--             path_query the query text on the lexical_path route, else NULL (no path list)
--             k          depth of each list and of the fused result
--             rrf_k      RRF constant (60)
-- {table} is replaced by the qualified data table (schema.data_<table>).
--
-- The query vector and the tsquery are separate CTEs. params (the vector) is NOT
-- MATERIALIZED: inlined, it is a constant, which an index ordering operator needs on
-- its right; read from a materialized CTE it is a column, and the dense list could
-- never be read from the HNSW index (tests/database/test_dense_parity.py forces the
-- index and checks the plan). lexical_params stays MATERIALIZED, as the tsquery was
-- when both shared one CTE read twice. The lexical list can be read with a bitmap scan of
-- the GIN index on text_tsv (test_dense_parity.py checks it with sequential scans off);
-- on fixture v1's few rows the planner reads the table instead, which no longer computes
-- to_tsvector for each row.
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
        -- The cut at k orders equal distances by node_id. That key cannot go into the
        -- index-ordered scan below: an index ordering operator must match every ORDER
        -- BY key, so a second key would leave the HNSW index unused. The scan reads the
        -- 2k nearest rows instead and this level keeps k of them, so a tie at the cut
        -- is decided by node_id unless the tied rows run past the 2k read. This level
        -- sorts by distance ascending, as the scan does: an all-zero embedding has a NaN
        -- distance, which sorts after every number, and a descending score would put it
        -- first.
        SELECT node_id, 1 - distance AS dense_score
        FROM (
            SELECT t.node_id, t.embedding <=> p.qvec AS distance
            FROM {table} t, params p
            WHERE p.qvec IS NOT NULL
            ORDER BY t.embedding <=> p.qvec
            LIMIT 2 * %(k)s
        ) nearest
        ORDER BY distance, node_id
        LIMIT %(k)s
    ) s
),
lexical AS (
    SELECT node_id, bm25_score,
           row_number() OVER (ORDER BY bm25_score DESC, node_id) AS bm25_rank
    FROM (
        SELECT t.node_id, ts_rank_cd(t.text_tsv, p.tsq) AS bm25_score
        FROM {table} t, lexical_params p
        WHERE t.text_tsv @@ p.tsq
        ORDER BY bm25_score DESC, t.node_id
        LIMIT %(k)s
    ) s
),
paths AS (
    -- The router's lexical_path route only: the path_query parameter is the query text there and
    -- NULL on every other route, which leaves this list empty without reading the table. A row scores
    -- as InMemoryHybridIndex.search_path scores its chunk, on lowercased text: 5 when its file
    -- name occurs in the query, 3 when its whole path does, 2 when its file name without the
    -- last extension (Python's Path.stem) does, summed; a row that scores 0 is not listed.
    -- Equal scores are ordered by node_id, as every other tie in this statement (the
    -- in-memory list keeps corpus order). Shared verbatim by hybrid_*.sql.
    SELECT node_id, path_score,
           row_number() OVER (ORDER BY path_score DESC, node_id) AS path_rank
    FROM (
        SELECT node_id, path_score
        FROM (
            SELECT node_id,
                   CASE WHEN name <> '' AND strpos(q, name) > 0 THEN 5 ELSE 0 END
                 + CASE WHEN path <> '' AND strpos(q, path) > 0 THEN 3 ELSE 0 END
                 + CASE WHEN stem <> '' AND strpos(q, stem) > 0 THEN 2 ELSE 0 END AS path_score
            FROM (
                SELECT node_id, q, path, name, regexp_replace(name, '(.)\.[^.]+$', '\1') AS stem
                FROM (
                    SELECT t.node_id, lower(%(path_query)s::text) AS q,
                           lower(coalesce(t.metadata_->>'path', '')) AS path,
                           lower(regexp_replace(coalesce(t.metadata_->>'path', ''), '^.*/', '')) AS name
                    FROM {table} t
                    WHERE %(path_query)s::text IS NOT NULL
                ) named
            ) parts
        ) scored
        WHERE path_score > 0
        ORDER BY path_score DESC, node_id
        LIMIT %(k)s
    ) s
),
fused AS (
    SELECT COALESCE(d.node_id, l.node_id, pa.node_id) AS node_id,
           d.dense_score, d.dense_rank, l.bm25_score, l.bm25_rank, pa.path_score, pa.path_rank,
           COALESCE(1.0 / (%(rrf_k)s + d.dense_rank), 0)
         + COALESCE(1.0 / (%(rrf_k)s + l.bm25_rank), 0)
         + COALESCE(1.0 / (%(rrf_k)s + pa.path_rank), 0) AS rrf_score
    FROM dense d
    FULL OUTER JOIN lexical l ON d.node_id = l.node_id
    FULL OUTER JOIN paths pa ON pa.node_id = COALESCE(d.node_id, l.node_id)
)
SELECT f.node_id, t.text, t.metadata_,
       f.bm25_score, f.bm25_rank, f.dense_score, f.dense_rank, f.rrf_score,
       f.path_score, f.path_rank
FROM fused f
JOIN {table} t ON t.node_id = f.node_id
ORDER BY f.rrf_score DESC, f.node_id
LIMIT %(k)s;
