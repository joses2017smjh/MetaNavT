-- Same shape as hybrid_tsrank.sql, with ParadeDB pg_search BM25 for the lexical
-- list (text @@@ query, paradedb.score). Runs only with LEXICAL_BACKEND=paradedb,
-- never because pg_search happens to be installed (the default is ts_rank_cd).
-- Not exercised by CI, which runs pgvector/pgvector:pg16 without ParadeDB, nor by
-- any results file; the result label says which ran. Its dense and paths CTEs are
-- hybrid_tsrank.sql's, verbatim (tests/engine/test_vector_store_sql.py).
WITH params AS (
    SELECT %(qvec)s::vector AS qvec
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
        SELECT t.node_id, paradedb.score(t.node_id) AS bm25_score
        FROM {table} t
        WHERE t.text @@@ %(query)s
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
