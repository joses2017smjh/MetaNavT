-- migrate: no-transaction
-- vector 0005: HNSW on the embedding column (cosine, the <=> of the dense list), m = 16,
-- ef_construction = 64. The name carries the parameters: a later change of m or
-- ef_construction is a new file that builds an index under the new name and drops this one,
-- where IF NOT EXISTS on a fixed name (ensure_hnsw at every start, before) never changed an
-- existing index. Built CONCURRENTLY; then the index ensure_hnsw made (same parameters, the
-- old fixed name) is dropped CONCURRENTLY. Each statement can run again after an
-- interruption: the first drops a half-built index of the new name, the second rebuilds it.

DROP INDEX CONCURRENTLY IF EXISTS {schema}."{table}_embedding_hnsw_m16_efc64";

CREATE INDEX CONCURRENTLY "{table}_embedding_hnsw_m16_efc64" ON {data_table}
    USING hnsw (embedding {vector_schema}.vector_cosine_ops) WITH (m = 16, ef_construction = 64);

DROP INDEX CONCURRENTLY IF EXISTS {schema}."{table}_embedding_hnsw";
