-- vector 0008: what the staleness clusters (app/graph/staleness.cluster_versions) need from each indexed file,
-- on its manifest row: the file's mtime when it was indexed and the node id of its last chunk (the member id a
-- cluster names). The indexer writes both with the rest of the row for every new or changed file, and start-up
-- builds the clusters from the manifest instead of reading every chunk's text back (app/engine/bootstrap.py).
-- A manifest written before this file gets both from its rows' metadata (mtime, and the row inserted last for each
-- path); no chunk text is read. Adding a nullable column without a default does not rewrite the table.

ALTER TABLE {manifest} ADD COLUMN IF NOT EXISTS mtime DOUBLE PRECISION;
ALTER TABLE {manifest} ADD COLUMN IF NOT EXISTS member_node_id TEXT;

UPDATE {manifest} m
SET mtime = d.mtime, member_node_id = d.node_id
FROM (
    SELECT DISTINCT ON (metadata_->>'path') metadata_->>'path' AS path, (metadata_->>'mtime')::double precision AS mtime, node_id
    FROM {data_table}
    WHERE metadata_->>'path' IS NOT NULL
    ORDER BY metadata_->>'path', id DESC
) d
WHERE m.path = d.path AND m.member_node_id IS NULL;
