-- 0005: the grant ledger and the audit trail of operator capabilities, for API replicas that share them
-- (METANAVIT_APPROVAL_LEDGER=postgres; app/mcp/postgres_ledger.py). A capability is consumed by
-- committing its grant row, in a transaction of its own, before the action runs. grant_id is the primary
-- key: of two verifiers presenting one capability, in any process on this database, exactly one insert
-- commits and the other's unique violation is read as already consumed. The stdio MCP servers keep a
-- SQLite ledger file (METANAVIT_APPROVAL_LEDGER_FILE) with the same two tables.

CREATE TABLE IF NOT EXISTS approval_grants (
    grant_id TEXT PRIMARY KEY,
    review_sha256 TEXT NOT NULL,
    actor TEXT NOT NULL,
    consumed_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS approval_audit (
    id BIGSERIAL PRIMARY KEY,
    at TIMESTAMPTZ NOT NULL DEFAULT now(),
    plan_id TEXT NOT NULL,
    action TEXT NOT NULL,
    result TEXT NOT NULL,
    detail TEXT
);
