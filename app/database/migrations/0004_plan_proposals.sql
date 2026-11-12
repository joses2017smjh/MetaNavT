-- 0004: the file plans proposed through /api/plans (app/api/routers/plans.py, PlanTable), shared by every
-- API replica on this database; they used to live in one process's memory.
-- One row per plan: the move (src, dst), the canonical review the operator signs (as text, byte for byte:
-- a capability carries the SHA-256 of these bytes) and that hash, the session it was proposed in, the
-- identity of the root it was proposed over, and the decision. Every request reads its plan by id, under
-- the root identity of the process's DATA_DIR; an approval rebuilds the review from src, dst and session.
-- plan_decisions (0001) stays the log of every approve / reject attempt.

CREATE TABLE IF NOT EXISTS plan_proposals (
    plan_id TEXT PRIMARY KEY,
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    review TEXT NOT NULL,
    review_sha256 TEXT NOT NULL,
    session TEXT NOT NULL,
    root_identity TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending_approval' CHECK (status IN ('pending_approval', 'applied', 'rejected')),
    created_at TIMESTAMPTZ NOT NULL,
    decided_at TIMESTAMPTZ,
    actor TEXT,
    note TEXT
);

-- GET /api/plans: the newest plans of one root.
CREATE INDEX IF NOT EXISTS plan_proposals_root_created_idx ON plan_proposals (root_identity, created_at DESC);
