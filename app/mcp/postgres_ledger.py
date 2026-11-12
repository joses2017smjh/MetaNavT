"""The grant ledger and the audit trail in Postgres, shared by the API's replicas (METANAVIT_APPROVAL_LEDGER=postgres).

Tables approval_grants and approval_audit (migration 0005), written through the API's connection pool. A capability
is consumed by committing its grant row, in a transaction of its own, before the action runs. The grant id is the
primary key, so of two verifiers presenting one capability, in one process or in several on the database, exactly
one insert commits and the other's unique violation (SQLSTATE 23505) is read as already consumed: the rule the SQLite
ledger file applies on one host, for every host on the database. Any other database error is raised as it is, never
read as consumed or as granted, and the action does not run. The stdio MCP servers have no database and keep the
SQLite ledger (app/mcp/approvals.py).

Imported only when METANAVIT_APPROVAL_LEDGER=postgres (verifier_from_env) and by the tests: the evaluation profile
runs the approval benchmark without psycopg2.
"""
from __future__ import annotations

import time

from app.mcp.approvals import ApprovalError

UNIQUE_VIOLATION = "23505"  # SQLSTATE of a duplicate key


class PostgresLedger:
    """What ApprovalVerifier writes to a ledger (consume, audit), in the database `database` connects to."""

    def __init__(self, database):
        """`database`: get_connection(), a context manager yielding a psycopg2 connection that commits when the
        block ends cleanly and rolls back on an exception (DatabaseManager: the API's vector store manager)."""
        self.database = database

    def consume(self, grant: dict, instant: float) -> None:
        """Commit the grant's row; ApprovalError (check "ledger") when its id is already there."""
        duplicate = None
        with self.database.get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        "INSERT INTO approval_grants (grant_id, review_sha256, actor, consumed_at) VALUES (%s, %s, %s, to_timestamp(%s));",
                        (grant["grant_id"], grant["review_sha256"], grant["actor"], instant),
                    )
                except Exception as exc:
                    if getattr(exc, "pgcode", None) != UNIQUE_VIOLATION:
                        raise
                    conn.rollback()  # an expected refusal, not a database failure
                    duplicate = exc
        if duplicate is not None:
            raise ApprovalError("approval already consumed", check="ledger") from duplicate

    def audit(self, plan_id: str, action: str, result: str, detail: str = "") -> None:
        with self.database.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO approval_audit (at, plan_id, action, result, detail) VALUES (to_timestamp(%s), %s, %s, %s, %s);",
                    (time.time(), plan_id, action, result, detail[:500]),
                )
