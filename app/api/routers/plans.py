"""Human-in-the-loop file plans: propose, approve, reject. Nothing moves without approval.

POST /api/plans                 {"src": "...", "dst": "..."} -> a pending MovePlan (no file touched)
GET  /api/plans                 the plans proposed over this DATA_DIR, newest first (at most MAX_LISTED_PLANS)
GET  /api/plans/{id}            one plan
POST /api/plans/{id}/approve    requires an externally issued operator capability; never mints approval
POST /api/plans/{id}/reject     mark rejected; the file is never touched

Backed by app.mcp.filesystem.FilesystemTools (the same gate the MCP tools use) over DATA_DIR, and by two
Postgres tables: the plans (plan_proposals, migration 0004) and a decision log (plan_decisions): every
approve / reject is written to the log before the action runs and updated with its result. If the log
cannot be written, the action is refused (503) rather than run unrecorded.

Every request reads its plan by id from plan_proposals and no saved plan is kept in a process between
requests (a proposal FilesystemTools refuses, such as a move onto an existing file, still leaves its
unsaved MovePlan in this process's FilesystemTools: propose_move stores the plan before building the
review that refuses it), so every API replica on the database, and a restarted one, serves the same
plans: a plan proposed on one replica is listed, viewed, approved or rejected on another, and its status
(pending_approval, applied, rejected) is the row's. A row holds the move, the canonical review the operator signs (its text and
SHA-256), the session it was proposed in, the identity of the root it was proposed over, and the decision
(when, by whom, the note). An approval loads the row into this process's FilesystemTools and runs its
gate unchanged: the review is rebuilt from the row's move and session and from the files as this process
sees them, must equal the stored review, and the capability must be signed over it. Plans are read under
the root identity of this process's DATA_DIR (its path, device and inode), so replicas share plans when
they see DATA_DIR at the same path on the same filesystem; a replica that sees it under another path or
device lists none of them and answers 404. A review also binds each file's device and inode, so a file
replaced since the review is refused as stale. A decision holds its row's lock until the
new status is saved, so an approval and a rejection of one plan, or two approvals, sent to two replicas
are decided one after the other.

A database failure answers 503 with a fixed text; its cause (which can name tables, hosts and ports) goes
to the server log only, and an operating system error's text (file names) is replaced by a fixed one in
the answer and in the decision log (told()). A record that fails to be written after the fact (the audit
event of a move that has run, the decision log's copy of the result of a saved decision) is logged and the
answer is the decision. build_plan_store(None), without a database (the API tests),
keeps the rows in a MemoryPlans object instead of the table.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.mcp.approvals import REFUSAL_CHECKS, review_hash, verifier_from_env
from app.mcp.filesystem import ApprovalRequired, FilesystemTools, MovePlan, refusal_check

plans_router = r = APIRouter()
logger = logging.getLogger("uvicorn")

MAX_RECENT_DECISIONS = 500  # upper bound on ?limit= for GET /decisions/recent
MAX_LISTED_PLANS = 500  # GET /api/plans answers with at most this many plans, the newest

# What a client is told when a database call fails; the cause is logged, never sent.
STORE_UNAVAILABLE = "plan store unavailable"
LOG_UNAVAILABLE = "decision log unavailable, action refused"
NOT_RECORDED = "the approval could not be recorded; nothing was applied"


def told(exc: BaseException, fallback: str | None) -> str | None:
    """What a client is told of a refused or failed file action, in the answer and in the decision log (which GET
    /api/plans/decisions/recent serves). A text this repository wrote (an exception raised without an errno: a
    check's error, "reviewed source is missing") goes as it is. An operating system error's text names files
    (absolute paths among them) and calls, and does not: the fixed description of the check that refused it
    (REFUSAL_CHECKS) when there is one, else `fallback`. The exception's own text goes to the server log only."""
    if getattr(exc, "errno", None) is None:
        return str(exc)
    check = refusal_check(exc)
    return REFUSAL_CHECKS[check] if check else fallback


class PlanCreate(BaseModel):
    src: str = Field(..., min_length=1, max_length=1024)
    dst: str = Field(..., min_length=1, max_length=1024)


class PlanView(BaseModel):
    plan_id: str
    src: str
    dst: str
    status: str  # pending_approval | approved | applied | rejected
    created_at: float
    decided_at: Optional[float] = None
    actor: Optional[str] = None
    note: Optional[str] = None
    review: dict[str, Any] | None = None


class Decision(BaseModel):
    actor: str = Field("web-ui", max_length=128)
    note: Optional[str] = Field(None, max_length=1000)
    approval_token: Optional[str] = Field(None, max_length=8192)


@dataclass
class PlanRow:
    """One proposed move: what plan_proposals stores (the field order is the table's SELECT order)."""

    plan_id: str
    src: str
    dst: str
    review: str  # the canonical review text as registered: byte for byte what a capability's hash covers
    review_sha256: str
    session: str  # the session of the FilesystemTools object that proposed it
    root_identity: str
    created_at: float
    status: str = "pending_approval"  # pending_approval | applied | rejected
    decided_at: Optional[float] = None
    actor: Optional[str] = None
    note: Optional[str] = None

    def view(self) -> PlanView:
        return PlanView(plan_id=self.plan_id, src=self.src, dst=self.dst, status=self.status, created_at=self.created_at,
                        decided_at=self.decided_at, actor=self.actor, note=self.note, review=json.loads(self.review))

    def move_plan(self) -> MovePlan:
        return MovePlan(plan_id=self.plan_id, src=self.src, dst=self.dst, created_at=self.created_at, session=self.session)


class MemoryPlans:
    """The plans of one process (a store built without a database, and the API tests): what PlanTable keeps in
    Postgres, with the same calls. Decisions are made one at a time."""

    def __init__(self):
        self._rows: dict[str, PlanRow] = {}
        self._lock = threading.Lock()
        self._deciding = threading.Lock()

    def add(self, row: PlanRow) -> None:
        with self._lock:
            self._rows[row.plan_id] = replace(row)

    def get(self, plan_id: str, root_identity: str) -> PlanRow | None:
        with self._lock:
            row = self._rows.get(plan_id)
            return replace(row) if row is not None and row.root_identity == root_identity else None

    def list(self, root_identity: str, limit: int) -> list[PlanRow]:
        with self._lock:
            rows = [replace(row) for row in self._rows.values() if row.root_identity == root_identity]
        return sorted(rows, key=lambda row: row.created_at, reverse=True)[:limit]

    @contextmanager
    def deciding(self, plan_id: str, root_identity: str) -> Iterator[PlanRow | None]:
        """The plan's row for one decision: changes to it are saved when the block ends without an exception."""
        with self._deciding:
            row = self.get(plan_id, root_identity)
            yield row
            if row is not None:
                with self._lock:
                    self._rows[plan_id] = replace(row)


class PlanTable:
    """plan_proposals in Postgres (migration 0004), shared by every replica on the database.

    Reads and inserts use the manager's connection pool. A decision takes the row's lock (SELECT ... FOR UPDATE)
    on a connection of its own and holds it until the new status is committed; meanwhile the decision log, the
    grant ledger and the audit trail write through the pool, one connection at a time, so a decision never waits
    for a second pool connection while it holds one. Closing that connection without a commit (an exception, or a
    process that dies) rolls the decision back and releases the lock.
    """

    TABLE = "plan_proposals"
    COLUMNS = ("plan_id, src, dst, review, review_sha256, session, root_identity, extract(epoch FROM created_at)::float8, "
               "status, extract(epoch FROM decided_at)::float8, actor, note")

    def __init__(self, database, connect=None):
        """`database`: get_connection() (a pooled connection) and conn_string, as the vector store manager has;
        `connect`: opens the connection a decision locks its row on (default psycopg2.connect(conn_string))."""
        self.database = database
        self._connect = connect or self._dedicated_connection

    def _dedicated_connection(self):
        import psycopg2

        return psycopg2.connect(self.database.conn_string)

    def add(self, row: PlanRow) -> None:
        with self.database.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {self.TABLE} (plan_id, src, dst, review, review_sha256, session, root_identity, status, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, to_timestamp(%s));",
                    (row.plan_id, row.src, row.dst, row.review, row.review_sha256, row.session, row.root_identity, row.status,
                     row.created_at),
                )

    def get(self, plan_id: str, root_identity: str) -> PlanRow | None:
        with self.database.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT {self.COLUMNS} FROM {self.TABLE} WHERE plan_id = %s AND root_identity = %s;", (plan_id, root_identity))
                found = cur.fetchone()
        return PlanRow(*found) if found else None

    def list(self, root_identity: str, limit: int) -> list[PlanRow]:
        with self.database.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT {self.COLUMNS} FROM {self.TABLE} WHERE root_identity = %s ORDER BY created_at DESC, plan_id LIMIT %s;",
                            (root_identity, int(limit)))
                return [PlanRow(*found) for found in cur.fetchall()]

    @contextmanager
    def deciding(self, plan_id: str, root_identity: str) -> Iterator[PlanRow | None]:
        """The plan's row, locked for one decision: changes to it are committed when the block ends without an exception."""
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT {self.COLUMNS} FROM {self.TABLE} WHERE plan_id = %s AND root_identity = %s FOR UPDATE;",
                            (plan_id, root_identity))
                found = cur.fetchone()
            row = PlanRow(*found) if found else None
            before = None if row is None else replace(row)
            yield row
            if row is not None and row != before:
                with conn.cursor() as cur:
                    cur.execute(
                        f"UPDATE {self.TABLE} SET status = %s, decided_at = to_timestamp(%s), actor = %s, note = %s WHERE plan_id = %s;",
                        (row.status, row.decided_at, row.actor, row.note, row.plan_id),
                    )
            conn.commit()
        finally:
            conn.close()


class PlanStore:
    """FilesystemTools (the gate), the plans (PlanTable, or MemoryPlans without a database) and the decision log."""

    def __init__(self, tools: FilesystemTools, log: "DecisionLog | None", plans: "PlanTable | MemoryPlans | None" = None):
        self.tools = tools
        self.log = log
        self.plans = plans if plans is not None else MemoryPlans()

    def _root_identity(self) -> str:
        try:
            return self.tools.root_identity
        except OSError as exc:
            logger.error(f"plan store: DATA_DIR cannot be read: {type(exc).__name__}: {exc}")
            raise HTTPException(status_code=503, detail="DATA_DIR cannot be read") from exc

    @staticmethod
    def _stored(what: str, call, *args):
        """Run one call on the plans; a failure is a 503 with a fixed text, its cause logged here."""
        try:
            return call(*args)
        except Exception as exc:  # noqa: BLE001 - refused with a fixed text, the cause logged
            logger.error(f"plan store: {what} failed: {type(exc).__name__}: {exc}")
            raise HTTPException(status_code=503, detail=STORE_UNAVAILABLE) from exc

    def create(self, src: str, dst: str) -> PlanView:
        try:
            proposal = self.tools.propose_move(src, dst)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"source not found: {told(exc, src)}") from exc
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=told(exc, "the destination already exists")) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=400, detail=told(exc, "the path was refused")) from exc
        plan_id = proposal["plan_id"]
        try:
            proposed = self.tools.plans[plan_id]
            review = self.tools.registered_review(plan_id)
            row = PlanRow(plan_id=plan_id, src=proposed.src, dst=proposed.dst, review=review,
                          review_sha256=review_hash(json.loads(review)), session=proposed.session,
                          root_identity=proposal["review"]["root_identity"], created_at=proposed.created_at)
        finally:
            self.tools.forget_plan(plan_id)  # the stored row is the only copy
        self._stored("saving a plan", self.plans.add, row)
        return row.view()

    def view(self, plan_id: str) -> PlanView:
        row = self._stored("reading a plan", self.plans.get, plan_id, self._root_identity())
        if row is None:
            raise HTTPException(status_code=404, detail=f"unknown plan {plan_id}")
        return row.view()

    def list(self) -> list[PlanView]:
        return [row.view() for row in self._stored("listing plans", self.plans.list, self._root_identity(), MAX_LISTED_PLANS)]

    def decide(self, plan_id: str, action: str, actor: str, note: Optional[str], approval_token: str | None = None) -> PlanView:
        root_identity = self._root_identity()
        try:
            with self.plans.deciding(plan_id, root_identity) as row:
                if row is None:
                    raise HTTPException(status_code=404, detail=f"unknown plan {plan_id}")
                if row.status in {"applied", "rejected"}:
                    raise HTTPException(status_code=409, detail=f"plan {plan_id} is already {row.status}")
                record = {"plan_id": plan_id, "src": row.src, "dst": row.dst, "action": action, "actor": actor, "note": note,
                          "decided_at": time.time(), "result": "pending"}
                if self.log is not None:
                    try:
                        self.log.write(record)  # intent first: an action never runs unrecorded
                    except Exception as exc:  # noqa: BLE001 - refused, not hidden
                        logger.error(f"decision log: writing the intent failed: {type(exc).__name__}: {exc}")
                        raise HTTPException(status_code=503, detail=LOG_UNAVAILABLE) from exc
                if action == "approve":
                    record["result"] = self._apply(row, approval_token, record)
                    row.status = "applied"
                else:
                    record["result"] = row.status = "rejected"
                row.decided_at, row.actor, row.note = record["decided_at"], actor, note
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - the plans table: reading or locking the row, or saving the decision
            logger.error(f"plan store: deciding plan {plan_id} failed: {type(exc).__name__}: {exc}")
            raise HTTPException(status_code=503, detail=STORE_UNAVAILABLE) from exc
        if self.log is not None:
            try:
                self.log.update(record)
            except Exception as exc:  # noqa: BLE001 - the decision is saved; only the log's copy of its result is missing
                logger.error(f"decision log: the result of {action} on plan {plan_id} ({record['result']}) was not "
                             f"written; its intent row reads pending: {type(exc).__name__}: {exc}")
        return self.view(plan_id)

    def _apply(self, row: PlanRow, approval_token: str | None, record: dict[str, Any]) -> str:
        """Load the row into FilesystemTools, run its gate and the move; the result for the decision log."""
        plan = row.move_plan()
        self.tools.load_move_plan(plan, row.review)
        try:
            return self.tools.apply_plan(row.plan_id, approval_token=approval_token).get("status", "applied")
        except (ApprovalRequired, PermissionError) as exc:
            text = self._result(record, "refused: ", exc, told(exc, "the file system refused the move"))
            raise HTTPException(status_code=403, detail=text) from exc
        except FileExistsError as exc:
            text = self._result(record, "refused: ", exc, told(exc, "the destination already exists"))
            raise HTTPException(status_code=409, detail=text) from exc
        except OSError as exc:
            text = told(exc, None)
            text = self._result(record, "failed: ", exc, f"move failed: {text}" if text else "move failed")
            raise HTTPException(status_code=500, detail=text) from exc
        except Exception as exc:  # noqa: BLE001 - not a refusal: the grant ledger's or the audit trail's database
            if plan.applied:  # the move ran; only the audit event written after it failed
                logger.error(f"plan {row.plan_id} applied, but its audit event was not written: {type(exc).__name__}: {exc}")
                return "applied"
            logger.error(f"approval of plan {row.plan_id} not recorded: {type(exc).__name__}: {exc}")
            record["result"] = f"failed: {type(exc).__name__}"
            if self.log is not None:
                self.log.update(record)
            raise HTTPException(status_code=503, detail=NOT_RECORDED) from exc
        finally:
            self.tools.forget_plan(row.plan_id)

    def _result(self, record: dict[str, Any], prefix: str, exc: BaseException, text: str) -> str:
        """Record a refused or failed approval: the decision log, which GET /api/plans/decisions/recent serves, gets
        the client's text (told()); the exception's own text goes to the server log. Returns the client's text."""
        if text != str(exc):
            logger.warning(f"plan {record['plan_id']}: {record['action']} {prefix.strip(': ')}: {type(exc).__name__}: {exc}")
        record["result"] = prefix + text
        if self.log is not None:
            self.log.update(record)
        return text


class DecisionLog:
    """plan_decisions table in Postgres, written through the vector store manager's pool: one row per approve or
    reject attempt, from any replica, written before the action and updated with its result."""

    TABLE = "plan_decisions"

    def __init__(self, vsm):
        self.vsm = vsm
        self.check_schema()

    def check_schema(self) -> None:
        """Refuse (MigrationsPending) unless the database's schema migrations are all applied.

        plan_decisions and plan_proposals are created by migrations 0001 and 0004 (app/database/migrations/),
        which the deploy step applies (`python -m app.database.migrate`); this only reads the ledger.
        """
        from app.database.migrate import require_current

        with self.vsm.get_connection() as conn:
            require_current(conn)

    def write(self, record: dict[str, Any]) -> None:
        """Insert the attempt; its row id goes into record["log_id"], which update() needs."""
        with self.vsm.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {self.TABLE} (plan_id, src, dst, action, actor, note, decided_at, result, payload) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;",
                    (record["plan_id"], record["src"], record["dst"], record["action"], record.get("actor"), record.get("note"),
                     datetime.fromtimestamp(record["decided_at"], tz=timezone.utc), record["result"], json.dumps(record)),
                )
                record["log_id"] = int(cur.fetchone()[0])
            conn.commit()

    def update(self, record: dict[str, Any]) -> None:
        """Set this attempt's result. It used to rewrite every row of the plan and action, so an earlier refusal
        read as the later attempt's result."""
        with self.vsm.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {self.TABLE} SET result = %s, payload = %s WHERE id = %s;",
                    (record["result"], json.dumps(record), record["log_id"]),
                )
            conn.commit()

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.vsm.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT plan_id, src, dst, action, actor, decided_at, result FROM {self.TABLE} ORDER BY id DESC LIMIT %s;", (limit,))
                return [dict(zip(("plan_id", "src", "dst", "action", "actor", "decided_at", "result"), row)) for row in cur.fetchall()]


def build_plan_store(vsm=None, root: str | None = None) -> PlanStore:
    """The API's plan store over `root` (DATA_DIR): plans and decisions in the database `vsm` connects to (the
    vector store manager: get_connection() and conn_string), or in this process only when `vsm` is None.

    The operator's capabilities are verified as configured (app/mcp/approvals.verifier_from_env); with
    METANAVIT_APPROVAL_LEDGER=postgres their grants are consumed in this database, the one every replica shares.
    """
    root = root or os.getenv("DATA_DIR", "data")
    verifier = verifier_from_env(Path(root), database=vsm)
    tools = FilesystemTools(root=root, approval_verifier=verifier)  # allow_apply stays False: only an operator capability applies
    if vsm is None:
        return PlanStore(tools, None)
    return PlanStore(tools, DecisionLog(vsm), PlanTable(vsm))


def public_startup_error(error: str | None) -> str:
    """What a client is told of a failed start: an approval configuration error as written (texts for the
    operator, naming no path, table or address), any other error by its class only; the server log has it all."""
    if not error:
        return "plan store not initialised"
    kind = error.split(":", 1)[0].strip()
    return error if kind == "ApprovalError" else f"{kind} (the server log has the reason)"


def _store(request: Request) -> PlanStore:
    store = getattr(request.app.state, "plans", None)
    if store is None:
        detail = public_startup_error(getattr(request.app.state, "startup_error", None))
        raise HTTPException(status_code=503, detail=f"plans unavailable: {detail}")
    return store


@r.get("/", response_model=list[PlanView])
async def list_plans(request: Request) -> list[PlanView]:
    return await run_in_threadpool(_store(request).list)


@r.post("/", response_model=PlanView, status_code=201)
async def create_plan(body: PlanCreate, request: Request) -> PlanView:
    return await run_in_threadpool(_store(request).create, body.src, body.dst)


@r.get("/decisions/recent")
async def recent_decisions(request: Request, limit: int = Query(50, ge=1, le=MAX_RECENT_DECISIONS)) -> list[dict[str, Any]]:
    store = _store(request)
    if store.log is None:
        return []
    try:
        return await run_in_threadpool(store.log.recent, limit)
    except Exception as exc:  # noqa: BLE001 - a fixed text for the client, the cause logged
        logger.error(f"decision log: reading failed: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=503, detail="decision log unavailable") from exc


@r.get("/{plan_id}", response_model=PlanView)
async def get_plan(plan_id: str, request: Request) -> PlanView:
    return await run_in_threadpool(_store(request).view, plan_id)


@r.post("/{plan_id}/approve", response_model=PlanView)
async def approve_plan(plan_id: str, request: Request, body: Decision | None = None) -> PlanView:
    body = body or Decision()
    return await run_in_threadpool(_store(request).decide, plan_id, "approve", body.actor, body.note, body.approval_token)


@r.post("/{plan_id}/reject", response_model=PlanView)
async def reject_plan(plan_id: str, request: Request, body: Decision | None = None) -> PlanView:
    body = body or Decision()
    return await run_in_threadpool(_store(request).decide, plan_id, "reject", body.actor, body.note)
