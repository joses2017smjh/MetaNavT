"""Human-in-the-loop file plans: propose, approve, reject. Nothing moves without approval.

POST /api/plans                 {"src": "...", "dst": "..."} -> a pending MovePlan (no file touched)
GET  /api/plans                 every plan this process knows, newest first
POST /api/plans/{id}/approve    apply_plan(approved=True): the only code path that moves the file
POST /api/plans/{id}/reject     mark rejected; the file is never touched

Backed by app.mcp.filesystem.FilesystemTools (the same gate the MCP tools use)
over DATA_DIR, and by a Postgres decision log (table plan_decisions): every
approve / reject is written before the action runs and updated with its
result. If the log cannot be written, the action is refused (503) rather than
run unrecorded.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.mcp.filesystem import ApprovalRequired, FilesystemTools

plans_router = r = APIRouter()


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


class Decision(BaseModel):
    actor: str = Field("web-ui", max_length=128)
    note: Optional[str] = Field(None, max_length=1000)


class PlanStore:
    """FilesystemTools plus per-plan decision state and the Postgres log."""

    def __init__(self, tools: FilesystemTools, log: "DecisionLog | None"):
        self.tools = tools
        self.log = log
        self.decisions: dict[str, dict[str, Any]] = {}

    def create(self, src: str, dst: str) -> PlanView:
        try:
            plan = self.tools.propose_move(src, dst)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"source not found: {exc}") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return self.view(plan["plan_id"])

    def view(self, plan_id: str) -> PlanView:
        plan = self.tools.plans.get(plan_id)
        if plan is None:
            raise HTTPException(status_code=404, detail=f"unknown plan {plan_id}")
        decision = self.decisions.get(plan_id, {})
        if plan.applied:
            status = "applied"
        elif decision.get("action") == "reject":
            status = "rejected"
        elif plan.approved:
            status = "approved"
        else:
            status = "pending_approval"
        return PlanView(plan_id=plan.plan_id, src=plan.src, dst=plan.dst, status=status, created_at=plan.created_at,
                        decided_at=decision.get("decided_at"), actor=decision.get("actor"), note=decision.get("note"))

    def list(self) -> list[PlanView]:
        return sorted((self.view(pid) for pid in self.tools.plans), key=lambda p: p.created_at, reverse=True)

    def decide(self, plan_id: str, action: str, actor: str, note: Optional[str]) -> PlanView:
        current = self.view(plan_id)
        if current.status in {"applied", "rejected"}:
            raise HTTPException(status_code=409, detail=f"plan {plan_id} is already {current.status}")
        record = {"plan_id": plan_id, "src": current.src, "dst": current.dst, "action": action, "actor": actor, "note": note,
                  "decided_at": time.time(), "result": "pending"}
        if self.log is not None:
            try:
                self.log.write(record)  # intent first: an action never runs unrecorded
            except Exception as exc:  # noqa: BLE001 - refused, not hidden
                raise HTTPException(status_code=503, detail=f"decision log unavailable, action refused: {exc}") from exc
        self.decisions[plan_id] = record
        if action == "approve":
            try:
                result = self.tools.apply_plan(plan_id, approved=True)
                record["result"] = result.get("status", "applied")
            except ApprovalRequired as exc:  # cannot happen with approved=True; kept explicit
                record["result"] = f"refused: {exc}"
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            except OSError as exc:
                record["result"] = f"failed: {exc}"
                if self.log is not None:
                    self.log.update(record)
                raise HTTPException(status_code=500, detail=f"move failed: {exc}") from exc
        else:
            record["result"] = "rejected"
        if self.log is not None:
            self.log.update(record)
        return self.view(plan_id)


class DecisionLog:
    """plan_decisions table in Postgres, written through the vector store manager's pool."""

    TABLE = "plan_decisions"

    def __init__(self, vsm):
        self.vsm = vsm
        self.ensure_table()

    def ensure_table(self) -> None:
        with self.vsm.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""CREATE TABLE IF NOT EXISTS {self.TABLE} (
                        id BIGSERIAL PRIMARY KEY,
                        plan_id TEXT NOT NULL,
                        src TEXT NOT NULL,
                        dst TEXT NOT NULL,
                        action TEXT NOT NULL,
                        actor TEXT,
                        note TEXT,
                        decided_at TIMESTAMPTZ NOT NULL,
                        result TEXT NOT NULL,
                        payload JSONB
                    );"""
                )
            conn.commit()

    def write(self, record: dict[str, Any]) -> None:
        with self.vsm.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {self.TABLE} (plan_id, src, dst, action, actor, note, decided_at, result, payload) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);",
                    (record["plan_id"], record["src"], record["dst"], record["action"], record.get("actor"), record.get("note"),
                     datetime.fromtimestamp(record["decided_at"], tz=timezone.utc), record["result"], json.dumps(record)),
                )
            conn.commit()

    def update(self, record: dict[str, Any]) -> None:
        with self.vsm.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {self.TABLE} SET result = %s, payload = %s WHERE plan_id = %s AND action = %s;",
                    (record["result"], json.dumps(record), record["plan_id"], record["action"]),
                )
            conn.commit()

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.vsm.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT plan_id, src, dst, action, actor, decided_at, result FROM {self.TABLE} ORDER BY id DESC LIMIT %s;", (limit,))
                return [dict(zip(("plan_id", "src", "dst", "action", "actor", "decided_at", "result"), row)) for row in cur.fetchall()]


def build_plan_store(vsm=None, root: str | None = None) -> PlanStore:
    root = root or os.getenv("DATA_DIR", "data")
    tools = FilesystemTools(root=root)  # allow_apply stays False: only an explicit approval applies
    log = DecisionLog(vsm) if vsm is not None else None
    return PlanStore(tools, log)


def _store(request: Request) -> PlanStore:
    store = getattr(request.app.state, "plans", None)
    if store is None:
        detail = getattr(request.app.state, "startup_error", None) or "plan store not initialised"
        raise HTTPException(status_code=503, detail=f"plans unavailable: {detail}")
    return store


@r.get("/", response_model=list[PlanView])
async def list_plans(request: Request) -> list[PlanView]:
    return _store(request).list()


@r.post("/", response_model=PlanView, status_code=201)
async def create_plan(body: PlanCreate, request: Request) -> PlanView:
    return await run_in_threadpool(_store(request).create, body.src, body.dst)


@r.get("/decisions/recent")
async def recent_decisions(request: Request, limit: int = 50) -> list[dict[str, Any]]:
    store = _store(request)
    if store.log is None:
        return []
    return await run_in_threadpool(store.log.recent, limit)


@r.get("/{plan_id}", response_model=PlanView)
async def get_plan(plan_id: str, request: Request) -> PlanView:
    return _store(request).view(plan_id)


@r.post("/{plan_id}/approve", response_model=PlanView)
async def approve_plan(plan_id: str, request: Request, body: Decision | None = None) -> PlanView:
    body = body or Decision()
    return await run_in_threadpool(_store(request).decide, plan_id, "approve", body.actor, body.note)


@r.post("/{plan_id}/reject", response_model=PlanView)
async def reject_plan(plan_id: str, request: Request, body: Decision | None = None) -> PlanView:
    body = body or Decision()
    return await run_in_threadpool(_store(request).decide, plan_id, "reject", body.actor, body.note)
