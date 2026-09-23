"""Retrieval-only API: the benchmarked pipeline over Postgres, typed.

POST /api/retrieve/  {"query": "...", "k": 8}
  -> route, per-stage latency, counts, and hits with path + byte range +
     per-stage scores (bm25, dense, rrf, rerank). 422 on a bad body, 503 when
     the backend is down. The index, vector store, clusters and retriever are
     built once in main.py's lifespan; the blocking retrieve runs in a worker
     thread and returns a per-request RetrievalOutcome (no shared state).
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

retrieve_router = r = APIRouter()

MAX_K = 50
MAX_QUERY_CHARS = 2000


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS, description="natural-language question or path")
    k: int = Field(8, ge=1, le=MAX_K, description="number of hits to return (1-50)")

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class StageScores(BaseModel):
    bm25: Optional[float] = None
    dense: Optional[float] = None
    rrf: Optional[float] = None
    rerank: Optional[float] = None


class RetrieveHit(BaseModel):
    rank: int
    chunk_id: str
    path: Optional[str]
    start_byte: Optional[int] = None
    end_byte: Optional[int] = None
    score: float
    scores: StageScores
    text: str


class Staleness(BaseModel):
    enabled: bool
    applied: bool
    dropped: int


class RetrieveResponse(BaseModel):
    query: str
    k: int
    route: Optional[str]
    retrieval_mode: str
    bm25_backend: Optional[str]
    embedding_provider: str
    reranker_loaded: bool
    staleness: Staleness
    counts: dict[str, int]
    latency_ms: dict[str, float]
    hits: list[RetrieveHit]


def _path(meta: dict) -> Optional[str]:
    # `path` is relative to DATA_DIR (app.engine.bootstrap); file_path is absolute.
    return meta.get("path") or meta.get("file_path") or meta.get("file_name")


def _int(value) -> Optional[int]:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


@r.post("/", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest, request: Request) -> RetrieveResponse:
    state = getattr(request.app.state, "retrieval", None)
    if state is None:
        detail = getattr(request.app.state, "startup_error", None) or "retrieval backend not initialised"
        raise HTTPException(status_code=503, detail=f"retrieval backend unavailable: {detail}")
    try:
        outcome = await run_in_threadpool(state.retriever.retrieve_detailed, req.query, req.k)
    except Exception as exc:  # noqa: BLE001 - any backend failure is a 503 to the client
        raise HTTPException(status_code=503, detail=f"retrieval backend unavailable: {exc}") from exc

    hits = []
    for rank, node in enumerate(outcome.nodes[: req.k], start=1):
        meta = getattr(node.node, "metadata", None) or {}
        sc = outcome.scores.get(node.node.node_id, {})
        hits.append(
            RetrieveHit(
                rank=rank,
                chunk_id=node.node.node_id,
                path=_path(meta),
                start_byte=_int(meta.get("start_byte")),
                end_byte=_int(meta.get("end_byte")),
                score=float(node.score or 0.0),
                scores=StageScores(**{k: sc.get(k) for k in ("bm25", "dense", "rrf", "rerank")}),
                text=node.node.get_content()[:800],
            )
        )
    return RetrieveResponse(
        query=req.query,
        k=req.k,
        route=outcome.route.route.value if outcome.route else None,
        retrieval_mode=outcome.mode,
        bm25_backend=outcome.bm25_backend,
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "huggingface"),
        reranker_loaded=getattr(state.retriever, "_reranker", None) is not None,
        staleness=Staleness(**outcome.staleness),
        counts=outcome.counts,
        latency_ms=outcome.stages_ms,
        hits=hits,
    )
