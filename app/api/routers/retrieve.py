"""Retrieval-only API: route + hybrid search + citations + stage latency.

The index, vector store and retriever are built once in main.py's lifespan
(app.state.retrieval); the blocking retrieve runs in a worker thread. The
typed response, k validation and per-stage scores arrive in M3.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

retrieve_router = r = APIRouter()


class RetrieveRequest(BaseModel):
    query: str
    k: int | None = None


def _path(meta: dict) -> str | None:
    # `path` is relative to DATA_DIR (set by app.engine.bootstrap); file_path is the
    # absolute path SimpleDirectoryReader records.
    return meta.get("path") or meta.get("file_path") or meta.get("file_name")


@r.post("/")
async def retrieve(req: RetrieveRequest, request: Request):
    state = getattr(request.app.state, "retrieval", None)
    if state is None:
        detail = getattr(request.app.state, "startup_error", None) or "retrieval backend not initialised"
        raise HTTPException(status_code=503, detail=f"retrieval backend unavailable: {detail}")
    try:
        nodes = await run_in_threadpool(state.retriever.retrieve, req.query)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"retrieval backend unavailable: {exc}") from exc

    hits = []
    for node in nodes[: (req.k or 8)]:
        meta = getattr(node.node, "metadata", None) or {}
        hits.append(
            {
                "node_id": node.node.node_id,
                "score": node.score,
                "text": node.node.get_content()[:800],
                "path": _path(meta),
            }
        )
    retriever = state.retriever
    timer = getattr(retriever, "last_timer", None)
    route = getattr(retriever, "last_route", None)
    return {
        "query": req.query,
        "route": route.route.value if route else None,
        "latency": timer.summary() if timer else {},
        "counts": getattr(retriever, "last_counts", {}),
        "bm25_backend": getattr(state.vsm, "last_bm25_backend", None),
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "huggingface"),
        "reranker_loaded": getattr(retriever, "_reranker", None) is not None,
        "hits": hits,
    }
