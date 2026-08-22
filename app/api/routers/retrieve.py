"""Retrieval-only API: route + hybrid search + citations + stage latency."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.engine.index import IndexConfig, get_index
from app.engine.retriever import create_hybrid_retriever
from app.database.vector_store import get_vector_store_manager

retrieve_router = r = APIRouter()


class RetrieveRequest(BaseModel):
    query: str
    k: int | None = None


@r.post("/")
async def retrieve(req: RetrieveRequest):
    try:
        index = get_index(IndexConfig())
        vsm = get_vector_store_manager()
        retriever = create_hybrid_retriever(index, vsm)
        nodes = retriever.retrieve(req.query)
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
                "path": meta.get("file_path") or meta.get("file_name") or meta.get("path"),
            }
        )
    timer = getattr(retriever, "last_timer", None)
    route = getattr(retriever, "last_route", None)
    return {
        "query": req.query,
        "route": route.route.value if route else None,
        "latency": timer.summary() if timer else {},
        "hits": hits,
    }
