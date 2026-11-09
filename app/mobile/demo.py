"""Download-free, read-only mobile demo using the existing retrieval API.

Run ``python -m uvicorn app.mobile.demo:app --host 127.0.0.1 --port 8000``.
The five packaged documents are byte-identical copies of the existing synthetic
benchmark fixture, not measured experiment results. This app exposes health and retrieval only: no chat, upload, file
mutation, shell execution, approval tokens, or model/database initialization.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers.retrieve import retrieve_router
from app.eval.index_loader import build_index

CORPUS = Path(__file__).with_name("fixtures")
STARTER_QUERIES = [
    "What is the learning rate in run_047.yaml?",
    "What checkpoint belongs to run_047.ckpt.meta.json?",
    "What GPU resources does run_047.sbatch request?",
]


@dataclass(frozen=True)
class _DemoNode:
    node_id: str
    text: str
    metadata: dict

    def get_content(self) -> str:
        return self.text


class DemoRetriever:
    """Adapt the actual in-memory index to the live router's outcome contract."""

    mode = "mobile_demo_hash"
    _reranker = None
    reranker_info = {"configured": False, "loaded": False}

    def __init__(self, index):
        self.index = index

    def retrieve_detailed(self, query: str, top_n: int | None = None):
        started = time.perf_counter()
        result = self.index.retrieve(query, k=top_n or 8)
        hits = result.hits[:top_n or 8]
        nodes = [
            SimpleNamespace(
                node=_DemoNode(
                    hit.chunk.chunk_id,
                    hit.chunk.text,
                    {"path": hit.chunk.path, "start_byte": hit.chunk.start_byte,
                     "end_byte": hit.chunk.end_byte},
                ),
                score=hit.score,
            )
            for hit in hits
        ]
        return SimpleNamespace(
            nodes=nodes,
            route=result.route,
            stages_ms={**result.stages_ms, "total": (time.perf_counter() - started) * 1000},
            counts={"returned": len(nodes), "indexed": len(self.index.chunks)},
            scores={hit.chunk.chunk_id: {name: getattr(hit, name)
                    for name in ("bm25", "dense", "rrf", "rerank")} for hit in hits},
            staleness={"enabled": False, "applied": False, "dropped": 0},
            mode=self.mode,
            bm25_backend="python_bm25_demo",
            degraded=[],
        )


def create_app(*, allowed_origins: tuple[str, ...] = ()) -> FastAPI:
    for origin in allowed_origins:
        parsed = urlsplit(origin)
        if (origin == "*" or any(char.isspace() for char in origin)
                or parsed.scheme not in {"http", "https"} or not parsed.hostname
                or "*" in parsed.netloc or parsed.username or parsed.password
                or parsed.path or parsed.query or parsed.fragment):
            raise ValueError("MOBILE_DEMO_ORIGINS must contain explicit HTTP(S) origins without paths or credentials")
        # Invalid ports should fail at startup rather than silently disable access.
        _ = parsed.port

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Hash vectors use no learned weights; citations retain raw UTF-8 spans.
        index = build_index(CORPUS, embedder_name="hash", enable_rerank=False,
                            chunk_strategy="fixed", include_path_context=True)
        app.state.retrieval = SimpleNamespace(
            retriever=DemoRetriever(index), embedding_provider="hash-demo-fixture",
        )
        yield
        app.state.retrieval = None

    app = FastAPI(title="MetaNavT mobile fixture demo", lifespan=lifespan)
    if allowed_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(allowed_origins),
                           allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    app.include_router(retrieve_router, prefix="/api/retrieve")

    @app.get("/health")
    def health():
        state = app.state.retrieval
        return {
            "status": "ok", "mode": "demo", "synthetic": True,
            "n_nodes": len(state.retriever.index.chunks),
            "n_files": sum(path.is_file() for path in CORPUS.rglob("*")),
            "embedding_provider": state.embedding_provider,
            "retrieval_mode": state.retriever.mode,
            "bm25_backend": "python_bm25_demo",
            "reranker_loaded": False,
            "reranker": state.retriever.reranker_info,
            "degraded": [],
            "starter_queries": STARTER_QUERIES,
            "notice": "Synthetic read-only fixture. Connect main:app for your indexed research files.",
        }

    return app


app = create_app(allowed_origins=tuple(origin.strip() for origin in
                 os.getenv("MOBILE_DEMO_ORIGINS", "").split(",") if origin.strip()))
