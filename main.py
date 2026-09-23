"""
MetaNaviT FastAPI Application Entry Point

This module serves as the main entry point for the MetaNaviT application.
It handles server configuration, middleware setup, and static file mounting
based on the environment (development or production).

Features:
    - FastAPI application initialization
    - Environment-based configuration
    - Static file serving
    - Frontend proxy middleware (dev mode)

Environment Variables:
    ENVIRONMENT: Running environment (dev/prod)
    FRONTEND_ENDPOINT: Frontend server URL for dev proxy
    APP_HOST: Server host address
    APP_PORT: Server port number

Dependencies:
    - FastAPI for API framework
    - Uvicorn for ASGI server
    - Custom middleware for frontend proxying
"""
    
# flake8: noqa: E402
from app.config import DATA_DIR, STATIC_DIR
from dotenv import load_dotenv

load_dotenv()

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from app.api.routers import api_router
from app.middlewares.frontend import FrontendProxyMiddleware
from app.observability import configure_logging, tracing_status
from app.settings import init_settings
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

environment = os.getenv("ENVIRONMENT", "dev")
LOG_FORMAT_USED = configure_logging()  # LOG_FORMAT=json -> one JSON object per line
logger = logging.getLogger("uvicorn")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the model settings and the retrieval singletons once, off the event loop.

    A failure (database down, bad DATA_DIR) is recorded in app.state.startup_error:
    the process stays up so /health can report it and the logs are readable, and
    /api/retrieve answers 503 until it is fixed.
    """
    from app.engine.bootstrap import build_retrieval_state

    app.state.retrieval = None
    app.state.startup_error = None
    app.state.plans = None
    try:
        init_settings()
        app.state.retrieval = await run_in_threadpool(build_retrieval_state)
        logger.info(f"Retrieval ready: {app.state.retrieval.indexing}")
        from app.api.routers.plans import build_plan_store

        app.state.plans = await run_in_threadpool(build_plan_store, app.state.retrieval.vsm, DATA_DIR)
        logger.info(f"Plan store ready over {DATA_DIR} (decisions logged to Postgres)")
    except Exception as exc:  # noqa: BLE001 - surfaced by /health
        app.state.startup_error = f"{type(exc).__name__}: {exc}"
        logger.error(f"Retrieval backend failed to start: {app.state.startup_error}", exc_info=True)
    yield


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    """200 with index stats once retrieval is ready; 503 with the startup error otherwise."""
    state = getattr(app.state, "retrieval", None)
    if state is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "error": getattr(app.state, "startup_error", None)},
        )
    n_nodes = await run_in_threadpool(state.vsm.count_nodes)
    reranker = getattr(state.retriever, "reranker_info", None) or {}
    degraded = []
    if reranker.get("configured") and not reranker.get("loaded"):
        degraded.append({"component": "rerank", "error": reranker.get("error") or "configured but not loaded"})
    return {
        "status": "ok" if not degraded else "degraded",
        "n_nodes": n_nodes,
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "huggingface"),
        "embed_model": type(state.index._embed_model).__name__ if getattr(state.index, "_embed_model", None) else None,
        "bm25_backend": getattr(state.vsm, "hybrid_backend", None) or state.vsm.last_bm25_backend,
        "retrieval_mode": getattr(state.retriever, "mode", None),
        "reranker_loaded": state.retriever._reranker is not None,
        "reranker": reranker,
        "degraded": degraded,
        "indexing": state.indexing,
        "observability": {"log_format": LOG_FORMAT_USED, "tracing": tracing_status()},
    }



def mount_static_files(directory, path, html=False):
    """
    Mount static file directories to serve through FastAPI.
    
    Args:
        directory: Local directory path to mount
        path: URL path to mount the directory at
        html: Whether to serve index.html for directory roots
    """
    if os.path.exists(directory):
        logger.info(f"Mounting static files '{directory}' at '{path}'")
        app.mount(
            path,
            StaticFiles(directory=directory, check_dir=False, html=html),
            name=f"{directory}-static",
        )


app.include_router(api_router, prefix="/api")

# Mount the data files to serve the file viewer
mount_static_files(DATA_DIR, "/api/files/data")
# Mount the output files from tools
mount_static_files("output", "/api/files/output")
 # Development mode: Use frontend proxy if configured
if environment == "dev":
    frontend_endpoint = os.getenv("FRONTEND_ENDPOINT")
    if frontend_endpoint:
        app.add_middleware(
            FrontendProxyMiddleware,
            frontend_endpoint=frontend_endpoint,
            excluded_paths=set(
                route.path for route in app.routes if hasattr(route, "path")
            ),
        )
    else:
        # No frontend in dev: redirect to API docs
        logger.warning("No frontend endpoint - starting API server only")

        @app.get("/")
        async def redirect_to_docs():
            return RedirectResponse(url="/docs")
else:
    # Mount the frontend static files (production)
    mount_static_files(STATIC_DIR, "/", html=True)

if __name__ == "__main__":
    app_host = os.getenv("APP_HOST", "0.0.0.0")
    app_port = int(os.getenv("APP_PORT", "8000"))
    reload = True if environment == "dev" else False

    uvicorn.run(app="main:app", host=app_host, port=app_port, reload=reload)
