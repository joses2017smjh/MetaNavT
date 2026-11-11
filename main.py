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

import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress

import uvicorn
from app.api.routers import api_router
from app.api.security import GuardedStaticFiles, security_status
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
    """Build the model settings and the retrieval singletons off the event loop.

    The first attempt runs before the server accepts requests. A failure (database down
    or still starting, a migration not applied yet, bad DATA_DIR) is recorded in
    app.state.startup_error and retried in the background with capped exponential backoff
    (app.engine.bootstrap.startup_retry_policy): the process stays up, /health answers 503
    with the error and the attempt count and /api/retrieve 503, until an attempt succeeds;
    then /health turns 200 without a restart. It used to keep the first error for the life
    of the process.
    """
    from app.engine.bootstrap import retry_delays, startup_retry_policy

    app.state.retrieval = None
    app.state.startup_error = None
    app.state.plans = None
    app.state.startup = {"attempts": 0, "retrying": False, "retry_delay_s": None}
    policy = startup_retry_policy()
    built: dict = {}  # what this lifespan's attempts have built; app.state is only written, never read back
    retry = None
    if not await _start_attempt(app, built):
        delays = retry_delays(policy.initial_s, policy.max_s)
        delay = _next_retry(app, policy, delays)
        if delay is not None:
            retry = asyncio.create_task(_retry_start(app, built, policy, delays, delay))
    try:
        yield
    finally:
        if retry is not None:  # the wait between attempts is where the cancellation lands
            retry.cancel()
            with suppress(asyncio.CancelledError):
                await retry


async def _start_attempt(app: FastAPI, built: dict) -> bool:
    """One start-up attempt: whatever is not built yet, in order. True when everything is.

    Each step that succeeded is kept and not run again: the model settings, the retrieval
    state (served as soon as it exists, as before retries existed), then the plan store.
    """
    from app.api.routers.plans import build_plan_store
    from app.engine.bootstrap import build_retrieval_state

    app.state.startup["attempts"] += 1
    attempt = app.state.startup["attempts"]
    try:
        if not built.get("settings"):
            init_settings()
            built["settings"] = True
        if built.get("retrieval") is None:
            built["retrieval"] = await run_in_threadpool(build_retrieval_state)
            app.state.retrieval = built["retrieval"]
            logger.info(f"Retrieval ready: {built['retrieval'].indexing}")
        if built.get("plans") is None:
            built["plans"] = await run_in_threadpool(build_plan_store, built["retrieval"].vsm, DATA_DIR)
            app.state.plans = built["plans"]
            logger.info(f"Plan store ready over {DATA_DIR} (decisions logged to Postgres)")
    except Exception as exc:  # noqa: BLE001 - surfaced by /health, then retried
        app.state.startup_error = f"{type(exc).__name__}: {exc}"
        if attempt == 1:
            logger.error(f"Retrieval backend failed to start: {app.state.startup_error}", exc_info=True)
        else:
            logger.warning(f"Start-up attempt {attempt} failed: {app.state.startup_error}")
        return False
    app.state.startup_error = None
    if attempt > 1:
        logger.info(f"Start-up succeeded on attempt {attempt}")
    return True


def _next_retry(app: FastAPI, policy, delays) -> float | None:
    """The wait before the next attempt, recorded for /health; None once policy.attempts (when set) are used."""
    if policy.attempts and app.state.startup["attempts"] >= policy.attempts:
        app.state.startup.update(retrying=False, retry_delay_s=None)
        logger.error(f"Start-up gave up after {app.state.startup['attempts']} attempt(s) (STARTUP_RETRY_ATTEMPTS)")
        return None
    delay = next(delays)
    app.state.startup.update(retrying=True, retry_delay_s=delay)
    logger.info(f"Retrying start-up in {delay:g} s")
    return delay


async def _retry_start(app: FastAPI, built: dict, policy, delays, delay: float | None) -> None:
    """Wait, attempt again, until an attempt succeeds or _next_retry gives up."""
    while delay is not None:
        await asyncio.sleep(delay)
        if await _start_attempt(app, built):
            app.state.startup.update(retrying=False, retry_delay_s=None)
            return
        delay = _next_retry(app, policy, delays)


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    """200 with index stats once retrieval is ready; 503 with the startup error otherwise.

    Also 503 (status "unavailable", n_nodes null, a "database" entry in degraded) when the
    data table cannot be read: a database error is never reported as an empty index.
    Never behind the API key (orchestrator probes must reach it). Both answers say
    whether authentication and the request budget are on (app/api/security.py).
    """
    state = getattr(app.state, "retrieval", None)
    # Start-up attempts so far, and whether (and after what wait) another one is coming.
    startup = dict(getattr(app.state, "startup", None) or {})
    if state is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "error": getattr(app.state, "startup_error", None), "startup": startup, **security_status()},
        )
    database_error = None
    try:
        n_nodes = await run_in_threadpool(state.vsm.count_nodes)
    except Exception as exc:  # noqa: BLE001 - the data table cannot be read: never report it as 0 rows
        n_nodes, database_error = None, f"{type(exc).__name__}: {exc}"
    dense_search = await run_in_threadpool(_dense_search_status, state)
    reranker = getattr(state.retriever, "reranker_info", None) or {}
    from app.settings import embedding_device

    embedder = getattr(state.index, "_embed_model", None)
    devices = {"embedding": embedding_device(embedder) if embedder is not None else None, "reranker": reranker.get("device")}
    degraded = []
    if reranker.get("configured") and not reranker.get("loaded"):
        degraded.append({"component": "rerank", "error": reranker.get("error") or "configured but not loaded"})
    required = os.getenv("RETRIEVAL_REQUIRE_DEVICE")
    device_errors = []
    if required:
        for component, actual in devices.items():
            if component == "reranker" and not reranker.get("configured"):
                continue
            if not str(actual or "").startswith(required):
                device_errors.append({"component": component, "error": f"required device {required}, loaded weights on {actual!r}"})
    degraded.extend(device_errors)
    if database_error:
        degraded.append({"component": "database", "error": database_error})
    payload = {
        "status": "unavailable" if database_error else "ok" if not degraded else "degraded",
        "n_nodes": n_nodes,
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "huggingface"),
        "embed_model": type(state.index._embed_model).__name__ if getattr(state.index, "_embed_model", None) else None,
        "devices": devices,
        "required_device": required,
        "bm25_backend": getattr(state.vsm, "hybrid_backend", None) or state.vsm.last_bm25_backend,
        "dense_search": dense_search,
        "retrieval_mode": getattr(state.retriever, "mode", None),
        "reranker_loaded": state.retriever._reranker is not None,
        "reranker": reranker,
        "degraded": degraded,
        "indexing": state.indexing,
        "startup": startup,
        "observability": {"log_format": LOG_FORMAT_USED, "tracing": tracing_status()},
        **security_status(),
    }
    return JSONResponse(status_code=503, content=payload) if device_errors or database_error else payload


def _dense_search_status(state) -> dict:
    """Depth of the dense list and the settings each hybrid statement runs with (hnsw.ef_search, ...)."""
    depth = getattr(state.retriever, "depth", None)
    settings_for = getattr(state.vsm, "search_settings", None)
    version = getattr(state.vsm, "pgvector_version", None)
    return {
        "depth": depth,
        "settings": settings_for(depth) if settings_for and depth else {},
        "pgvector": ".".join(str(p) for p in version) if version else None,
    }



def mount_static_files(directory, path, html=False, static_cls=StaticFiles):
    """
    Mount static file directories to serve through FastAPI.
    
    Args:
        directory: Local directory path to mount
        path: URL path to mount the directory at
        html: Whether to serve index.html for directory roots
        static_cls: StaticFiles, or GuardedStaticFiles for mounts under /api
    """
    if os.path.exists(directory):
        logger.info(f"Mounting static files '{directory}' at '{path}'")
        app.mount(
            path,
            static_cls(directory=directory, check_dir=False, html=html),
            name=f"{directory}-static",
        )


app.include_router(api_router, prefix="/api")

# Mounts are not routes of api_router, so its dependencies do not reach them: the
# guarded class applies the same API key check and request budget to /api/files/*.
# Mount the data files to serve the file viewer
mount_static_files(DATA_DIR, "/api/files/data", static_cls=GuardedStaticFiles)
# Mount the output files from tools
mount_static_files("output", "/api/files/output", static_cls=GuardedStaticFiles)
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
