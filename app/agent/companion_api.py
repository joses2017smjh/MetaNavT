"""HTTP gateway for the read-only companion agent, without LlamaIndex/models.

    uvicorn app.agent.companion_api:app --host 127.0.0.1 --port 8001

Demo reuses the mobile fixture retrieval router in process. Live retrieval and
Ollama endpoints are operator configuration, never supplied by a caller.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

from app.agent.companion import AgentResponse, AskRequest, ScriptedDemoModel, TOOL_SPECS, run_agent
from app.api.routers.retrieve import RetrieveRequest, RetrieveResponse, retrieve


def _base_url(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or any(char.isspace() for char in value)):
        raise ValueError("Agent endpoint must be an HTTP(S) base URL without credentials, query or fragment")
    _ = parsed.port
    return value.rstrip("/")


def _origin(value: str) -> str:
    _base_url(value)
    parsed = urlsplit(value)
    if parsed.path or "*" in value:
        raise ValueError("Agent CORS origins must be explicit HTTP(S) origins without paths")
    return value


@dataclass(frozen=True)
class AgentConfig:
    api_key: str = ""
    retrieval_url: str = ""
    retrieval_key: str = ""
    model_url: str = "http://127.0.0.1:11434"
    model: str = ""
    model_key: str = ""
    allowed_origins: tuple[str, ...] = ()
    timeout_seconds: float = 60
    max_concurrent: int = 4

    def __post_init__(self):
        if self.retrieval_url:
            _base_url(self.retrieval_url)
        _base_url(self.model_url)
        for origin in self.allowed_origins:
            _origin(origin)
        if not 5 <= self.timeout_seconds <= 120:
            raise ValueError("Agent timeout must be between 5 and 120 seconds")
        if not 1 <= self.max_concurrent <= 8:
            raise ValueError("Agent concurrent request limit must be between 1 and 8")

    @property
    def live_configured(self) -> bool:
        return bool(self.api_key and self.retrieval_url and self.model)

    @classmethod
    def from_env(cls):
        return cls(api_key=os.getenv("AGENT_API_KEY", ""),
                   retrieval_url=os.getenv("AGENT_RETRIEVAL_URL", ""),
                   retrieval_key=os.getenv("AGENT_RETRIEVAL_API_KEY", ""),
                   model_url=os.getenv("AGENT_OLLAMA_URL", "http://127.0.0.1:11434"),
                   model=os.getenv("AGENT_OLLAMA_MODEL", ""),
                   model_key=os.getenv("AGENT_OLLAMA_API_KEY", ""),
                   allowed_origins=tuple(x.strip() for x in os.getenv("AGENT_ALLOWED_ORIGINS", "").split(",") if x.strip()),
                   timeout_seconds=float(os.getenv("AGENT_TIMEOUT_SECONDS", "60")),
                   max_concurrent=int(os.getenv("AGENT_MAX_CONCURRENT", "4")))


async def _json_request(client: httpx.AsyncClient, url: str, payload: dict, headers: dict, *, max_bytes: int = 256_000) -> dict:
    # Bound untrusted upstream bodies before decoding. Never follow redirects
    # or reveal raw provider responses, URLs, headers or exception messages.
    async with client.stream("POST", url, json=payload, headers=headers) as response:
        response.raise_for_status()
        data = bytearray()
        async for part in response.aiter_bytes():
            data.extend(part)
            if len(data) > max_bytes:
                raise ValueError("upstream response exceeds limit")
    result = json.loads(data)
    if not isinstance(result, dict):
        raise ValueError("invalid upstream response")
    return result


class HttpRetrievalBackend:
    def __init__(self, config: AgentConfig, client: httpx.AsyncClient):
        self.config, self.client = config, client

    async def search(self, query: str, k: int) -> RetrieveResponse:
        headers = {"X-API-Key": self.config.retrieval_key} if self.config.retrieval_key else {}
        payload = await _json_request(self.client, self.config.retrieval_url.rstrip("/") + "/api/retrieve/",
                                      {"query": query, "k": k}, headers)
        return RetrieveResponse.model_validate(payload)


class LocalRetrievalBackend:
    def __init__(self, demo_app: FastAPI):
        self.app = demo_app

    async def search(self, query: str, k: int) -> RetrieveResponse:
        request = Request({"type": "http", "app": self.app})
        return await retrieve(RetrieveRequest(query=query, k=k), request)


class OllamaDecisionModel:
    def __init__(self, config: AgentConfig, client: httpx.AsyncClient):
        self.config, self.client = config, client
        self.name = config.model

    async def next(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        headers = {"Authorization": "Bearer " + self.config.model_key} if self.config.model_key else {}
        payload = await _json_request(self.client, self.config.model_url.rstrip("/") + "/api/chat",
                                      {"model": self.name, "messages": messages, "tools": TOOL_SPECS,
                                       "stream": False, "think": False,
                                       "options": {"temperature": 0, "num_predict": 1500}}, headers)
        message = payload.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise ValueError("invalid model response")
        if message.get("content") is not None and not isinstance(message["content"], str):
            raise ValueError("invalid model response")
        # Only known protocol fields enter the next prompt. Thinking text,
        # provider metadata, logprobs and returned headers are not published.
        return {key: message[key] for key in ("role", "content", "tool_calls") if key in message}


def create_app(config: AgentConfig | None = None, *, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    config = config or AgentConfig.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from app.mobile.demo import create_app as create_demo_app

        demo_app = create_demo_app()
        async with demo_app.router.lifespan_context(demo_app):
            async with httpx.AsyncClient(transport=transport, timeout=min(config.timeout_seconds, 30),
                                         follow_redirects=False, trust_env=False) as client:
                app.state.demo_backend = LocalRetrievalBackend(demo_app)
                app.state.live_backend = HttpRetrievalBackend(config, client)
                app.state.live_model = OllamaDecisionModel(config, client)
                app.state.active_requests = 0
                app.state.ready = True
                yield
                app.state.ready = False

    app = FastAPI(title="MetaNavT read-only research companion agent", lifespan=lifespan)
    app.state.ready = False
    if config.allowed_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(config.allowed_origins),
                           allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-API-Key"])

    @app.get("/agent/health")
    async def health():
        return {"service": "metanavt-agent", "demo_ready": app.state.ready,
                "live_configured": config.live_configured,
                "live_readiness": "not_probed", "demo_model": ScriptedDemoModel.name,
                "live_model": config.model or None,
                "notice": "Configured does not mean that the model is installed, supports tools, or is ready. Live capability is validated by an actual request."}

    @app.post("/agent/ask", response_model=AgentResponse)
    async def ask(request: AskRequest, x_api_key: str | None = Header(default=None)):
        if request.mode == "live":
            if not config.api_key or not x_api_key or not hmac.compare_digest(x_api_key.encode(), config.api_key.encode()):
                raise HTTPException(status_code=401, detail="Live research-agent access requires the operator's API key.")
            if not config.live_configured:
                raise HTTPException(status_code=503, detail="The live research agent is not configured.")
        if not app.state.ready:
            raise HTTPException(status_code=503, detail="The research agent is starting.")
        # No awaits between checking and incrementing: one ASGI process admits
        # at most this many tasks. Multiple workers have independent limits.
        if app.state.active_requests >= config.max_concurrent:
            raise HTTPException(status_code=429, detail="The research agent is busy. Please try again.")
        app.state.active_requests += 1
        try:
            model = ScriptedDemoModel() if request.mode == "demo" else app.state.live_model
            backend = app.state.demo_backend if request.mode == "demo" else app.state.live_backend
            async with asyncio.timeout(config.timeout_seconds):
                return await run_agent(request, model, backend)
        except TimeoutError:
            raise HTTPException(status_code=504, detail="The research-agent request timed out.") from None
        except Exception:
            raise HTTPException(status_code=503, detail="The research-agent model or retrieval service is unavailable.") from None
        finally:
            app.state.active_requests -= 1

    return app


app = create_app()
