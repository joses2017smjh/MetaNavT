"""POST /api/answer/: retrieve, then a cited answer from Claude, under shared quotas and a daily budget (public demo).

The order is fixed so a refused request costs nothing: the caller's quota first, then retrieval (the served
retriever, as /api/retrieve), then the answer cache, then the daily budget's reservation, then the call. Every path
returns the retrieved hits; `answer` is null with a `reason` whenever no answer is delivered:

  answers off           ANTHROPIC_API_KEY is not set
  limits unavailable    REDIS_URL is not set or Redis cannot be reached (fail closed: no call without the budget)
  daily budget spent    today's reservations plus this call's worst case are over ANSWER_DAILY_BUDGET_USD
  uncited, refused (...), cut at max_tokens, upstream ...   from app.answer.claude (an uncited reply is returned in
                        `uncited_reply`, unverified: usually that the excerpts do not hold the answer)
  answer spend limit reached   the API refused the call because the account cannot pay (a spend limit someone set,
                        or no credit left: app.answer.claude.UNPAID); nothing was billed, so no spend is recorded

Over the caller's quota is a 429 with Retry-After. Claude's answers are not evaluated in this repository.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from app.answer import claude
from app.api import limits as shared_limits
from app.api.security import Caller, require_api_key
from app.api.routers.retrieve import RetrieveHit, RetrieveRequest, retrieve

answer_router = r = APIRouter()
logger = logging.getLogger("uvicorn")

MAX_QUESTION_CHARS = 500
MAX_K = 10


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)
    k: int = Field(6, ge=1, le=MAX_K, description="excerpts retrieved and sent to the model (1-10)")

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class AnswerCitation(BaseModel):
    chunk_id: str
    path: Optional[str]
    cited_text: str
    start_char: int
    end_char: int
    start_byte: Optional[int] = None
    end_byte: Optional[int] = None


class AnswerResponse(BaseModel):
    question: str
    answer: Optional[str]
    reason: Optional[str]
    uncited_reply: Optional[str] = None
    citations: list[AnswerCitation]
    model: Optional[str]
    fallback_used: bool = False
    cached: bool = False
    cost_usd_estimate: Optional[float] = None
    evaluated: bool = False  # Claude's answers are not evaluated in this repository
    hits: list[RetrieveHit]


def _state(app) -> dict:
    """The route's clients, built once per app: the Anthropic client (None: answers off) and the Limits (None: no
    REDIS_URL)."""
    state = getattr(app.state, "answer", None)
    if state is None:
        settings = claude.AnswerSettings.from_env()
        redis = shared_limits.connect()
        state = {"settings": settings, "client": claude.make_client(settings),
                 "limits": shared_limits.Limits(redis, shared_limits.LimitSettings.from_env()) if redis is not None else None}
        app.state.answer = state
    return state


def caller_identity(request: Request, caller: Caller | None) -> str:
    """The quota's key: the API key's name when authentication is on, else the client address behind the trusted
    proxies (limits.client_address; TRUSTED_PROXY_HOPS)."""
    if caller is not None and caller.name:
        return f"key:{caller.name}"
    peer = request.client.host if request.client else None
    address = shared_limits.client_address(peer, request.headers.get("x-forwarded-for"), shared_limits.trusted_proxy_hops())
    return f"ip:{address or 'unknown'}"


def _response(req: AnswerRequest, hits, *, reason=None, result: Optional[claude.Answer] = None, cached=None) -> AnswerResponse:
    if cached is not None:
        return AnswerResponse(question=req.question, hits=hits, **{**cached, "cached": True})
    delivered = result is not None and result.delivered
    return AnswerResponse(
        question=req.question, hits=hits,
        answer=result.text if delivered else None,
        reason=None if delivered else (reason or (result.reason if result else "not answered")),
        uncited_reply=result.text if result is not None and result.reason == "uncited" else None,
        citations=[AnswerCitation(**vars(c)) for c in (result.citations if delivered else [])],
        model=result.model if result else None,
        fallback_used=bool(result and result.fallback_used),
        cost_usd_estimate=result.cost_usd if result else None)


@r.post("/", response_model=AnswerResponse)
async def answer(req: AnswerRequest, request: Request, caller: Caller = Depends(require_api_key)) -> AnswerResponse:
    state = _state(request.app)
    limits = state["limits"]
    if limits is not None:
        try:
            allowed, retry_after = await run_in_threadpool(limits.allow, caller_identity(request, caller))
        except Exception as exc:  # noqa: BLE001 - Redis down: fail closed for the model, keep retrieval below
            logger.warning(f"answer: limits unavailable ({type(exc).__name__})")
            limits = None
        else:
            if not allowed:
                raise HTTPException(status_code=429, detail="answer quota reached for this caller",
                                    headers={"Retry-After": str(retry_after)})
    retrieved = await retrieve(RetrieveRequest(query=req.question, k=req.k), request)
    hits = retrieved.hits
    if state["client"] is None:
        return _response(req, hits, reason="answers off")
    if limits is None:
        return _response(req, hits, reason="limits unavailable")
    settings: claude.AnswerSettings = state["settings"]
    key = limits.cache_key(settings.model, req.question)
    try:
        cached = await run_in_threadpool(limits.cache_get, key)
    except Exception:  # noqa: BLE001 - a cache miss is safe; the budget below still guards the call
        cached = None
    if cached is not None:
        return _response(req, hits, cached=cached)
    excerpts = [claude.Excerpt(chunk_id=h.chunk_id, path=h.path, text=h.text, start_byte=h.start_byte) for h in hits if h.text]
    if not excerpts:
        return _response(req, hits, reason="nothing retrieved")
    try:
        reservation = await run_in_threadpool(limits.reserve, claude.worst_case_usd(settings, req.question, excerpts))
    except Exception as exc:  # noqa: BLE001 - fail closed
        logger.warning(f"answer: budget unavailable ({type(exc).__name__})")
        return _response(req, hits, reason="limits unavailable")
    if reservation is None:
        return _response(req, hits, reason="daily budget spent")
    result = None
    try:
        result = await run_in_threadpool(claude.answer, state["client"], req.question, excerpts, settings)
    finally:
        # a failure before a reply keeps the whole reservation: whether it was billed is unknown. A refusal because the
        # account cannot pay (claude.UNPAID) costs 0, so its whole reservation is given back: no spend is recorded.
        # Settling goes to the reservation's own day, even when the reply comes back after UTC midnight.
        actual = result.cost_usd if result is not None else None
        try:
            await run_in_threadpool(limits.settle, reservation, actual)
        except Exception as exc:  # noqa: BLE001 - the reservation stays counted, which errs on the safe side
            logger.warning(f"answer: could not settle the budget ({type(exc).__name__})")
    response = _response(req, hits, result=result)
    if result.delivered:
        cached_fields = response.model_dump(exclude={"question", "hits", "cached"})
        try:
            await run_in_threadpool(limits.cache_set, key, cached_fields)
        except Exception:  # noqa: BLE001 - not cached; the answer is still returned
            pass
    return response
