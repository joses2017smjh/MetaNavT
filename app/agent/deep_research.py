"""Phase 8: Deep Research loop — multi-query RRF, citation scratchpad, token budget.

Industry shape (OpenAI / Gemini / Perplexity Deep Research):
plan → sub-queries → hybrid retrieve → scratchpad → critic → cited respond.
Does not replace BM25 with a raw filesystem walk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from app.agent.retrieval_loop import (
    AgentAnswer,
    AgentEvent,
    Citation,
    RetrievalAgent,
    extractive_answer,
    grade_chunk,
)
from app.graph.staleness import cluster_versions, prefer_current
from app.retrieval.bm25 import tokenize
from app.retrieval.fuse import rrf_score_map
from app.retrieval.hybrid import InMemoryHybridIndex, RetrievalHit, RetrievalResult
from app.retrieval.router import QueryRouter, RouteDecision, RouteType


def expand_queries(
    query: str,
    n: int = 3,
    llm: Callable[[str], str] | None = None,
) -> list[str]:
    """Three paraphrases. Heuristic if no LLM: original, identifiers, 'current' form."""
    q = (query or "").strip()
    out: list[str] = [q] if q else [""]
    if llm and q:
        raw = llm(
            f"Write {n - 1} short alternative search queries for: {q}\n"
            "One per line, no numbering."
        ) or ""
        for line in raw.splitlines():
            line = re.sub(r"^[\d.)\-\s]+", "", line).strip()
            if line and line.lower() not in {x.lower() for x in out}:
                out.append(line)
            if len(out) >= n:
                break
    ids = re.findall(
        r"\b(?:run[_\s-]?\d+|dinov2|fusion|rmse|learning[_\s]?rate)\b", q, re.I
    )
    if ids:
        alt = " ".join(ids)
        if alt.lower() not in {x.lower() for x in out}:
            out.append(alt)
    if q and "current" not in q.lower() and any(
        w in q.lower() for w in ("learning rate", "lr", "draft", "config")
    ):
        out.append("current " + q)
    seen: set[str] = set()
    uniq: list[str] = []
    for item in out:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
    while len(uniq) < n and q:
        uniq.append(q)
    return uniq[:n] if n > 0 else uniq


def rrf_union_paths(
    index: InMemoryHybridIndex,
    queries: Sequence[str],
    k: int = 50,
    rrf_k: int = 60,
) -> list[RetrievalHit]:
    ranked_ids: list[list[str]] = []
    hit_by_id: dict[str, RetrievalHit] = {}
    for q in queries:
        result = index.retrieve(q, k=k)
        ranked_ids.append([h.chunk.chunk_id for h in result.hits])
        for h in result.hits:
            hit_by_id.setdefault(h.chunk.chunk_id, h)
    fused = rrf_score_map(ranked_ids, k=rrf_k)
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    hits: list[RetrievalHit] = []
    for rank, (cid, score) in enumerate(ordered, start=1):
        base = hit_by_id[cid]
        hits.append(
            RetrievalHit(
                chunk=base.chunk,
                score=score,
                rank=rank,
                bm25=base.bm25,
                dense=base.dense,
                rrf=score,
                rerank=base.rerank,
            )
        )
    return hits


def multi_query_retrieve(
    index: InMemoryHybridIndex,
    query: str,
    *,
    n: int = 3,
    k: int = 50,
    llm: Callable[[str], str] | None = None,
) -> RetrievalResult:
    queries = expand_queries(query, n=n, llm=llm)
    hits = rrf_union_paths(index, queries, k=k)
    return RetrievalResult(
        query=query,
        hits=hits,
        route=RouteDecision(RouteType.SEMANTIC, "multi-query rrf union"),
        stages_ms={},
    )


@dataclass
class Evidence:
    path: str
    start_byte: int
    end_byte: int
    text: str
    chunk_id: str

    def tokens(self) -> int:
        return max(1, len(tokenize(self.text)))

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "chunk_id": self.chunk_id,
        }


@dataclass
class Scratchpad:
    items: list[Evidence] = field(default_factory=list)
    token_budget: int = 2048

    @property
    def tokens_used(self) -> int:
        return sum(e.tokens() for e in self.items)

    def can_respond(self) -> bool:
        return bool(self.items) and all(
            e.path and e.end_byte >= e.start_byte for e in self.items
        )

    def add_hits(self, hits: Sequence[RetrievalHit], query: str = "") -> None:
        del query  # critic already filtered; budget is the constraint
        for hit in hits:
            if self.tokens_used >= self.token_budget:
                break
            ev = Evidence(
                path=hit.chunk.path,
                start_byte=hit.chunk.start_byte,
                end_byte=hit.chunk.end_byte,
                text=hit.chunk.text,
                chunk_id=hit.chunk.chunk_id,
            )
            if ev.tokens() + self.tokens_used > self.token_budget:
                continue
            if any(x.chunk_id == ev.chunk_id for x in self.items):
                continue
            self.items.append(ev)


class DeepResearchAgent:
    def __init__(
        self,
        index: InMemoryHybridIndex,
        *,
        n_queries: int = 3,
        token_budget: int = 2048,
        llm: Callable[[str], str] | None = None,
        generator: Callable | None = None,
        router: QueryRouter | None = None,
        grade_threshold: float = 0.12,
        staleness_tier1: bool = True,
    ):
        self.index = index
        self.n_queries = n_queries
        self.token_budget = token_budget
        self.llm = llm
        self.generator = generator or extractive_answer
        self.grade_threshold = grade_threshold
        self.staleness_tier1 = staleness_tier1
        self.version_clusters = (
            cluster_versions(index.chunks) if staleness_tier1 else {}
        )
        self.inner = RetrievalAgent(
            index, router=router, generator=self.generator, grade_threshold=grade_threshold
        )

    def run(self, query: str) -> AgentAnswer:
        events: list[AgentEvent] = []
        queries = expand_queries(query, n=self.n_queries, llm=self.llm)
        events.append(AgentEvent("plan", {"queries": queries}))
        hits = rrf_union_paths(self.index, queries)
        events.append(AgentEvent("search", {"n_hits": len(hits), "n_queries": len(queries)}))
        if self.version_clusters:
            before = {h.chunk.chunk_id: h.chunk.path for h in hits}
            hits = prefer_current(hits, self.version_clusters, query)
            after = {h.chunk.chunk_id for h in hits}
            dropped = [path for cid, path in before.items() if cid not in after]
            events.append(AgentEvent("staleness", {"dropped_paths": dropped}))
        pad = Scratchpad(token_budget=self.token_budget)
        graded = [h for h in hits if grade_chunk(query, h.chunk.text) >= self.grade_threshold]
        events.append(AgentEvent("grade_relevance", {"n_relevant": len(graded)}))
        pad.add_hits(graded or hits, query)
        events.append(
            AgentEvent("scratchpad", {"n": len(pad.items), "tokens": pad.tokens_used})
        )
        route = self.inner.router.route(query)
        if not pad.can_respond():
            answer = AgentAnswer(
                text="No cited evidence fit the token budget. Refusing to guess.",
                citations=[],
                confidence="empty",
                route=route,
                iterations=len(queries),
                events=events,
                failed=True,
                fail_reason="empty_scratchpad",
            )
            events.append(AgentEvent("respond", answer.as_dict()))
            return answer
        by_id = {h.chunk.chunk_id: h for h in hits}
        real = [by_id[e.chunk_id] for e in pad.items if e.chunk_id in by_id]
        text = self.generator(query, real)
        citations = [
            Citation(
                path=e.path,
                start_byte=e.start_byte,
                end_byte=e.end_byte,
                chunk_id=e.chunk_id,
            )
            for e in pad.items
        ]
        answer = AgentAnswer(
            text=text,
            citations=citations,
            confidence="high",
            route=route,
            iterations=len(queries),
            events=events,
            failed=False,
        )
        events.append(AgentEvent("respond", answer.as_dict()))
        return answer
