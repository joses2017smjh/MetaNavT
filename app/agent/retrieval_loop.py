"""Agentic retrieval loop.

nodes: route -> plan -> search -> grade_relevance -> (rewrite | search_more | answer) -> verify -> respond
edges: conditional, hard iteration cap

The relevance-grading node is the cheap version of Self-RAG reflection tokens:
if too few chunks pass, rewrite and search again rather than answering from bad context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterator

from app.retrieval.bm25 import tokenize
from app.retrieval.hybrid import InMemoryHybridIndex, RetrievalHit, RetrievalResult
from app.retrieval.router import QueryRouter, RouteDecision, RouteType


@dataclass
class Citation:
    path: str
    start_byte: int
    end_byte: int
    chunk_id: str

    def resolve(self) -> bool:
        return bool(self.path) and self.end_byte >= self.start_byte

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "chunk_id": self.chunk_id,
        }


@dataclass
class AgentEvent:
    step: str
    payload: dict


@dataclass
class AgentAnswer:
    text: str
    citations: list[Citation]
    confidence: str  # high | low | empty | unsupported
    route: RouteDecision
    iterations: int
    events: list[AgentEvent] = field(default_factory=list)
    failed: bool = False
    fail_reason: str = ""

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "citations": [c.as_dict() for c in self.citations],
            "confidence": self.confidence,
            "route": self.route.route.value,
            "iterations": self.iterations,
            "failed": self.failed,
            "fail_reason": self.fail_reason,
        }


def grade_chunk(query: str, text: str) -> float:
    q = set(tokenize(query))
    d = set(tokenize(text))
    if not q or not d:
        return 0.0
    overlap = len(q & d) / len(q)
    return overlap


def rewrite_query(query: str, hits: list[RetrievalHit], iteration: int) -> str:
    extra_terms = []
    for hit in hits[:3]:
        extra_terms.extend(tokenize(hit.chunk.path))
    # drop already-present tokens
    present = set(tokenize(query))
    added = [t for t in extra_terms if t not in present and len(t) > 2]
    # unique, preserve order
    seen = set()
    uniq = []
    for t in added:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    if not uniq:
        return query + f" details iteration {iteration}"
    return query + " " + " ".join(uniq[:6])


class RetrievalAgent:
    def __init__(
        self,
        index: InMemoryHybridIndex,
        *,
        max_iters: int = 3,
        min_relevant: int = 1,
        grade_threshold: float = 0.12,
        router: QueryRouter | None = None,
        generator: Callable[[str, list[RetrievalHit]], str] | None = None,
    ):
        self.index = index
        self.max_iters = max_iters
        self.min_relevant = min_relevant
        self.grade_threshold = grade_threshold
        self.router = router or QueryRouter()
        self.generator = generator or extractive_answer

    def run(self, query: str) -> AgentAnswer:
        return self._execute(query)

    def stream(self, query: str) -> Iterator[AgentEvent | AgentAnswer]:
        answer = self._execute(query)
        for ev in answer.events:
            yield ev
        yield answer

    def _execute(self, query: str) -> AgentAnswer:
        events: list[AgentEvent] = []

        def emit(step: str, payload: dict) -> AgentEvent:
            ev = AgentEvent(step=step, payload=payload)
            events.append(ev)
            return ev

        route = self.router.route(query)
        emit("route", {"route": route.route.value, "reason": route.reason})
        emit("plan", {"query": query, "max_iters": self.max_iters})

        current = query
        last_result: RetrievalResult | None = None
        relevant: list[RetrievalHit] = []
        iteration = 0

        for iteration in range(1, self.max_iters + 1):
            result = self.index.retrieve(current)
            last_result = result
            emit(
                "search",
                {
                    "iteration": iteration,
                    "query": current,
                    "n_hits": len(result.hits),
                    "paths": result.paths[:8],
                },
            )
            grades = [(hit, grade_chunk(query, hit.chunk.text)) for hit in result.hits]
            relevant = [hit for hit, g in grades if g >= self.grade_threshold]
            emit(
                "grade_relevance",
                {
                    "iteration": iteration,
                    "n_relevant": len(relevant),
                    "threshold": self.grade_threshold,
                },
            )
            if len(relevant) >= self.min_relevant:
                break
            if iteration < self.max_iters:
                current = rewrite_query(current, result.hits, iteration)
                emit("rewrite", {"query": current})

        hits = relevant or (last_result.hits if last_result else [])
        if not hits:
            answer = AgentAnswer(
                text="No supporting files were retrieved for this query. Refusing to guess.",
                citations=[],
                confidence="empty",
                route=route,
                iterations=iteration,
                events=events,
                failed=True,
                fail_reason="empty_retrieval",
            )
            emit("respond", answer.as_dict())
            return answer

        citations = [
            Citation(
                path=h.chunk.path,
                start_byte=h.chunk.start_byte,
                end_byte=h.chunk.end_byte,
                chunk_id=h.chunk.chunk_id,
            )
            for h in hits[:8]
        ]
        if any(not c.resolve() for c in citations):
            answer = AgentAnswer(
                text="Generation failed: one or more citations could not be resolved to a file path and byte range.",
                citations=citations,
                confidence="unsupported",
                route=route,
                iterations=iteration,
                events=events,
                failed=True,
                fail_reason="unresolvable_citation",
            )
            emit("respond", answer.as_dict())
            return answer

        text = self.generator(query, hits)
        confidence = "high" if relevant else "low"
        if confidence == "low":
            text = (
                "Low-confidence answer — few retrieved chunks passed relevance grading.\n\n"
                + text
            )
        emit("verify", {"confidence": confidence, "n_citations": len(citations)})
        answer = AgentAnswer(
            text=text,
            citations=citations,
            confidence=confidence,
            route=route,
            iterations=iteration,
            events=events,
            failed=False,
        )
        emit("respond", answer.as_dict())
        return answer


def extractive_answer(query: str, hits: list[RetrievalHit]) -> str:
    """Deterministic extractive answer used when no LLM is wired in."""
    from app.graph.conflicts import annotate_for_generator

    q_tokens = set(tokenize(query))
    snippets = []
    for hit in hits[:4]:
        sentences = re.split(r"(?<=[.!?\n])\s+", hit.chunk.text.strip())
        best = max(
            sentences or [hit.chunk.text],
            key=lambda s: len(q_tokens & set(tokenize(s))),
        )
        snippets.append(f"[{hit.chunk.path}] {best.strip()[:400]}")
    if not snippets:
        return "No extractable answer."
    text = "\n".join(snippets)
    note = annotate_for_generator(hits).get("generator_note") or ""
    if note:
        return text + "\n\n" + note
    return text
