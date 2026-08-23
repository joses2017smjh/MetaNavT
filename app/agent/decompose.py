"""Query decomposition: break a complex question into retrievable sub-queries.

Multi-hop and comparative questions ("which run with dinov2 had the lowest
RMSE") require evidence from multiple files.  Decomposition retrieves for
each sub-question separately, then merges via RRF — complementing the
multi-query paraphrase expansion in deep_research.py.

CI path: heuristic decomposition extracts entity-bearing fragments without
an LLM, so `make bench` stays GPU-free.
"""

from __future__ import annotations

import re
from typing import Callable

from app.retrieval.router import RouteType


def heuristic_decompose(query: str) -> list[str]:
    """Split a query into sub-queries using syntactic cues.

    Handles:
    - "X and Y" → two sub-queries
    - "which run ... had the lowest ..." → entity lookup + aggregation
    - "compare A with B" → two lookups
    - Falls back to [query] if no decomposition fires
    """
    q = (query or "").strip()
    if not q:
        return [q]

    subs: list[str] = []

    compare = re.match(
        r"(?:compare|difference between|diff between)\s+(.+?)\s+(?:and|with|vs\.?)\s+(.+)",
        q,
        re.I,
    )
    if compare:
        subs.append(compare.group(1).strip())
        subs.append(compare.group(2).strip())
        return _dedup(subs)

    conj = re.split(r"\s+and\s+(?:also\s+)?", q, flags=re.I)
    if len(conj) >= 2 and all(len(c.split()) >= 3 for c in conj):
        return _dedup([c.strip() for c in conj if c.strip()])

    superlative = re.match(
        r"(?:which|what)\s+(\w+)\s+.*?(?:had|has|with)\s+the\s+"
        r"(?:lowest|highest|best|worst|most|least|maximum|minimum)\s+(.+)",
        q,
        re.I,
    )
    if superlative:
        entity_type = superlative.group(1)
        metric = superlative.group(2).strip().rstrip("?")
        subs.append(f"list all {entity_type} {metric}")
        subs.append(q)
        return _dedup(subs)

    chain = re.match(
        r"(.+?)\s+(?:then|and then|,\s*then)\s+(.+)", q, re.I
    )
    if chain:
        subs.append(chain.group(1).strip())
        subs.append(chain.group(2).strip())
        return _dedup(subs)

    return [q]


def llm_decompose(
    query: str,
    complete: Callable[[str], str],
    max_subs: int = 4,
) -> list[str]:
    """Use an LLM to decompose a query into sub-queries."""
    prompt = (
        "Break the following question into independent sub-questions that "
        "can each be answered by searching a file corpus. Each sub-question "
        f"should be self-contained. Return at most {max_subs} sub-questions, "
        "one per line, no numbering. If the question is already simple, "
        "return it unchanged.\n\n"
        f"Question: {query}\n\nSub-questions:"
    )
    raw = (complete(prompt) or "").strip()
    if not raw:
        return heuristic_decompose(query)
    subs = []
    for line in raw.splitlines():
        line = re.sub(r"^[\d.)\-\s]+", "", line).strip()
        if line and len(line) > 5:
            subs.append(line)
        if len(subs) >= max_subs:
            break
    return _dedup(subs) if subs else [query]


def should_decompose(query: str, route_type: RouteType | None = None) -> bool:
    """Decide whether decomposition would help this query."""
    if route_type in (RouteType.LEXICAL_PATH, RouteType.CODE_GEN):
        return False
    q = (query or "").lower()
    if any(
        w in q
        for w in (
            "compare", "difference", "and also", "then ", "which run",
            "what run", "lowest", "highest", "best", "worst",
        )
    ):
        return True
    if route_type in (RouteType.AGGREGATION, RouteType.RESEARCH_ARTIFACT):
        return True
    return q.count(" and ") >= 1 and len(q.split()) > 10


def _dedup(subs: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in subs:
        key = s.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    return out if out else [""]
