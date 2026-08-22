"""Rule-based query router.

Routes:
  lexical_path      — looks like a filename/path; skip embed+rerank
  aggregation_query — count/min/max/which-run questions; metadata path
  semantic          — default hybrid+rerank

Zero training cost. Log every decision next to retrieval outcome.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class RouteType(str, Enum):
    LEXICAL_PATH = "lexical_path"
    AGGREGATION = "aggregation_query"
    SEMANTIC = "semantic"


PATH_RE = re.compile(
    r"""
    (?:^|[\s`'\"(])
    (
        (?:[\w.\-]+/)+[\w.\-]+          # path with slash
        | [\w.\-]+\.(?:ya?ml|jsonl?|py|md|txt|out|sbatch|csv|pdf|ckpt|pt|png)
        | (?:config_)?run[_\-]?0*\d+\.(?:ya?ml|json)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

AGG_RE = re.compile(
    r"\b("
    r"how many|which run|which ablation|lowest|highest|average|mean|"
    r"fewest|most|count|min(?:imum)?|max(?:imum)?|compare|versus|vs\.?"
    r")\b",
    re.IGNORECASE,
)

EXACT_TOKEN_RE = re.compile(
    r"(--[\w\-]+(?:=\S+)?|`[^`]+`|\"[^\"]+\"|'[^']+')"
)


@dataclass(frozen=True)
class RouteDecision:
    route: RouteType
    reason: str
    matched: str | None = None

    def skip_embed(self) -> bool:
        return self.route == RouteType.LEXICAL_PATH

    def skip_rerank(self) -> bool:
        return self.route in {RouteType.LEXICAL_PATH, RouteType.AGGREGATION}


class QueryRouter:
    def route(self, query: str) -> RouteDecision:
        q = (query or "").strip()
        path_match = PATH_RE.search(q)
        if path_match:
            return RouteDecision(
                route=RouteType.LEXICAL_PATH,
                reason="path-like token",
                matched=path_match.group(1),
            )
        agg_match = AGG_RE.search(q)
        if agg_match:
            return RouteDecision(
                route=RouteType.AGGREGATION,
                reason="aggregation verb",
                matched=agg_match.group(1).lower(),
            )
        exact = EXACT_TOKEN_RE.search(q)
        extra = f"; exact-token {exact.group(1)}" if exact else ""
        return RouteDecision(
            route=RouteType.SEMANTIC,
            reason="default semantic" + extra,
            matched=exact.group(1) if exact else None,
        )
