"""Phase 9: HippoRAG-style Personalized PageRank over GraphRAG triples.

Gutiérrez et al., HippoRAG 2 (ICML 2025): seed entities from the query, run PPR
on the triple graph, boost files that evidence high-PPR nodes.

This is a rerank/boost of already fused hits — not hop expansion.
simple_factual stays hops=0; apply only for aggregation / multi-hop / comparative.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Sequence

import numpy as np

from app.graph.file_graph import extract_run_id
from app.graph.graphrag import Triple, extract_entities, extract_triples
from app.retrieval.hybrid import RetrievalHit
from app.retrieval.router import QueryRouter, RouteType
from app.retrieval.types import Chunk

PPR_CATEGORIES = frozenset({"aggregation", "multi_hop", "comparative"})


def query_seeds(query: str) -> list[str]:
    ents = list(extract_entities("", query or ""))
    run = extract_run_id("", query or "")
    if run:
        ents.append(f"run:{run}")
    for m in re.finditer(r"\brun\s+(\d+)\b", query or "", re.I):
        ents.append(f"run:{int(m.group(1))}")
    seen: set[str] = set()
    out: list[str] = []
    for e in ents:
        if e in seen:
            continue
        seen.add(e)
        out.append(e)
    return out


def triples_from_chunks(chunks: Sequence[Chunk]) -> list[Triple]:
    triples: list[Triple] = []
    for chunk in chunks:
        triples.extend(extract_triples(chunk.path, chunk.text))
    return triples


def adjacency(triples: Sequence[Triple]) -> dict[str, dict[str, float]]:
    adj: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for t in triples:
        adj[t.src][t.dst] += 1.0
        adj[t.dst][t.src] += 1.0
    return adj


def personalized_pagerank(
    adj: dict[str, dict[str, float]],
    seeds: Sequence[str],
    *,
    damping: float = 0.85,
    n_iter: int = 25,
) -> dict[str, float]:
    nodes = sorted(adj.keys())
    if not nodes:
        return {}
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    w = np.zeros((n, n), dtype=np.float64)
    for src, dests in adj.items():
        j = idx[src]
        total = sum(dests.values())
        if total <= 0:
            continue
        for dst, weight in dests.items():
            w[idx[dst], j] += weight / total
    p = np.zeros(n, dtype=np.float64)
    seed_set = [s for s in seeds if s]
    for s in seed_set:
        if s in idx:
            p[idx[s]] += 1.0
        sl = s.lower()
        for node, i in idx.items():
            if sl and sl in node.lower():
                p[i] += 0.5
    if p.sum() <= 0:
        p[:] = 1.0 / n
    else:
        p /= p.sum()
    personal = p.copy()
    for _ in range(n_iter):
        p = damping * (w @ p) + (1.0 - damping) * personal
    return {nodes[i]: float(p[i]) for i in range(n)}


def path_scores(triples: Sequence[Triple], ppr: dict[str, float]) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for t in triples:
        mass = ppr.get(t.src, 0.0) + ppr.get(t.dst, 0.0)
        if t.evidence and mass:
            scores[t.evidence] += mass
    return dict(scores)


def boost_hits(
    hits: Sequence[RetrievalHit],
    path_ppr: dict[str, float],
    *,
    alpha: float = 1.0,
) -> list[RetrievalHit]:
    mx = max(path_ppr.values()) if path_ppr else 1.0
    mx = mx if mx > 0 else 1.0
    rescored: list[RetrievalHit] = []
    for hit in hits:
        extra = path_ppr.get(hit.chunk.path, 0.0) / mx
        rescored.append(
            RetrievalHit(
                chunk=hit.chunk,
                score=float(hit.score) + alpha * extra,
                rank=hit.rank,
                bm25=hit.bm25,
                dense=hit.dense,
                rrf=hit.rrf,
                rerank=hit.rerank,
            )
        )
    rescored.sort(key=lambda h: h.score, reverse=True)
    out = []
    for rank, hit in enumerate(rescored, start=1):
        out.append(
            RetrievalHit(
                chunk=hit.chunk,
                score=hit.score,
                rank=rank,
                bm25=hit.bm25,
                dense=hit.dense,
                rrf=hit.rrf,
                rerank=hit.rerank,
            )
        )
    return out


def should_apply(query: str, *, category: str = "", router: QueryRouter | None = None) -> bool:
    if (category or "").lower() in PPR_CATEGORIES:
        return True
    decision = (router or QueryRouter()).route(query)
    return decision.route == RouteType.AGGREGATION


def apply_hipporag(
    query: str,
    hits: Sequence[RetrievalHit],
    triples: Sequence[Triple],
    *,
    category: str = "",
    router: QueryRouter | None = None,
    alpha: float = 1.0,
) -> list[RetrievalHit]:
    """No-op on simple_factual. Boost aggregation / multi-hop fused hits."""
    if not hits or not triples or not should_apply(query, category=category, router=router):
        return list(hits)
    seeds = query_seeds(query)
    ppr = personalized_pagerank(adjacency(triples), seeds)
    return boost_hits(hits, path_scores(triples, ppr), alpha=alpha)
