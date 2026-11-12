"""GraphRAG-lite: LLM-optional entity/relation extraction + community summaries.

Deterministic extractors first (run ids, encoders, config fields, file refs).
An optional `llm` callable can add extra (entity, relation, entity) triples.
Communities are connected components; each gets an extractive (or LLM) summary.
This is what answers global 'what is in this corpus' questions.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from app.graph.file_graph import extract_run_id
from app.retrieval.types import Chunk, RetrievalHit

ENCODER_RE = re.compile(r"\b(dinov2|resnet50|clip|vit-b/?\d*)\b", re.IGNORECASE)
BARK_RE = re.compile(r"\bbark(?:_type)?\s*[:=]\s*([A-Za-z]+)", re.IGNORECASE)
FIELD_RE = re.compile(
    r"\b(learning_rate|val_rmse|num_pairs|fusion)\s*[:=]\s*([A-Za-z0-9.eE+\-]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Triple:
    src: str
    relation: str
    dst: str
    evidence: str = ""


@dataclass
class Community:
    community_id: int
    members: list[str]
    summary: str
    triples: list[Triple] = field(default_factory=list)


def extract_entities(path: str, text: str) -> set[str]:
    ents: set[str] = set()
    run = extract_run_id(path, text)
    if run:
        ents.add(f"run:{run}")
    for m in ENCODER_RE.finditer(text or ""):
        ents.add(f"encoder:{m.group(1).lower()}")
    for m in BARK_RE.finditer(text or ""):
        ents.add(f"bark:{m.group(1).lower()}")
    if path:
        ents.add(f"file:{path.replace(chr(92), '/')}")
    return ents


def extract_triples(path: str, text: str) -> list[Triple]:
    triples: list[Triple] = []
    run = extract_run_id(path, text)
    run_ent = f"run:{run}" if run else None
    for enc in ENCODER_RE.findall(text or ""):
        if run_ent:
            triples.append(Triple(run_ent, "uses_encoder", f"encoder:{enc.lower()}", path))
    for bark in BARK_RE.findall(text or ""):
        if run_ent:
            triples.append(Triple(run_ent, "uses_bark", f"bark:{bark.lower()}", path))
    for field_name, value in FIELD_RE.findall(text or ""):
        if run_ent:
            triples.append(
                Triple(run_ent, f"has_{field_name.lower()}", value.lower(), path)
            )
    if run_ent and path:
        triples.append(Triple(run_ent, "documented_in", f"file:{path.replace(chr(92), '/')}", path))
    return triples


def _components(triples: Sequence[Triple]) -> list[set[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    nodes: set[str] = set()
    for t in triples:
        nodes.add(t.src)
        nodes.add(t.dst)
        adj[t.src].add(t.dst)
        adj[t.dst].add(t.src)
    seen: set[str] = set()
    out: list[set[str]] = []
    for n in sorted(nodes):  # a set of strings iterates in PYTHONHASHSEED order; equal-size components keep this order
        if n in seen:
            continue
        stack = [n]
        comp: set[str] = set()
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.add(u)
            stack.extend(adj[u] - seen)
        out.append(comp)
    out.sort(key=len, reverse=True)
    return out


def _extractive_summary(members: Iterable[str], triples: Sequence[Triple]) -> str:
    runs = sorted(m.split(":", 1)[1] for m in members if m.startswith("run:"))
    encs = sorted({m.split(":", 1)[1] for m in members if m.startswith("encoder:")})
    files = [m.split(":", 1)[1] for m in members if m.startswith("file:")]
    bits = []
    if runs:
        bits.append(f"runs {', '.join(runs)}")
    if encs:
        bits.append(f"encoders {', '.join(encs)}")
    if files:
        bits.append(f"{len(files)} files")
    rels = {t.relation for t in triples if t.src in members or t.dst in members}
    if rels:
        bits.append("relations: " + ", ".join(sorted(rels)[:6]))
    return "; ".join(bits) or "empty community"


def build_graphrag(
    chunks: Sequence[Chunk],
    *,
    llm: Callable[[str], str] | None = None,
    min_community: int = 2,
) -> list[Community]:
    triples: list[Triple] = []
    for chunk in chunks:
        triples.extend(extract_triples(chunk.path, chunk.text))
        if llm:
            prompt = (
                "Extract (subject, relation, object) triples from this chunk. "
                f"Path: {chunk.path}\n{chunk.text[:800]}"
            )
            raw = llm(prompt) or ""
            for line in raw.splitlines():
                parts = [p.strip() for p in re.split(r"[|,>/]", line) if p.strip()]
                if len(parts) >= 3:
                    triples.append(Triple(parts[0], parts[1], parts[2], chunk.path))
    comps = _components(triples)
    communities = []
    cid = 0
    for comp in comps:
        if len(comp) < min_community:
            continue
        local = [t for t in triples if t.src in comp or t.dst in comp]
        if llm:
            summary = llm(
                "Summarize this entity community in 2 sentences:\n"
                + ", ".join(sorted(comp)[:40])
            ).strip() or _extractive_summary(comp, local)
        else:
            summary = _extractive_summary(comp, local)
        communities.append(
            Community(
                community_id=cid,
                members=sorted(comp),
                summary=summary,
                triples=local,
            )
        )
        cid += 1
    return communities


def _overlap_scores(question: str, communities: Sequence[Community]) -> list[tuple[int, Community]]:
    """(token overlap, community), best first; ties keep community order."""
    q_toks = set(re.findall(r"[a-z0-9:]+", (question or "").lower()))
    scored = []
    for c in communities:
        blob = (c.summary + " " + " ".join(c.members)).lower()
        overlap = len(q_toks & set(re.findall(r"[a-z0-9:]+", blob)))
        scored.append((overlap, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def answer_global(question: str, communities: Sequence[Community]) -> str:
    """Route a corpus-level question onto community summaries."""
    scored = _overlap_scores(question, communities)
    top = [c for s, c in scored[:3] if s > 0] or [c for _, c in scored[:1]]
    if not top:
        return "No community summary matched."
    return "\n".join(f"[community {c.community_id}] {c.summary}" for c in top)


def match_communities(question: str, communities: Sequence[Community], top: int = 3) -> list[Community]:
    """The communities whose summary or members share a token with the question
    (the same score as answer_global), best first. Empty when nothing overlaps:
    retrieval, unlike the global answer, gets no default community."""
    return [c for s, c in _overlap_scores(question, communities)[:top] if s > 0]


def community_files(community: Community) -> set[str]:
    return {m.split(":", 1)[1] for m in community.members if m.startswith("file:")}


def boost_by_communities(hits: Sequence[RetrievalHit], matched: Sequence[Community], *, rrf_k: int = 60) -> list[RetrievalHit]:
    """Rank-fuse the base order with community membership, without adding candidates.

    A hit whose file belongs to the i-th matched community gets 1/(rrf_k + i)
    on top of 1/(rrf_k + base_rank). Every file of one community receives the
    same bonus, so the summaries can only separate files that sit in different
    communities: with a single community this is a member / non-member split.
    """
    community_rank: dict[str, int] = {}
    for i, c in enumerate(matched, start=1):
        for path in community_files(c):
            community_rank.setdefault(path, i)
    rescored = []
    for base_rank, hit in enumerate(hits, start=1):
        score = 1.0 / (rrf_k + base_rank)
        if hit.chunk.path in community_rank:
            score += 1.0 / (rrf_k + community_rank[hit.chunk.path])
        rescored.append((hit, score))
    rescored.sort(key=lambda pair: pair[1], reverse=True)
    return [
        RetrievalHit(chunk=h.chunk, score=score, rank=rank, bm25=h.bm25, dense=h.dense, rrf=h.rrf, rerank=h.rerank)
        for rank, (h, score) in enumerate(rescored, start=1)
    ]


def community_stats(communities: Sequence[Community], n_corpus_files: int) -> dict:
    """What a results row needs to say whether the communities can discriminate."""
    member_files = set().union(*(community_files(c) for c in communities)) if communities else set()
    return {
        "n_communities": len(communities),
        "community_sizes": [len(c.members) for c in communities],
        "n_member_files": len(member_files),
        "n_corpus_files": n_corpus_files,
        "discriminates": len(communities) > 1,
        "clustering": "connected components (no Leiden)",
        "summaries": "extractive (no LLM)",
    }
