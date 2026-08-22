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

from app.graph.file_graph import RUN_ID_RE, extract_run_id
from app.retrieval.types import Chunk

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
    for field, value in FIELD_RE.findall(text or ""):
        if run_ent:
            triples.append(
                Triple(run_ent, f"has_{field.lower()}", value.lower(), path)
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
    for n in nodes:
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


def answer_global(question: str, communities: Sequence[Community]) -> str:
    """Route a corpus-level question onto community summaries."""
    q = (question or "").lower()
    scored = []
    q_toks = set(re.findall(r"[a-z0-9:]+", q))
    for c in communities:
        blob = (c.summary + " " + " ".join(c.members)).lower()
        overlap = len(q_toks & set(re.findall(r"[a-z0-9:]+", blob)))
        scored.append((overlap, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for s, c in scored[:3] if s > 0] or [c for _, c in scored[:1]]
    if not top:
        return "No community summary matched."
    return "\n".join(f"[community {c.community_id}] {c.summary}" for c in top)
