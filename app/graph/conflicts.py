"""Staleness Tier 2: flag semantically conflicting facts in the retrieved set.

Tier 1 (version clusters) is deterministic at index time.
Tier 2 runs at query time only, over the top-k hits: same entity, disagreeing values.
An optional LLM classifier can catch paraphrase-level conflicts (two paper drafts).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

from app.graph.file_graph import extract_run_id
from app.retrieval.types import Chunk, RetrievalHit

FIELD_RE = re.compile(
    r"\b(learning_rate|val_rmse|num_pairs|fusion|encoder|bark_type)\s*[:=]\s*([A-Za-z0-9.eE+\-]+)",
    re.IGNORECASE,
)

NEGATION_PAIRS = (
    ("does help", "does not help"),
    ("fusion helps", "fusion does not"),
    ("not help", "does help"),
)


def _claim_text(text: str) -> str:
    """Normalize Markdown and punctuation before deterministic claim matching."""
    normalized = re.sub(r"[*_`~]+", "", (text or "").lower())
    normalized = re.sub(r"[^a-z0-9.]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


@dataclass
class Conflict:
    entity: str
    field: str
    values: list[str]
    paths: list[str]
    kind: str  # structural | semantic

    def as_dict(self) -> dict:
        return {
            "entity": self.entity,
            "field": self.field,
            "values": self.values,
            "paths": self.paths,
            "kind": self.kind,
        }


def _facts(chunk: Chunk) -> list[tuple[str, str, str]]:
    run = extract_run_id(chunk.path, chunk.text) or chunk.path
    entity = f"run:{run}" if extract_run_id(chunk.path, chunk.text) else f"file:{chunk.path}"
    out = []
    for field, value in FIELD_RE.findall(chunk.text or ""):
        out.append((entity, field.lower(), value.lower()))
    return out


def detect_structural_conflicts(hits: Sequence[RetrievalHit]) -> list[Conflict]:
    grouped: dict[tuple[str, str], dict[str, set[str]]] = {}
    for hit in hits:
        for entity, field, value in _facts(hit.chunk):
            slot = grouped.setdefault((entity, field), {})
            slot.setdefault(value, set()).add(hit.chunk.path)
    conflicts = []
    for (entity, field), values in grouped.items():
        if len(values) < 2:
            continue
        paths = sorted({p for ps in values.values() for p in ps})
        conflicts.append(
            Conflict(
                entity=entity,
                field=field,
                values=sorted(values),
                paths=paths,
                kind="structural",
            )
        )
    return conflicts


def detect_semantic_conflicts(
    hits: Sequence[RetrievalHit],
    llm: Callable[[str], str] | None = None,
) -> list[Conflict]:
    texts = [(h.chunk.path, _claim_text(h.chunk.text)) for h in hits]
    conflicts: list[Conflict] = []
    for a_path, a in texts:
        for b_path, b in texts:
            if a_path >= b_path:
                continue
            for pos, neg in NEGATION_PAIRS:
                if pos in a and neg in b or pos in b and neg in a:
                    conflicts.append(
                        Conflict(
                            entity="paper_claim",
                            field="conclusion",
                            values=[pos, neg],
                            paths=[a_path, b_path],
                            kind="semantic",
                        )
                    )
    if llm and len(hits) >= 2:
        blob = "\n---\n".join(
            f"{h.chunk.path}: {h.chunk.text[:400]}" for h in hits[:8]
        )
        verdict = llm(
            "Do any of these chunks state conflicting facts about the same entity? "
            "Reply CONFLICT <entity> | <summary> or OK.\n" + blob
        )
        if verdict and verdict.strip().upper().startswith("CONFLICT"):
            conflicts.append(
                Conflict(
                    entity="llm",
                    field="judge",
                    values=[verdict.strip()[:200]],
                    paths=[h.chunk.path for h in hits[:8]],
                    kind="semantic",
                )
            )
    # unique by paths+field
    seen = set()
    unique = []
    for c in conflicts:
        key = (c.field, tuple(c.paths), c.kind)
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    return unique


def annotate_for_generator(hits: Sequence[RetrievalHit], llm=None) -> dict:
    structural = detect_structural_conflicts(hits)
    semantic = detect_semantic_conflicts(hits, llm=llm)
    all_c = structural + semantic
    note = ""
    if all_c:
        lines = [
            "Conflicting sources in the retrieved set — address explicitly, do not silently pick one:"
        ]
        for c in all_c:
            lines.append(
                f"- {c.entity} {c.field}: {', '.join(c.values)} ({'; '.join(c.paths)})"
            )
        note = "\n".join(lines)
    return {"conflicts": [c.as_dict() for c in all_c], "generator_note": note}
