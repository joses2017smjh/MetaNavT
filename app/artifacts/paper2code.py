"""Paper2Code-style plan → analyze → generate over this corpus.

Seo et al., Paper2Code (ICLR 2026): specialized stages, not one-shot
'write the repo from the PDF'. We stay inside the frozen file tree:
paper drafts + src + configs are the gold, not a downloaded GitHub repo.

Claim-support (AutoResearch 2026): every generated identifier that looks
like a metric or hyperparameter must appear in a cited file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from app.artifacts.spec import SpecCard, spec_from_query
from app.artifacts.templates import render
from app.graph.conflicts import detect_semantic_conflicts, detect_structural_conflicts
from app.retrieval.types import RetrievalHit

CLAIM_FIELDS = ("learning_rate", "val_rmse", "encoder", "fusion", "bark_type", "num_pairs")


@dataclass
class Paper2CodeResult:
    plan: dict
    analysis: dict
    code: str
    spec: SpecCard
    unsupported_claims: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "plan": self.plan,
            "analysis": self.analysis,
            "code": self.code,
            "spec": self.spec.as_dict(),
            "unsupported_claims": self.unsupported_claims,
            "faithful": not self.unsupported_claims,
        }


def plan_repo(hits: Sequence[RetrievalHit], query: str) -> dict:
    files = []
    seen = set()
    for hit in hits:
        if hit.chunk.path in seen:
            continue
        seen.add(hit.chunk.path)
        files.append(
            {
                "path": hit.chunk.path,
                "role": "paper" if hit.chunk.path.endswith(".md") else "code" if hit.chunk.path.endswith(".py") else "config",
            }
        )
    return {
        "query": query,
        "architecture": "retrieve cited files, then emit one reproduce.py",
        "files": files[:12],
        "config": "configs/run_*.yaml when a run id is present",
    }


def analyze_hits(hits: Sequence[RetrievalHit]) -> dict:
    blob = "\n".join(h.chunk.text for h in hits)
    fields = {}
    for name in CLAIM_FIELDS:
        m = re.search(rf"\b{name}\s*[:=]\s*([A-Za-z0-9.eE+\-]+)", blob, re.I)
        if m:
            fields[name] = m.group(1)
    if "dinov2" in blob.lower():
        fields.setdefault("encoder", "dinov2")
    conflicts = detect_structural_conflicts(list(hits)) + detect_semantic_conflicts(list(hits))
    return {
        "fields": fields,
        "conflicts": [c.as_dict() for c in conflicts],
        "n_hits": len(hits),
    }


def claim_support(code: str, hits: Sequence[RetrievalHit]) -> list[str]:
    """Return claim tokens in the generated code that no citation supports."""
    blob = "\n".join(h.chunk.text for h in hits).lower()
    unsupported = []
    for name in CLAIM_FIELDS:
        for m in re.finditer(rf"{name}\s*[:=]\s*([A-Za-z0-9.eE+\-]+)", code, re.I):
            val = m.group(1).lower()
            if val not in blob and name.lower() not in blob:
                unsupported.append(f"{name}={m.group(1)}")
    # encoder string literals
    for enc in ("dinov2", "resnet50", "clip"):
        if re.search(rf"ENCODER\s*=\s*['\"]{enc}['\"]", code) and enc not in blob:
            unsupported.append(f"encoder={enc}")
    return unsupported


def paper2code(query: str, hits: Sequence[RetrievalHit]) -> Paper2CodeResult:
    hits = list(hits)
    spec = spec_from_query(query, hits)
    spec.template = "research-repro"
    spec.file_path = "artifacts/reproduce.py"
    planned = plan_repo(hits, query)
    analysis = analyze_hits(hits)
    code = render(spec, hits)
    return Paper2CodeResult(
        plan=planned,
        analysis=analysis,
        code=code,
        spec=spec,
        unsupported_claims=claim_support(code, hits),
    )
