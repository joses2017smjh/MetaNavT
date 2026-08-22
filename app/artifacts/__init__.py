"""Phase 11: research artifacts + production code (academia and industry, 2025–2026).

Academic: ACM artifact badges, Paper2Code plan→analyze→generate (Seo et al.,
ICLR 2026), execution-grounded claim checks (AutoResearch / SciCode).
Industry: spec card before code (Codex / Cursor), SEARCH/REPLACE patches
(Aider / OpenHands), restricted sandbox (E2B-style, no pip), HITL apply
(same contract as propose_move). Claude Artifacts / v0 templates without
auto-writing the tree.

LlamaIndex-free. The legacy tool in app.engine.tools.artifact still exists
for the FastAPI agent; this package is what eval and MCP call.
"""

from app.artifacts.manifest import (
    ArtifactBundle,
    ArtifactFile,
    Badge,
    collect_run_artifact,
    score_badges,
)
from app.artifacts.paper2code import paper2code
from app.artifacts.patch import FilePatch, apply_search_replace, parse_search_replace
from app.artifacts.pipeline import ArtifactAgent, ProposedArtifact
from app.artifacts.sandbox import ExecResult, run_sandboxed
from app.artifacts.spec import SpecCard, spec_from_query

__all__ = [
    "ArtifactBundle",
    "ArtifactFile",
    "Badge",
    "collect_run_artifact",
    "score_badges",
    "paper2code",
    "FilePatch",
    "apply_search_replace",
    "parse_search_replace",
    "ArtifactAgent",
    "ProposedArtifact",
    "ExecResult",
    "run_sandboxed",
    "SpecCard",
    "spec_from_query",
]
