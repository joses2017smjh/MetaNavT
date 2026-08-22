"""Orchestrate spec → generate → sandbox tests → HITL propose.

Does not write the tree. Callers (MCP) store the proposal and apply only
after approved=True.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from app.artifacts.manifest import ArtifactBundle, collect_run_artifact, score_badges
from app.artifacts.paper2code import paper2code
from app.artifacts.sandbox import ExecResult, run_sandboxed
from app.artifacts.spec import SpecCard, spec_from_query
from app.artifacts.templates import render
from app.retrieval.hybrid import InMemoryHybridIndex
from app.retrieval.router import QueryRouter, RouteType


@dataclass
class ProposedArtifact:
    plan_id: str
    spec: SpecCard
    code: str
    exec_result: ExecResult
    kind: str  # code | research
    extra: dict = field(default_factory=dict)
    approved: bool = False
    applied: bool = False

    def as_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "kind": self.kind,
            "spec": self.spec.as_dict(),
            "code": self.code,
            "exec": self.exec_result.as_dict(),
            "extra": self.extra,
            "status": "applied" if self.applied else "pending_approval",
            "note": "Destructive write. Call apply_artifact only after human approval.",
        }


class ArtifactAgent:
    def __init__(self, index: InMemoryHybridIndex, *, router: QueryRouter | None = None):
        self.index = index
        self.router = router or QueryRouter()

    def retrieve(self, query: str, k: int = 8):
        return self.index.retrieve(query, n=k).hits[:k]

    def produce(self, query: str) -> ProposedArtifact:
        hits = self.retrieve(query)
        route = self.router.route(query)
        if route.route == RouteType.RESEARCH_ARTIFACT or "paper" in query.lower() or "reproduc" in query.lower():
            p2c = paper2code(query, hits)
            result = run_sandboxed(p2c.code)
            spec = p2c.spec
            extra = p2c.as_dict()
            extra.pop("code", None)
            kind = "research"
            code = p2c.code
        else:
            spec = spec_from_query(query, hits)
            code = render(spec, hits)
            result = run_sandboxed(code)
            extra = {"route": route.route.value}
            kind = "code"
        return ProposedArtifact(
            plan_id=str(uuid.uuid4()),
            spec=spec,
            code=code,
            exec_result=result,
            kind=kind,
            extra=extra,
        )

    def collect_run(self, files_root, run_id: str) -> ArtifactBundle:
        bundle = collect_run_artifact(files_root, run_id)
        # optional functional badge: generate a repro from retrieved hits
        hits = self.retrieve(f"run {run_id} learning rate encoder fusion")
        p2c = paper2code(f"reproduce run {run_id} from the paper", hits)
        executed = run_sandboxed(p2c.code)
        bundle.badges = score_badges(bundle, tests_passed=executed.ok)
        bundle.citations = p2c.spec.citations
        return bundle
