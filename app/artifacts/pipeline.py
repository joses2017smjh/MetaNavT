"""Orchestrate spec → generate → sandbox tests → HITL propose.

Does not write the tree. Callers (MCP) store the proposal and apply only
after approved=True.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.artifacts.manifest import ArtifactBundle, collect_run_artifact, score_badges
from app.artifacts.paper2code import paper2code
from app.artifacts.sandbox import ExecResult, run_sandboxed
from app.artifacts.spec import SpecCard, spec_from_query
from app.artifacts.templates import render
from app.graph.file_graph import extract_run_id
from app.graph.staleness import cluster_versions, prefer_current
from app.retrieval.bm25 import tokenize
from app.retrieval.hybrid import InMemoryHybridIndex
from app.retrieval.router import QueryRouter, RouteType
from app.retrieval.types import RetrievalHit

RESEARCH_SOURCE_TERMS = ("fusion", "dinov2", "trellis", "encoder", "render")


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

    def retrieve_research_evidence(self, query: str, k: int = 12) -> list[RetrievalHit]:
        """Hybrid retrieve, then bounded run-entity and source-code expansion."""
        base = self.index.retrieve(query, k=max(50, k), n=k).hits
        candidates = {hit.chunk.chunk_id: hit for hit in base}
        run_match = re.search(r"\brun[_\s-]?0*(\d+)\b", query or "", re.I)
        run_id = str(int(run_match.group(1))) if run_match else None

        if run_id:
            for chunk in self.index.chunks:
                if extract_run_id(chunk.path, chunk.text) == run_id:
                    candidates.setdefault(
                        chunk.chunk_id,
                        RetrievalHit(chunk=chunk, score=0.0, rank=len(candidates) + 1),
                    )

        entity_blob = "\n".join(hit.chunk.text for hit in candidates.values()).lower()
        source_terms = [term for term in RESEARCH_SOURCE_TERMS if term in entity_blob]
        if source_terms:
            for chunk in self.index.chunks:
                if Path(chunk.path).suffix.lower() != ".py":
                    continue
                blob = f"{chunk.path} {chunk.text}".lower()
                if any(term in blob for term in source_terms):
                    candidates.setdefault(
                        chunk.chunk_id,
                        RetrievalHit(chunk=chunk, score=0.0, rank=len(candidates) + 1),
                    )

        current = prefer_current(
            list(candidates.values()),
            cluster_versions(self.index.chunks),
            query,
        )
        base_rank = {hit.chunk.chunk_id: rank for rank, hit in enumerate(base, start=1)}
        q_tokens = set(tokenize(query))

        def evidence_key(hit: RetrievalHit) -> tuple:
            path = hit.chunk.path.replace("\\", "/")
            lower = path.lower()
            exact_config = bool(
                run_id
                and lower == f"configs/run_{int(run_id):03d}.yaml"
            )
            if exact_config:
                role = 0
            elif lower.startswith("paper/"):
                role = 1
            elif lower.endswith(".py"):
                role = 2
            elif lower.startswith("logs/"):
                role = 3
            elif lower.endswith(".sbatch"):
                role = 4
            elif "checkpoint" in lower:
                role = 5
            else:
                role = 6
            source_overlap = len(
                set(source_terms) & set(tokenize(f"{path} {hit.chunk.text}"))
            )
            query_overlap = len(q_tokens & set(tokenize(f"{path} {hit.chunk.text}")))
            return (
                role,
                -source_overlap,
                -query_overlap,
                base_rank.get(hit.chunk.chunk_id, 10_000),
                path,
            )

        ordered = sorted(current, key=evidence_key)
        for rank, hit in enumerate(ordered, start=1):
            hit.rank = rank
        return ordered[:k]

    def produce(self, query: str) -> ProposedArtifact:
        route = self.router.route(query)
        is_research = (
            route.route == RouteType.RESEARCH_ARTIFACT
            or "paper" in query.lower()
            or "reproduc" in query.lower()
        )
        hits = (
            self.retrieve_research_evidence(query)
            if is_research
            else self.retrieve(query)
        )
        if is_research:
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
        hits = self.retrieve_research_evidence(
            f"reproduce run {run_id} learning rate encoder fusion from the paper"
        )
        p2c = paper2code(f"reproduce run {run_id} from the paper", hits)
        executed = run_sandboxed(p2c.code)
        bundle.badges = score_badges(bundle, tests_passed=executed.ok)
        bundle.citations = p2c.spec.citations
        return bundle
