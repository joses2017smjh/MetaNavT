"""M7: the prototype arms are evaluated without gold fields reaching retrieval."""

from __future__ import annotations

import inspect
from dataclasses import replace

import pytest

from app.eval.harness import BenchConfig, _retrieve
from app.eval.latency import StageTimer
from app.graph.hipporag import triples_from_chunks
from app.retrieval.hybrid import InMemoryHybridIndex
from app.retrieval.router import RouteType
from app.retrieval.types import Chunk


def _tiny_index() -> InMemoryHybridIndex:
    chunks = [
        Chunk("a", "configs/run_040.yaml", "run_id: 40 encoder: resnet50 learning_rate: 1e-4", 0, 60),
        Chunk("b", "configs/run_047.yaml", "run_id: 47 encoder: dinov2 learning_rate: 3e-4", 0, 60),
        Chunk("c", "slurm/run_047.sbatch", "run_id: 47 sbatch job for dinov2", 0, 40),
    ]
    return InMemoryHybridIndex(chunks, retrieve_k=5, enable_rerank=False, enable_router=False)


HIPPO = BenchConfig(name="hybrid+hipporag", mode="hybrid", enable_rerank=False, enable_router=False,
                    staleness_tier1=False, hipporag=True, e2e=False, log_triples=False)


def test_retrieve_has_no_gold_category_parameter():
    # The harness used to forward the gold category of a question into the PPR gate.
    assert "category" not in inspect.signature(_retrieve).parameters
    index = _tiny_index()
    with pytest.raises(TypeError):
        _retrieve(index, "which slurm job launched run 47", HIPPO, StageTimer(), {}, None, category="multi_hop")


def test_hipporag_gate_in_the_harness_is_the_router_not_the_gold_label(monkeypatch):
    import app.eval.harness as harness

    index = _tiny_index()
    triples = triples_from_chunks(index.chunks)
    seen: list[dict] = []
    real = harness.apply_hipporag

    def spy(query, hits, triples, **kwargs):
        seen.append(kwargs)
        return real(query, hits, triples, **kwargs)

    monkeypatch.setattr(harness, "apply_hipporag", spy)
    # Gold calls this kind of question multi_hop; the router does not call it aggregation.
    query = "which slurm job launched run 47"
    assert index.router.route(query).route != RouteType.AGGREGATION
    base, _ = _retrieve(index, query, replace(HIPPO, hipporag=False), StageTimer(), {}, None)
    boosted, _ = _retrieve(index, query, HIPPO, StageTimer(), {}, None, hippo_triples=triples)
    assert boosted == base, "PPR must not fire on a question the router does not route to aggregation"
    assert seen and all(not kw.get("category") for kw in seen)
    assert all(kw.get("router") is index.router for kw in seen)
