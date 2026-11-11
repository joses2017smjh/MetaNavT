"""Distill a cross-encoder teacher into a cheap student ranker.

Teacher: the cross-encoder's scores over the fused candidates (make bench-rerank-arms
computes them for the train questions; app/eval/rerank_arms.py).
Student: least squares or gradient-boosted trees over the features it has when it
serves (serving_features): the fused score and rank, text overlap, path facts.
Mirrors the Depth capstone teacher/student pattern.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from app.retrieval.hybrid import Chunk
from app.retrieval.rerank import FEATURE_NAMES, RerankTriple, extract_features, feature_vector

NO_RERANK_TEACHER = "fused score (no rerank ran)"


def serving_features(query: str, pairs: Sequence[tuple[Chunk, float]]) -> list[dict[str, float]]:
    """The features a student has when it reranks a fused list: the fused score and
    rank, text overlap and path facts. No dense or BM25 score, and never the teacher."""
    max_mtime = max((c.mtime for c, _ in pairs), default=1.0) or 1.0
    return [
        extract_features(query, chunk, rrf=rrf, rrf_rank=rank, max_mtime=max_mtime)
        for rank, (chunk, rrf) in enumerate(pairs, start=1)
    ]


def triples_from_pairs(query: str, pairs: Sequence[tuple[Chunk, float]], teacher_scores: dict[str, float]) -> list[RerankTriple]:
    """Training rows whose features are exactly what DistilledReranker.__call__ computes:
    the serving features of the fused list, in fused order, labelled with the teacher's
    score. Never the dense or BM25 score (absent when serving) and never the rank after
    the teacher reranked the list (that would leak the label into a feature)."""
    return [
        RerankTriple(query=query, chunk_id=chunk.chunk_id, path=chunk.path, ce_score=float(teacher_scores[chunk.chunk_id]), features=feats)
        for (chunk, _), feats in zip(pairs, serving_features(query, pairs))
        if chunk.chunk_id in teacher_scores
    ]


def bench_log_triples(query: str, result) -> list[RerankTriple]:
    """The rows `make bench` logs for one retrieval (app/eval/harness.py, <sha>.triples.jsonl).

    One row per candidate of the fused list the reranker received (result.fused, the
    hybrid index's RetrievalResult), with triples_from_pairs' serving features, labelled
    with the reranker's score (teacher "reranker"). When nothing reranked the list (the
    row has no reranker, or the router skipped it) the label is the fused score, which
    is also the rrf_score feature: such rows carry teacher NO_RERANK_TEACHER and teach
    nothing a student could not read off its own features."""
    reranked = {h.chunk.chunk_id: h.rerank for h in result.hits if h.rerank is not None}
    labels = {chunk.chunk_id: reranked.get(chunk.chunk_id, score) for chunk, score in result.fused}
    return [
        RerankTriple(query=t.query, chunk_id=t.chunk_id, path=t.path, ce_score=t.ce_score, features=t.features,
                     teacher="reranker" if t.chunk_id in reranked else NO_RERANK_TEACHER)
        for t in triples_from_pairs(query, result.fused, labels)
    ]


def write_triples(path: str | Path, triples: Iterable[RerankTriple]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for t in triples:
            f.write(
                json.dumps(
                    {
                        "query": t.query,
                        "chunk_id": t.chunk_id,
                        "path": t.path,
                        "ce_score": t.ce_score,
                        "teacher": t.teacher,
                        "features": t.features,
                    }
                )
                + "\n"
            )


@dataclass
class DistilledReranker:
    feature_names: tuple[str, ...] = FEATURE_NAMES
    model: object | None = None
    backend: str = "none"

    def fit(self, triples: Sequence[RerankTriple]) -> "DistilledReranker":
        if len(triples) < 8:
            raise ValueError("Need at least 8 triples to train a student")
        X = np.array([feature_vector(t.features) for t in triples], dtype=np.float64)
        y = np.array([t.ce_score for t in triples], dtype=np.float64)
        try:
            import lightgbm as lgb  # type: ignore

            model = lgb.LGBMRegressor(
                n_estimators=80,
                max_depth=4,
                learning_rate=0.1,
                objective="regression",
                verbosity=-1,
            )
            model.fit(X, y)
            self.model = model
            self.backend = "lightgbm"
            return self
        except Exception:
            model = np.linalg.lstsq(X, y, rcond=None)[0]
            self.model = model
            self.backend = "numpy_lstsq"
            return self

    def predict_scores(self, feature_dicts: Sequence[dict[str, float]]) -> list[float]:
        if self.model is None:
            raise RuntimeError("DistilledReranker.fit() first")
        X = np.array([feature_vector(f) for f in feature_dicts], dtype=np.float64)
        if self.backend == "numpy_lstsq":
            raw = X @ np.asarray(self.model, dtype=np.float64)
            return [float(s) for s in raw]
        return [float(s) for s in self.model.predict(X)]

    def __call__(
        self, query: str, pairs: Sequence[tuple[Chunk, float]]
    ) -> list[tuple[Chunk, float]]:
        scores = self.predict_scores(serving_features(query, pairs))
        ranked = list(zip([p[0] for p in pairs], scores))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked
