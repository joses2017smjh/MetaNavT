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
STUDENT_BACKENDS = ("numpy_lstsq", "lightgbm")
# LightGBM's own training API (lightgbm.train: no scikit-learn needed), 80 boosting rounds of
# depth-4 regression trees. One thread, a fixed seed and the deterministic mode, so the same rows
# give the same trees; every other parameter is LightGBM's default.
LIGHTGBM_ROUNDS = 80
LIGHTGBM_PARAMS = {
    "objective": "regression",
    "max_depth": 4,
    "learning_rate": 0.1,
    "num_threads": 1,
    "seed": 0,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}


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
    nothing a student could not read off its own features. A result with hits but no
    fused list (assembled outside InMemoryHybridIndex.retrieve) is an error, not an
    empty log."""
    if result.hits and not result.fused:
        raise ValueError("this retrieval result carries no fused list (it was not built by InMemoryHybridIndex.retrieve); nothing to log")
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


def _matrix(feature_dicts: Sequence[dict[str, float]]) -> np.ndarray:
    return np.array([feature_vector(f) for f in feature_dicts], dtype=np.float64).reshape(len(feature_dicts), len(FEATURE_NAMES))


def _with_intercept(X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((X.shape[0], 1))])


@dataclass
class DistilledReranker:
    feature_names: tuple[str, ...] = FEATURE_NAMES
    model: object | None = None
    backend: str = "none"

    def fit(self, triples: Sequence[RerankTriple], backend: str = "numpy_lstsq") -> "DistilledReranker":
        """Fit the student on the teacher's scores with the backend asked for, and only that one.

        numpy_lstsq: linear least squares with an explicit intercept column. On fixture v1
        the five file-type flags already sum to one for every chunk, so there the column
        changes no prediction; with other file types (a .pdf, a .toml) the fit would
        otherwise be forced through the origin.
        lightgbm: lightgbm.train(LIGHTGBM_PARAMS, ..., LIGHTGBM_ROUNDS). Without the library
        this raises ImportError (the `distill` extra); a fitting error propagates. Neither
        falls back to least squares, so `backend` always names the model that was fitted.
        """
        if backend not in STUDENT_BACKENDS:
            raise ValueError(f"unknown student backend {backend!r}; expected one of {STUDENT_BACKENDS}")
        if len(triples) < 8:
            raise ValueError("Need at least 8 triples to train a student")
        X = _matrix([t.features for t in triples])
        y = np.array([t.ce_score for t in triples], dtype=np.float64)
        if backend == "lightgbm":
            import lightgbm as lgb  # type: ignore[import-not-found]

            self.model = lgb.train(dict(LIGHTGBM_PARAMS), lgb.Dataset(X, label=y), num_boost_round=LIGHTGBM_ROUNDS)
        else:
            self.model = np.linalg.lstsq(_with_intercept(X), y, rcond=None)[0]
        self.backend = backend
        return self

    def predict_scores(self, feature_dicts: Sequence[dict[str, float]]) -> list[float]:
        if self.model is None:
            raise RuntimeError("DistilledReranker.fit() first")
        if not feature_dicts:
            return []
        X = _matrix(feature_dicts)
        if self.backend == "numpy_lstsq":
            return [float(s) for s in _with_intercept(X) @ np.asarray(self.model, dtype=np.float64)]
        return [float(s) for s in self.model.predict(X)]

    def __call__(
        self, query: str, pairs: Sequence[tuple[Chunk, float]]
    ) -> list[tuple[Chunk, float]]:
        scores = self.predict_scores(serving_features(query, pairs))
        ranked = list(zip([p[0] for p in pairs], scores))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked
