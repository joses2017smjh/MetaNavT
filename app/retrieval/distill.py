"""Distill a cross-encoder teacher into a cheap student ranker.

Teacher: cross-encoder scores logged during bench.
Student: logistic/GBT over {dense, BM25, RRF, overlap, path, recency, filetype}.
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


def triples_from_hits(query: str, hits, max_mtime: float = 1.0) -> list[RerankTriple]:
    triples = []
    for hit in hits:
        feats = extract_features(
            query,
            hit.chunk,
            dense=hit.dense,
            bm25=hit.bm25,
            rrf=hit.rrf,
            rrf_rank=hit.rank,
            max_mtime=max_mtime,
        )
        teacher = hit.rerank if hit.rerank is not None else hit.score
        triples.append(
            RerankTriple(
                query=query,
                chunk_id=hit.chunk.chunk_id,
                path=hit.chunk.path,
                ce_score=float(teacher),
                features=feats,
            )
        )
    return triples


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
                        "features": t.features,
                    }
                )
                + "\n"
            )


def distill_winner(triples: Sequence[RerankTriple]) -> DistilledReranker:
    """Fit the CI student on the winning teacher's logged scores."""
    return DistilledReranker().fit(triples)


def distill_from_path(path: str | Path) -> DistilledReranker:
    return distill_winner(load_triples(path))


def load_triples(path: str | Path) -> list[RerankTriple]:
    triples = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            triples.append(
                RerankTriple(
                    query=raw["query"],
                    chunk_id=raw["chunk_id"],
                    path=raw["path"],
                    ce_score=float(raw["ce_score"]),
                    features=raw["features"],
                )
            )
    return triples


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
        max_mtime = max((c.mtime for c, _ in pairs), default=1.0) or 1.0
        feats = []
        for rank, (chunk, rrf) in enumerate(pairs, start=1):
            feats.append(
                extract_features(
                    query,
                    chunk,
                    rrf=rrf,
                    rrf_rank=rank,
                    max_mtime=max_mtime,
                )
            )
        scores = self.predict_scores(feats)
        ranked = list(zip([p[0] for p in pairs], scores))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked
