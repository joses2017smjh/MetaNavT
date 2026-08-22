"""Cross-encoder rerank + feature extraction for distillation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from app.retrieval.bm25 import tokenize
from app.retrieval.hybrid import Chunk


def jaccard_overlap(query: str, text: str) -> float:
    q = set(tokenize(query))
    d = set(tokenize(text))
    if not q or not d:
        return 0.0
    return len(q & d) / len(q | d)


def exact_token_hits(query: str, text: str) -> float:
    q = tokenize(query)
    if not q:
        return 0.0
    d = set(tokenize(text))
    return sum(1 for t in q if t in d) / len(q)


def path_depth(path: str) -> int:
    return len(Path(path).parts)


def filetype(path: str) -> str:
    return Path(path).suffix.lower().lstrip(".") or "none"


FEATURE_NAMES = (
    "dense_cosine",
    "bm25_score",
    "rrf_score",
    "rrf_rank",
    "jaccard",
    "exact_overlap",
    "path_depth",
    "recency",
    "is_yaml",
    "is_log",
    "is_code",
    "is_csv",
    "is_md",
)


def extract_features(
    query: str,
    chunk: Chunk,
    *,
    dense: float | None = None,
    bm25: float | None = None,
    rrf: float | None = None,
    rrf_rank: int | None = None,
    max_mtime: float = 1.0,
) -> dict[str, float]:
    ext = filetype(chunk.path)
    recency = 0.0
    if max_mtime > 0 and chunk.mtime:
        recency = chunk.mtime / max_mtime
    return {
        "dense_cosine": float(dense or 0.0),
        "bm25_score": float(bm25 or 0.0),
        "rrf_score": float(rrf or 0.0),
        "rrf_rank": float(rrf_rank or 0),
        "jaccard": jaccard_overlap(query, chunk.text),
        "exact_overlap": exact_token_hits(query, chunk.text),
        "path_depth": float(path_depth(chunk.path)),
        "recency": recency,
        "is_yaml": 1.0 if ext in {"yaml", "yml"} else 0.0,
        "is_log": 1.0 if ext in {"out", "log"} else 0.0,
        "is_code": 1.0 if ext in {"py", "js", "java", "sbatch", "sh"} else 0.0,
        "is_csv": 1.0 if ext in {"csv", "jsonl", "json"} else 0.0,
        "is_md": 1.0 if ext in {"md", "txt"} else 0.0,
    }


def feature_vector(feats: dict[str, float]) -> list[float]:
    return [float(feats[name]) for name in FEATURE_NAMES]


@dataclass
class RerankTriple:
    query: str
    chunk_id: str
    path: str
    ce_score: float
    features: dict[str, float]


class OverlapReranker:
    """Cheap lexical reranker used when no cross-encoder is loaded.

    Not a substitute for bge-reranker-v2-m3, but keeps the rerank on/off
    ablation runnable in CI and produces a teacher signal for distillation tests.
    """

    def __call__(
        self, query: str, pairs: Sequence[tuple[Chunk, float]]
    ) -> list[tuple[Chunk, float]]:
        scored = []
        for chunk, rrf_score in pairs:
            overlap = exact_token_hits(query, chunk.text) + jaccard_overlap(query, chunk.text)
            path_boost = 0.25 if Path(chunk.path).name.lower() in query.lower() else 0.0
            scored.append((chunk, float(overlap + path_boost + 0.05 * rrf_score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored


def get_cross_encoder(model_name: str = "BAAI/bge-reranker-v2-m3"):
    if not model_name or model_name.lower() == "none":
        return None
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(model_name)
    except Exception:
        return None


class CrossEncoderReranker:
    def __init__(self, model=None, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model = model if model is not None else get_cross_encoder(model_name)

    def __call__(
        self, query: str, pairs: Sequence[tuple[Chunk, float]]
    ) -> list[tuple[Chunk, float]]:
        if self.model is None:
            return OverlapReranker()(query, pairs)
        texts = [[query, chunk.text] for chunk, _ in pairs]
        scores = self.model.compute_score(texts)
        if isinstance(scores, (int, float)):
            scores = [scores]
        ranked = [(pairs[i][0], float(scores[i])) for i in range(len(pairs))]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked
