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
    teacher: str | None = None  # what ce_score is, when the caller records it (the bench log does)


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


SCORE_SCALE = "sigmoid probability in [0, 1]"


def get_cross_encoder(
    model_name: str = "BAAI/bge-reranker-v2-m3",
    precision: str = "fp32",
    max_length: int | None = None,
    device: str | None = None,
    *,
    raise_errors: bool = False,
):
    """Load a sentence-transformers CrossEncoder, or None when it cannot be loaded.

    precision is the dtype the weights load in: "fp16" torch.float16 and "bf16" torch.bfloat16 on
    CUDA (bf16 is for the large causal-LM rerankers, Qwen3-Reranker, whose weights are bfloat16);
    "fp32", and either half precision on a CPU, torch.float32. The float32 is named, not left to the
    library: transformers >= 5 otherwise loads a checkpoint in the dtype it was saved in, bfloat16 for
    the causal-LM rerankers (the bge-reranker-v2-m3 checkpoint is float32 either way). max_length caps
    the tokenized (query, passage) pair. Both are recorded next to every result that used them.

    raise_errors: raise instead of returning None (weights not cached, a load that fails), so a caller
    that requires the model (a benchmark row) stops with the reason.
    """
    if not model_name or model_name.lower() == "none":
        return None
    import os
    from pathlib import Path

    allow = os.environ.get("BGE_ALLOW_DOWNLOAD", "").strip().lower() in {"1", "true", "yes"}
    local = Path(model_name).exists()
    if not allow and not local and not _hf_cache_has(model_name):
        if raise_errors:
            raise FileNotFoundError(f"{model_name} is not in the Hugging Face cache (HF_HOME) and BGE_ALLOW_DOWNLOAD is not set")
        return None
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        if raise_errors:
            raise
        return None
    kwargs: dict = {}
    if max_length is not None:
        kwargs["max_length"] = int(max_length)
    if device:
        kwargs["device"] = device
    try:
        import torch

        half = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
        on_cuda = device != "cpu" and torch.cuda.is_available()
        kwargs["model_kwargs"] = {"torch_dtype": half if half is not None and on_cuda else torch.float32}
    except Exception:
        pass
    try:
        if allow or local:
            return CrossEncoder(model_name, **kwargs)
        try:
            return CrossEncoder(model_name, local_files_only=True, **kwargs)
        except TypeError:
            return CrossEncoder(model_name, **kwargs)
    except Exception:
        if raise_errors:
            raise
        return None


def cross_encoder_provenance(model) -> dict:
    """What a results row records about a loaded CrossEncoder beyond its name and revision: the dtype its weights loaded
    in, its token cap, and the prompt it puts in front of every pair by default (Qwen3-Reranker's instruction; None for
    the bge and mxbai rerankers)."""
    from app.retrieval.embedders import weights_dtype

    name = getattr(model, "default_prompt_name", None)
    max_length = getattr(model, "max_seq_length", None)
    if max_length is None:
        max_length = getattr(model, "max_length", None)
    return {"dtype": weights_dtype(model), "loaded_max_length": max_length, "default_prompt_name": name,
            "default_prompt": (getattr(model, "prompts", None) or {}).get(name) if name else None}


def _hf_cache_has(model_name: str) -> bool:
    """Filesystem-only check. Do not call huggingface_hub (it can hit the network)."""
    import os
    from pathlib import Path

    slug = "models--" + model_name.replace("/", "--")
    roots = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "huggingface",
    ]
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.insert(0, Path(hf_home) / "hub")
        roots.insert(0, Path(hf_home))
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        roots.insert(0, Path(hub_cache))
    for root in roots:
        try:
            if (root / slug).exists():
                return True
        except Exception:
            continue
    return False


def _sigmoid(x: float) -> float:
    import math

    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _identity(x):
    return x


def cross_encoder_scores(model, pairs: Sequence[Sequence[str]], batch_size: int = 32) -> list[float]:
    """Score (query, text) pairs with either cross-encoder API, on one scale (SCORE_SCALE).

    sentence-transformers' CrossEncoder.predict() applies the model's default
    activation (Sigmoid for one-label rerankers, so probabilities); FlagEmbedding's
    FlagReranker.compute_score() returns raw logits unless normalize=True. Raw
    logits are requested from both where the API allows it and passed through a
    sigmoid here, so a score is always the sigmoid probability in [0, 1] whatever
    library loaded the model. Ranking is unchanged by the monotone map.
    """
    if not pairs:
        return []
    batch = [list(p) for p in pairs]
    if hasattr(model, "predict"):
        raw = None
        for kw in ("activation_fn", "activation_fct"):  # sentence-transformers >= 5 / < 5
            try:
                raw = model.predict(batch, batch_size=batch_size, show_progress_bar=False, **{kw: _identity})
                break
            except TypeError:
                continue
        if raw is None:  # no activation override: assume the library returned probabilities
            probs = model.predict(batch, batch_size=batch_size, show_progress_bar=False)
            return [float(x) for x in (probs if not isinstance(probs, (int, float)) else [probs])]
        scores = [_sigmoid(float(x)) for x in (raw if not isinstance(raw, (int, float)) else [raw])]
        return scores
    if hasattr(model, "compute_score"):
        try:
            probs = model.compute_score(batch, normalize=True)
            return [float(x) for x in (probs if not isinstance(probs, (int, float)) else [probs])]
        except TypeError:
            raw = model.compute_score(batch)
            return [_sigmoid(float(x)) for x in (raw if not isinstance(raw, (int, float)) else [raw])]
    raise TypeError(f"{type(model).__name__} has neither predict() nor compute_score()")


def cross_encoder_logits(model, pairs: Sequence[Sequence[str]], batch_size: int = 32) -> list[float]:
    """Raw logits of (query, text) pairs from a sentence-transformers CrossEncoder: what cross_encoder_scores puts
    through the sigmoid. Distillation (M43) trains a student on the teacher's logits, so a library that cannot hand
    them back without its activation is an error here, never a silent probability."""
    if not pairs:
        return []
    batch = [list(p) for p in pairs]
    for kw in ("activation_fn", "activation_fct"):  # sentence-transformers >= 5 / < 5
        try:
            raw = model.predict(batch, batch_size=batch_size, show_progress_bar=False, **{kw: _identity})
        except TypeError:
            continue
        return [float(x) for x in (raw if not isinstance(raw, (int, float)) else [raw])]
    raise TypeError(f"{type(model).__name__}.predict() takes no activation override; its raw logits cannot be read")


class CrossEncoderReranker:
    """A cross-encoder as a rerank_fn (scores on SCORE_SCALE).

    When no model loads (weights not cached, no sentence-transformers) calling it
    raises, unless the caller asked for the overlap fallback with
    allow_fallback=True: then `fallback` is "overlap" and the scores are
    OverlapReranker's. A caller that allows it must label its results:
    build_index records reranker_fallback, bge_reranker returns loaded=False.
    """

    def __init__(self, model=None, model_name: str = "BAAI/bge-reranker-v2-m3", *, allow_fallback: bool = False,
                 precision: str = "fp32", max_length: int | None = None):
        """precision / max_length: as get_cross_encoder loads the weights (the served RERANKER_PRECISION /
        RERANKER_MAX_LENGTH); fp32 and the model's own token cap by default."""
        self.model = model if model is not None else get_cross_encoder(model_name, precision=precision, max_length=max_length)
        self.model_name = model_name
        self.precision = precision
        self.max_length = max_length
        self.allow_fallback = allow_fallback
        self.fallback = "overlap" if self.model is None and allow_fallback else None

    def __call__(
        self, query: str, pairs: Sequence[tuple[Chunk, float]]
    ) -> list[tuple[Chunk, float]]:
        if self.model is None:
            if not self.allow_fallback:
                raise RuntimeError(
                    f"cross-encoder {self.model_name!r} is not loaded; construct CrossEncoderReranker(allow_fallback=True) "
                    "to rerank with token overlap instead, and label the results as such"
                )
            return OverlapReranker()(query, pairs)
        scores = cross_encoder_scores(self.model, [(query, chunk.text) for chunk, _ in pairs])
        ranked = [(pairs[i][0], scores[i]) for i in range(len(pairs))]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked
