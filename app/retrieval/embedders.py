"""Deterministic embedders for the local bench.

hash     — character n-gram hashing trick (no downloads, CI-safe)
tfidf    — sklearn TF-IDF + TruncatedSVD dense proxy
st       — sentence-transformers (real numbers; optional)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def encode_queries(embedder, texts: Sequence[str]) -> np.ndarray:
    """Queries through the embedder's query side when it has one (SentenceTransformerEmbedder.encode_queries: the
    model's own query prompt); otherwise exactly encode(), as every embedder without a query prompt always did."""
    side = getattr(embedder, "encode_queries", None)
    return side(texts) if side is not None else embedder.encode(texts)


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return mat / norms


class HashEmbedder:
    """Signed hashing trick over character 3- and 4-grams. Deterministic, no deps besides numpy."""

    name = "hash"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            blob = (text or "").lower()
            for n in (3, 4):
                if len(blob) < n:
                    continue
                for j in range(len(blob) - n + 1):
                    gram = blob[j : j + n]
                    digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                    idx = int.from_bytes(digest[:4], "little") % self.dim
                    sign = 1.0 if digest[4] & 1 else -1.0
                    out[i, idx] += sign
        return l2_normalize(out)


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self, dim: int = 128):
        self.dim = dim
        self._vectorizer = None
        self._svd = None
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> "TfidfEmbedder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        n = max(len(texts), 1)
        ngram = (1, 2)
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=ngram,
            min_df=1,
            max_features=4096,
            token_pattern=r"[A-Za-z0-9_./-]+",
        )
        sparse = self._vectorizer.fit_transform(texts)
        n_comp = min(self.dim, max(1, min(sparse.shape) - 1), n - 1)
        if n_comp < 1:
            self._svd = None
            self.dim = sparse.shape[1] or 1
            self._fitted = True
            return self
        self._svd = TruncatedSVD(n_components=n_comp, random_state=0)
        self._svd.fit(sparse)
        self.dim = n_comp
        self._fitted = True
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted or self._vectorizer is None:
            raise RuntimeError("TfidfEmbedder.fit() must be called before encode()")
        sparse = self._vectorizer.transform(texts)
        if self._svd is None:
            dense = sparse.toarray().astype(np.float32)
        else:
            dense = self._svd.transform(sparse).astype(np.float32)
        return l2_normalize(dense)


# Query-side prompts by model (M41): the name of a prompt in the model's own config_sentence_transformers.json, which
# sentence-transformers puts before each query (encode(prompt_name=...)); documents get none. These models were
# trained with an instruction on the query side and score worse without it. A model not listed here gets no prompt,
# exactly as every committed row did (bge-small, bge-base), so those rows reproduce unchanged.
QUERY_PROMPT_NAMES = {
    "Qwen/Qwen3-Embedding-0.6B": "query",
    "Qwen/Qwen3-Embedding-8B": "query",
    "microsoft/harrier-oss-v1-27b": "web_search_query",
}
# The dtype each of these models loads in (M41), passed as model_kwargs={"dtype": ...}. transformers >= 5 loads a
# checkpoint in the dtype it was saved in when none is named (its from_pretrained default is "auto"): float32 for the
# bge checkpoints, bfloat16 for Qwen3-Embedding and harrier. So a dtype is named for every new model: "auto" (bfloat16)
# for the large ones, which need it (harrier-oss-v1-27b is about 108 GB in float32 and 54 GB in bfloat16), "float32"
# for the one small enough to serve on a CPU, which is how a CPU would serve it. A model not listed gets no
# model_kwargs, exactly as every committed row did (bge-small, bge-base: float32 either way). The dtype the weights
# have once loaded is read back and recorded with every row (provenance()).
MODEL_DTYPES = {
    "Qwen/Qwen3-Embedding-0.6B": "float32",
    "Qwen/Qwen3-Embedding-8B": "auto",
    "microsoft/harrier-oss-v1-27b": "auto",
}
# Texts per encode() batch for the large models (sentence-transformers' default is 32), so the longest batch of a BEIR
# corpus (batches are sorted by length) fits beside the weights; recorded with every row. A model not listed gets the
# library default, as every committed row did.
ENCODE_BATCH_SIZES = {"Qwen/Qwen3-Embedding-8B": 16, "microsoft/harrier-oss-v1-27b": 8}


# Models whose published sentence-transformers files sentence-transformers (6.1) cannot load as they are, and the one key
# that stops them: (file, key, value). harrier-oss-v1-27b's tokenizer_config.json names processor_class Gemma3Processor,
# the image-and-text processor of the multimodal Gemma 3, but the checkpoint is text only (Gemma3TextModel) and ships no
# image processor, so the library's AutoProcessor fails ("Can't load image processor"). Such a model loads from a
# temporary view of its cached snapshot in which every file and folder is a symlink to the snapshot's own except that
# file, written without that key: AutoProcessor then falls back to the model's own tokenizer (gemma3_text has no
# processor class), which still begins a text with <bos> and ends it with <eos>, the position last-token pooling reads.
# The cache itself is not edited, and every row records what the view dropped (provenance()["snapshot_view"]).
SNAPSHOT_VIEWS = {"microsoft/harrier-oss-v1-27b": ("tokenizer_config.json", "processor_class", "Gemma3Processor")}


def _snapshot_path(model_name: str) -> Path:
    """The cached snapshot of `model_name` (huggingface_hub honours HF_HUB_OFFLINE: the cache, no network)."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_name))


def snapshot_view(model_name: str):
    """(a TemporaryDirectory holding the view of SNAPSHOT_VIEWS[model_name], what the view changed). Raises when the
    snapshot's file no longer holds that key with that value: the model's files changed upstream, so the view must be
    looked at again rather than applied blindly."""
    import json
    import tempfile

    file, key, value = SNAPSHOT_VIEWS[model_name]
    snapshot = _snapshot_path(model_name)
    config = json.loads((snapshot / file).read_text())
    if config.get(key) != value:
        raise RuntimeError(f"{model_name}: {file} has {key}={config.get(key)!r}, not {value!r}; the snapshot view no longer applies")
    tmp = tempfile.TemporaryDirectory(prefix="st-snapshot-view-")
    view = Path(tmp.name)
    for entry in snapshot.iterdir():
        if entry.name != file:
            (view / entry.name).symlink_to(entry)
    del config[key]
    (view / file).write_text(json.dumps(config, indent=2))
    return tmp, {"loaded_from": f"a temporary view of the cached snapshot: every file a symlink to the snapshot's own, except {file}",
                 "file": file, "dropped": {key: value}}


def weights_dtype(model) -> str | None:
    """The dtype a loaded torch model's weights are in ("torch.bfloat16"), read from its first parameter; None when
    there is none to read."""
    try:
        return str(next(iter(model.parameters())).dtype)
    except Exception:
        return None


class SentenceTransformerEmbedder:
    name = "st"

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", revision: str | None = None):
        """revision (M48): when given, the cached snapshot must be that commit (refs/main, app.eval.provenance.hf_revision)
        and it is the one loaded; any other cached revision raises. None loads by hub id, as every committed row did."""
        if revision is not None:  # before the library is imported: a wrong snapshot is refused on any machine
            from app.eval.provenance import hf_revision

            found = hf_revision(model_name)
            if found != revision:
                raise RuntimeError(f"{model_name}: the cached revision is {found}, the pinned one is {revision}")
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.revision = revision
        self.dtype = MODEL_DTYPES.get(model_name)  # None: no model_kwargs, the library's default
        kwargs = {"model_kwargs": {"dtype": self.dtype}} if self.dtype else {}
        if revision is not None:
            kwargs["revision"] = revision
        self.snapshot_view = None  # what a SNAPSHOT_VIEWS model's view dropped; None: loaded by its hub id, as committed
        source = model_name
        if model_name in SNAPSHOT_VIEWS:
            self._view, self.snapshot_view = snapshot_view(model_name)  # the TemporaryDirectory lives as long as the embedder
            source = self._view.name
        self._model = SentenceTransformer(source, **kwargs)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"st:{model_name}"
        self.batch_size = ENCODE_BATCH_SIZES.get(model_name)
        self.query_prompt_name = QUERY_PROMPT_NAMES.get(model_name)
        self.query_prompt = None
        if self.query_prompt_name is not None:
            prompts = getattr(self._model, "prompts", None) or {}
            if self.query_prompt_name not in prompts:  # a renamed prompt must not silently become no prompt
                raise ValueError(f"{model_name} has no prompt named {self.query_prompt_name!r} (has {sorted(prompts)})")
            self.query_prompt = prompts[self.query_prompt_name]

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Queries with the model's query prompt (QUERY_PROMPT_NAMES); without one, exactly encode()."""
        if self.query_prompt_name is None:
            return self.encode(texts)
        return self._encode(texts, prompt_name=self.query_prompt_name)

    def provenance(self) -> dict:
        """What a results row records about how this embedder encodes: the query prompt (name and text), the dtype asked
        for (None: the library's default) and the one the weights loaded in, the encode batch size (None: the library's
        default), the token cap, and what a snapshot view changed (SNAPSHOT_VIEWS; None for every other model)."""
        return {"query_prompt_name": self.query_prompt_name, "query_prompt": self.query_prompt, "dtype_requested": self.dtype,
                "dtype": weights_dtype(self._model), "batch_size": self.batch_size,
                "max_seq_length": getattr(self._model, "max_seq_length", None), "snapshot_view": self.snapshot_view}

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def _encode(self, texts: Sequence[str], **extra) -> np.ndarray:
        if self.batch_size is not None:
            extra["batch_size"] = self.batch_size
        vecs = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            **extra,
        )
        return np.asarray(vecs, dtype=np.float32)

    def token_embeddings(self, text: str, max_tokens: int = 510) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Token-level embeddings of a whole document plus each token's character span
        (for late chunking). Longer documents are encoded in consecutive windows of
        `max_tokens`, so a token's context is its window, not always the full file."""
        import torch

        tokenizer = self._model.tokenizer
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False, verbose=False)
        ids, offsets = list(enc["input_ids"]), [(int(a), int(b)) for a, b in enc["offset_mapping"]]
        transformer = self._model[0].auto_model
        device = next(transformer.parameters()).device
        parts = []
        for i in range(0, len(ids), max_tokens):
            window = [tokenizer.cls_token_id, *ids[i : i + max_tokens], tokenizer.sep_token_id]
            with torch.no_grad():
                hidden = transformer(
                    input_ids=torch.tensor([window], device=device),
                    attention_mask=torch.ones((1, len(window)), dtype=torch.long, device=device),
                ).last_hidden_state[0, 1:-1]
            parts.append(hidden.float().cpu().numpy())
        mat = np.concatenate(parts) if parts else np.zeros((0, self.dim), dtype=np.float32)
        return mat.astype(np.float32), offsets


def cosine_scores(query_vec: np.ndarray, doc_mat: np.ndarray) -> np.ndarray:
    q = query_vec.reshape(1, -1)
    q = l2_normalize(q)
    docs = l2_normalize(doc_mat)
    return (docs @ q.T).ravel()
