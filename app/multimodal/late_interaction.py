"""ColBERT / ColPali-style late interaction: MaxSim over token/patch embeddings.

score(q, d) = Σ_i max_j (q_i · d_j)

Storage is one vector per token or patch, not one per chunk: report it as a cost/recall trade-off, not as the default
production path. For text, M49 measures it instead of estimating it: the token counts its runs record give the index
bytes per corpus against bge-small's one vector per document (app/eval/late_interaction.py, storage_table; the README's
late-interaction block).

M49 adds, for text (app/eval/late_interaction.py):
  TokenStore               every document's scored token vectors, ragged: one flat float32 array and offsets, no padding
  maxsim_scores            maxsim's arithmetic over a whole store at once, batched and block-bounded, in float64
  MultiVectorTextEncoder   a sentence-transformers MultiVectorEncoder (ColBERT checkpoints) as the repository uses it:
                           settings read back, token counts, provenance; torch and sentence-transformers are imported
                           lazily, so this module imports with numpy alone
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.retrieval.embedders import l2_normalize


def maxsim(query_tok: np.ndarray, doc_tok: np.ndarray) -> float:
    """query_tok: (Q, D), doc_tok: (N, D). Both should be L2-normalized."""
    if query_tok.size == 0 or doc_tok.size == 0:
        return 0.0
    q = l2_normalize(np.asarray(query_tok, dtype=np.float32))
    d = l2_normalize(np.asarray(doc_tok, dtype=np.float32))
    sims = q @ d.T  # (Q, N)
    return float(sims.max(axis=1).sum())


def maxsim_search(
    query_tok: np.ndarray,
    corpus: list[tuple[str, np.ndarray]],
    k: int = 10,
) -> list[tuple[str, float]]:
    scored = [(doc_id, maxsim(query_tok, tokens)) for doc_id, tokens in corpus]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]


def storage_multiplier(seq_len: int) -> int:
    """Approximate storage vs one vector per chunk, counting vectors only: it ignores the dimensions (ColBERT keeps 96
    or 128 floats per token, bge-small 384 per chunk). The measured bytes per corpus, from real token counts and dims,
    are M49's (app.eval.late_interaction.storage_table)."""
    return max(1, int(seq_len))


# ---------------------------------------------------------------- M49: a token store and the batched pass over it


@dataclass
class TokenStore:
    """Every document's scored token vectors, in corpus order: `vectors` (total tokens, dim) float32 and `offsets`
    (n_docs + 1), document i's rows being vectors[offsets[i]:offsets[i + 1]] (a document may have none)."""

    vectors: np.ndarray
    offsets: np.ndarray

    @classmethod
    def from_matrices(cls, matrices: Sequence[np.ndarray], dim: int) -> "TokenStore":
        lengths = np.array([int(np.asarray(m).shape[0]) if np.asarray(m).size else 0 for m in matrices], dtype=np.int64)
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        vectors = np.empty((int(offsets[-1]), dim), dtype=np.float32)
        for i, m in enumerate(matrices):
            if lengths[i]:
                vectors[offsets[i]:offsets[i + 1]] = np.asarray(m, dtype=np.float32).reshape(-1, dim)
        return cls(vectors=vectors, offsets=offsets)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def doc(self, i: int) -> np.ndarray:
        return self.vectors[self.offsets[i]:self.offsets[i + 1]]

    @property
    def nbytes(self) -> int:
        return int(self.vectors.nbytes)


def _blocks(lengths: np.ndarray, cap: int) -> list[tuple[int, int, int]]:
    """Consecutive [start, end) runs whose count times longest length stays within `cap` (always at least one item):
    (start, end, longest). The run's padded size is (end - start) * longest."""
    out: list[tuple[int, int, int]] = []
    start, longest = 0, 0
    for i, n in enumerate(lengths.tolist()):
        wider = max(longest, int(n))
        if i > start and (i + 1 - start) * wider > cap:
            out.append((start, i, longest))
            start, wider = i, int(n)
        longest = wider
    if len(lengths):
        out.append((start, len(lengths), longest))
    return out


QUERY_ROWS_PER_BLOCK = 2048  # padded query rows per block; with MAXSIM_BLOCK_BYTES it sets the document block (no effect on a score)


def maxsim_scores(queries: Sequence[np.ndarray], store: TokenStore, *, block_bytes: int = 1 << 30, device: str | None = None,
                  query_rows: int = QUERY_ROWS_PER_BLOCK) -> np.ndarray:
    """(n_queries, n_docs) float64: maxsim's arithmetic for every (query, document) of the store, in one pass.

    The same score as maxsim(q, d): rows L2-normalised, every query row's maximum dot product with a real document row,
    summed over the query rows; a document or a query with no rows scores 0. Here in float64: queries and documents go
    in blocks, each document block padded on the device to its longest document and masked, each block upcast to float64
    there; the largest intermediate (padded query rows x padded document rows x 8 B) stays within `block_bytes`, so a
    full float64 copy of the store is never made. `device` None: numpy on the host; otherwise torch on that device
    ("cuda", "cpu"). The blocking changes no score beyond float64 rounding (tests/eval/test_late_interaction_scoring.py)."""
    n_q, n_d = len(queries), len(store)
    out = np.zeros((n_q, n_d), dtype=np.float64)
    if n_q == 0 or n_d == 0:
        return out
    dim = store.dim
    q_lengths = np.array([int(np.asarray(q).shape[0]) if np.asarray(q).size else 0 for q in queries], dtype=np.int64)
    q_blocks = [b for b in _blocks(q_lengths, max(1, query_rows)) if b[2] > 0]
    if not q_blocks:
        return out
    widest_q = max((e - s) * w for s, e, w in q_blocks)
    d_blocks = _blocks(store.lengths(), max(1, int(block_bytes) // (8 * max(1, widest_q))))
    if device is None:
        _maxsim_numpy(queries, q_blocks, store, d_blocks, dim, out)
    else:
        _maxsim_torch(queries, q_blocks, store, d_blocks, dim, out, device)
    return out


def _padded(mats: Sequence[np.ndarray], start: int, end: int, width: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """(end - start, width, dim) float32 and its (end - start, width) mask, from a list of ragged matrices."""
    block = np.zeros((end - start, width, dim), dtype=np.float32)
    mask = np.zeros((end - start, width), dtype=bool)
    for j, i in enumerate(range(start, end)):
        m = np.asarray(mats[i], dtype=np.float32)
        if m.size:
            block[j, : m.shape[0]] = m.reshape(-1, dim)
            mask[j, : m.shape[0]] = True
    return block, mask


def _store_block(store: TokenStore, start: int, end: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    lengths = store.lengths()[start:end]
    block = np.zeros((end - start, width, store.dim), dtype=np.float32)
    mask = np.arange(width)[None, :] < lengths[:, None]
    block[mask] = store.vectors[store.offsets[start]:store.offsets[end]]
    return block, mask


def _unit64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def _maxsim_numpy(queries, q_blocks, store, d_blocks, dim, out) -> None:
    prepared = []
    for s, e, w in q_blocks:
        qb, qm = _padded(queries, s, e, w, dim)
        prepared.append((s, e, _unit64(qb).reshape(-1, dim), qm))
    for ds, de, dw in d_blocks:
        if dw == 0:
            continue  # documents with no rows score 0
        db, dm = _store_block(store, ds, de, dw)
        d64 = _unit64(db).reshape(-1, dim)
        for s, e, q64, qm in prepared:
            sims = (q64 @ d64.T).reshape(e - s, qm.shape[1], de - ds, dw)
            sims = np.where(dm[None, None, :, :], sims, -np.inf).max(axis=3)  # (queries, query rows, documents)
            sims = np.where(np.isneginf(sims), 0.0, sims) * qm[:, :, None]
            out[s:e, ds:de] = sims.sum(axis=1)


def _maxsim_torch(queries, q_blocks, store, d_blocks, dim, out, device) -> None:
    import torch

    dev = torch.device(device)
    prepared = []
    for s, e, w in q_blocks:
        qb, qm = _padded(queries, s, e, w, dim)
        q64 = torch.nn.functional.normalize(torch.from_numpy(qb).to(dev).double(), p=2, dim=-1, eps=1e-12).reshape(-1, dim)
        prepared.append((s, e, q64, torch.from_numpy(qm).to(dev)))
    with torch.inference_mode():
        for ds, de, dw in d_blocks:
            if dw == 0:
                continue
            db, dm = _store_block(store, ds, de, dw)
            d64 = torch.nn.functional.normalize(torch.from_numpy(db).to(dev).double(), p=2, dim=-1, eps=1e-12).reshape(-1, dim)
            dmask = torch.from_numpy(dm).to(dev)
            for s, e, q64, qm in prepared:
                sims = (q64 @ d64.T).view(e - s, qm.shape[1], de - ds, dw)
                sims.masked_fill_(~dmask[None, None, :, :], float("-inf"))
                best = sims.amax(dim=3)
                best = torch.where(torch.isneginf(best), torch.zeros_like(best), best) * qm[:, :, None]
                out[s:e, ds:de] = best.sum(dim=1).cpu().numpy()
                del sims, best


# ---------------------------------------------------------------- M49: the text encoder


ENCODE_CHUNK = 4096  # texts per encode call (sentence-transformers sorts by length inside a call); fixed, so runs repeat


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class MultiVectorTextEncoder:
    """A sentence-transformers MultiVectorEncoder (6.1+) over text, loaded at a pinned revision in eval mode and fp32,
    with the model's own query and document markers, query expansion and punctuation skiplist (Stanford-NLP
    artifact.metadata or the PyLate config). encode_* return one float32 matrix of scored token vectors per text (the
    scoring mask applied: skiplisted document tokens dropped, query expansion positions kept), which the repository's
    maxsim takes as they are. `document_length` overrides the saved document length on the backbone Transformer module
    (the knob artifact.metadata's doc_maxlen seeds), read back afterwards."""

    def __init__(self, model_name: str, *, revision: str | None = None, device: str | None = None, batch_size: int = 32,
                 document_length: int | None = None, dtype: str = "float32"):
        from sentence_transformers import MultiVectorEncoder

        self.model_name, self.revision, self.batch_size, self.requested_dtype = model_name, revision, int(batch_size), dtype
        self.model = MultiVectorEncoder(model_name, revision=revision, device=device, model_kwargs={"dtype": dtype})
        self.model.eval()
        self.transformer = next(m for m in self.model if type(m).__name__ == "Transformer")
        self.saved_document_length = self.transformer.document_length
        if document_length is not None:
            self.set_document_length(document_length)

    # -- settings
    def set_document_length(self, length: int) -> None:
        self.transformer.document_length = int(length)
        if self.transformer.document_length != int(length):
            raise RuntimeError(f"{self.model_name}: document_length reads back {self.transformer.document_length}, not {length}")

    @property
    def dim(self) -> int:
        return int(self.model.get_embedding_dimension())

    def _mask_module(self):
        return next((m for m in self.model if type(m).__name__ == "MultiVectorMask"), None)

    def settings(self) -> dict[str, Any]:
        """What the loaded model will do, read back from its modules (asserted against the declared settings by the caller)."""
        t = self.transformer
        expansion = t.query_expansion
        mask = self._mask_module()
        tok = self.model.tokenizer
        prompts = {k: v for k, v in (self.model.prompts or {}).items() if k in ("query", "document")}
        markers = {}
        for side, prompt in prompts.items():
            word = (prompt or "").split()[0] if (prompt or "").split() else ""
            first = tok.tokenize((prompt or "") + "x")[0] if word else None  # the piece the marker becomes ("[Q] " for an added "[Q] ")
            markers[side] = {"prompt": prompt, "token": word, "piece": first, "id": tok.convert_tokens_to_ids(first) if first else None,
                             "single_piece": bool(word) and first is not None and first.strip() == word}
        return {"dim": self.dim,
                "query_length": int(expansion["length"]) if expansion else (int(t.query_length) if t.query_length else None),
                "query_expansion": bool(expansion), "expansion_attended": bool(expansion.get("attend")) if expansion else False,
                "expansion_strategy": expansion.get("strategy") if expansion else None,
                "document_length": int(t.document_length) if t.document_length is not None else None,
                "saved_document_length": int(self.saved_document_length) if self.saved_document_length is not None else None,
                "skiplist_words": list(mask.skiplist_words) if mask is not None else [],
                "skiplist_ids": sorted(int(i) for i in (mask._skiplist_ids.tolist() if mask is not None and mask._skiplist_ids is not None else [])),
                "skiplist_tasks": list(mask.skiplist_tasks) if mask is not None else [],
                "markers": markers, "do_lower_case": bool(getattr(t, "do_lower_case", False)),
                # a saved processing_kwargs text max_length would win over document_length in Transformer.preprocess
                "text_max_length_override": ((getattr(t, "processing_kwargs", None) or {}).get("text") or {}).get("max_length"),
                "similarity_fn_name": self.model.similarity_fn_name, "modules": [type(m).__name__ for m in self.model],
                "max_seq_length": t.max_seq_length}

    # -- encoding
    def _encode(self, texts: Sequence[str], side: str) -> list[np.ndarray]:
        fn = self.model.encode_query if side == "query" else self.model.encode_document
        out: list[np.ndarray] = []
        for start in range(0, len(texts), ENCODE_CHUNK):
            part = fn(list(texts[start:start + ENCODE_CHUNK]), batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False)
            out += [np.asarray(m, dtype=np.float32) for m in part]
        return out

    def encode_queries(self, texts: Sequence[str]) -> list[np.ndarray]:
        return self._encode(texts, "query")

    def encode_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        return self._encode(texts, "document")

    def token_counts(self, texts: Sequence[str], side: str) -> list[int]:
        """Each text's token count before any truncation: the marker prompt, the text and the special tokens, as the
        model's tokenizer splits them (a do_lower_case backbone lower-cases inside the tokenizer's normalizer, so the
        same call counts what the model reads)."""
        prompt = (self.model.prompts or {}).get(side) or ""
        tok = self.model.tokenizer
        counts: list[int] = []
        for start in range(0, len(texts), ENCODE_CHUNK):
            batch = [prompt + t for t in texts[start:start + ENCODE_CHUNK]]
            counts += [len(ids) for ids in tok(batch, add_special_tokens=True, truncation=False)["input_ids"]]
        return counts

    def similarity(self, queries: Sequence[np.ndarray], documents: Sequence[np.ndarray]) -> np.ndarray:
        """sentence-transformers' own MaxSim (util.similarity.maxsim), as float64 on the host: a cross-check only."""
        return np.asarray(self.model.similarity(list(queries), list(documents)).float().cpu().numpy(), dtype=np.float64)

    # -- provenance
    def weights_file(self) -> Path | None:
        from huggingface_hub import try_to_load_from_cache

        found = try_to_load_from_cache(self.model_name, "model.safetensors", revision=self.revision)
        return Path(found) if isinstance(found, str) else None

    def weights_check(self) -> dict[str, Any]:
        """Every tensor of the checkpoint's weight files against the loaded modules: the backbone's model.safetensors
        (Stanford-NLP keys carry a `bert.` prefix, and `linear.weight` is the projection) and a PyLate save's Dense
        files. Missing, unexpected and unequal tensors are listed; a correct load has none (TRIPWIRE checklist 6)."""
        import torch
        from huggingface_hub import try_to_load_from_cache
        from safetensors import safe_open

        # parameters and every buffer (a non-persistent one such as position_ids is not in state_dict())
        backbone = {**dict(self.transformer.model.named_parameters()), **dict(self.transformer.model.named_buffers())}
        dense = [m for m in self.model if type(m).__name__ == "Dense"]
        report: dict[str, Any] = {"files": {}, "missing_in_model": [], "unequal": [], "linear_weight_loaded": None}
        path = self.weights_file()
        if path is None:
            return {**report, "error": "model.safetensors is not in the cache"}
        matched = 0
        with safe_open(str(path), framework="pt") as fh:
            keys = list(fh.keys())
            for key in keys:
                tensor = fh.get_tensor(key)
                if key == "linear.weight":
                    ok = bool(dense) and torch.equal(dense[0].linear.weight.detach().cpu().to(tensor.dtype), tensor)
                    report["linear_weight_loaded"] = ok
                    matched += ok
                    continue
                name = next((n for n in (key, key.split(".", 1)[-1]) if n in backbone), None)
                if name is None:
                    report["missing_in_model"].append(key)
                    continue
                if not torch.equal(backbone[name].detach().cpu().to(tensor.dtype), tensor):
                    report["unequal"].append(key)
                matched += 1
        report["files"]["model.safetensors"] = {"keys": len(keys), "matched": matched}
        for i, module in enumerate(dense):
            sub = f"{i + 1}_Dense/model.safetensors"
            found = try_to_load_from_cache(self.model_name, sub, revision=self.revision)
            if isinstance(found, str):
                with safe_open(found, framework="pt") as fh:
                    for key in fh.keys():
                        ok = torch.equal(module.linear.weight.detach().cpu().to(fh.get_tensor(key).dtype), fh.get_tensor(key))
                        report["files"][sub] = {"key": key, "equal": ok}
        report["ok"] = not report["missing_in_model"] and not report["unequal"] and report["linear_weight_loaded"] is not False
        return report

    def provenance(self) -> dict[str, Any]:
        import sentence_transformers
        import torch
        import transformers

        from app.retrieval.embedders import weights_dtype

        path = self.weights_file()
        params = list(self.model.parameters())
        return {"model": self.model_name, "revision": self.revision, "weights_sha256": _sha256(path) if path else None,
                "weights_bytes": path.stat().st_size if path else None, "dtype": weights_dtype(self.model),
                "all_parameters_dtype": sorted({str(p.dtype) for p in params}), "requested_dtype": self.requested_dtype,
                "parameters": int(sum(p.numel() for p in params)), "device": str(self.model.device), "batch_size": self.batch_size,
                "versions": {"sentence_transformers": sentence_transformers.__version__, "transformers": transformers.__version__,
                             "torch": torch.__version__},
                "settings": self.settings(), "training": self.model.training}
