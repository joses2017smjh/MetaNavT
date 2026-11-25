"""Train an adapted embedder (M48), by the recipe declared in app/eval/adapted_embedders.py.

    python -m app.eval.embedder_train scale --data-dir <BEIR dir>                 # -> bench/distill/embedder_scale.json
    python -m app.eval.embedder_train train --arm bge-small-marginmse --data-dir <BEIR dir> --checkpoint-dir <dir> \
                                            --resume-dir <dir> [--micro-lists n] [--time-limit-min 225]
    python -m app.eval.embedder_train manifest --checkpoint-dir <dir>             # -> bench/distill/embedders.json

An arm is bge-small (or bge-base) fine-tuned without a label: its targets are the teacher's logits for the candidates
of M43's synthetic queries (bench/distill/teacher_scores.jsonl.gz), its student score for a (query, document) pair is
s * cos(E(q), E(d)) with E the SentenceTransformer forward on the model's own preprocess() output (the path encode()
takes when the model serves), and its checkpoint is the epoch end with the lowest loss on the synthetic queries of the
held-out source documents. s is fixed before training by a label-free rule (`scale`). Nothing here opens a gold
question, a BEIR query or a qrel: only the two committed files, their metas and the texts of the candidates (the
fixture's chunks, each BEIR corpus.jsonl).

Every setting comes from RECIPE; this module has no hyper-parameter of its own. MarginMSE is M43's definition
(app.eval.distill_train.margin_mse); the exploratory MNRL arm's loss is written here, with the doc-id mask the library's
loss has no hook for.

A run saves app state to <resume-dir>/<arm>/resume.pt (atomically) at every epoch end and, when it stops early, after
the step it is on: at --time-limit-min, or on SIGUSR1 (the batch script forwards Slurm's warning signal). The same
command then resumes from the file. Each job that worked on a run is a "segment" in training.json (job, device,
micro_lists, TF32 flags, first and last step).

What is written: the checkpoint (sentence-transformers format, float32 safetensors) under --checkpoint-dir/<arm>/
with training.json and SHA256SUMS beside it; `manifest` copies each training.json, without the checkpoint, into the
committed bench/distill/embedders.json, and a later load is checked against the reference embeddings it records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from app.eval.adapted_embedders import ARMS, ATTEMPT, ENCODING, INITS, MANIFEST_FILE, RECIPE, SCALE_FILE, TRAINING_SETS
from app.eval.distill_queries import QUERIES_FILE, read_jsonl
from app.eval.distill_teacher import SCORES_FILE, corpus_texts
from app.eval.distill_train import MixedSampler, build_lists, choose_epoch, lr_factor, macro_validation, margin_mse, micro_chunks

TRAINING_FILE = "training.json"
RESUME_FILE = "resume.pt"
WEIGHTS_FILE = "model.safetensors"
SUMS_FILE = "SHA256SUMS"
STOPPED = 3  # exit status of a run that saved its state and stopped early (time limit, USR1): run the same command again
DEFAULT_TIME_LIMIT_MIN = 225.0  # under a 4 h Slurm limit (-t 4:00): the job saves and exits on its own
FORWARD_TEXTS = 64  # texts per forward pass inside a step, sorted by length (less padding); a speed setting, not a hyper-parameter
VALIDATION_TEXTS = 128  # texts per no-gradient forward pass
TOKENIZE_BLOCK = 512  # texts per preprocess() call when every unique text is tokenized once
REFERENCE_TOLERANCE = 1e-4  # a reload's embeddings of the reference texts may differ from the trained ones by this much
REFERENCE_QUERIES, REFERENCE_DOCUMENTS = 2, 2  # fixture texts whose embeddings the manifest records (the fixture is in the repository)
SOURCES = ("app/eval/embedder_train.py", "app/eval/distill_train.py")  # the trainer's code: sha256 per run
LEAK_NEEDLES = ("/nfs/", "/scratch/", "/home/", "/tmp/", "/users/")  # an absolute path in a saved checkpoint file


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_sha256(root: Path | None = None) -> dict[str, str]:
    """The sha256 of the trainer's two files and of the declarations it reads (adapted_embedders.declarations_sha256:
    RECIPE, ARMS, INITS, ENCODING, TRAINING_SETS as canonical JSON): a later arm must be trained by the same."""
    from app.eval.adapted_embedders import declarations_sha256

    root = root or _root()
    return {**{rel: sha256_of(root / rel) for rel in SOURCES}, "declarations": declarations_sha256()}


# ---------------------------------------------------------------- data


def verify_data(queries_path: Path, scores_path: Path) -> dict[str, str]:
    """{file: sha256} of the two training files, after checking each against the sha256 its committed meta records (the
    meta beside it: queries.meta.json, teacher_scores.meta.json). Raises when one differs: they are M43's files, used
    as committed and never regenerated."""
    out = {}
    for path, meta_name in ((queries_path, "queries.meta.json"), (scores_path, "teacher_scores.meta.json")):
        meta = json.loads((Path(path).parent / meta_name).read_text())
        expected = meta["files"][Path(path).name]["sha256"] if "files" in meta else meta["file"]["sha256"]
        found = sha256_of(path)
        if found != expected:
            raise SystemExit(f"{Path(path).name} has sha256 {found}, its meta records {expected}: not M43's committed file")
        out[Path(path).name] = found
    return out


def load_lists(root: Path, data_dir: Path, queries_path: Path, scores_path: Path) -> tuple[list[dict], dict[str, dict[str, list[dict]]]]:
    """(the synthetic queries, {split: {corpus: [list]}}) with each list's source document id; every training id is a
    query of queries.jsonl.gz with split train."""
    queries, scores = read_jsonl(queries_path), read_jsonl(scores_path)
    texts = {c: corpus_texts(c, root, data_dir) for c in sorted({q["corpus"] for q in queries})}
    lists = build_lists(queries, scores, texts)
    by_id = {q["id"]: q for q in queries}
    for split, corpora in lists.items():
        for corpus, rows in corpora.items():
            for r in rows:
                q = by_id[r["id"]]
                if q["split"] != split or q["corpus"] != corpus:
                    raise ValueError(f"{r['id']}: in the {split} lists of {corpus}, but its query says {q['split']} / {q['corpus']}")
                r["source"], r["corpus"] = q["doc_id"], corpus
                if r["source"] not in r["candidates"]:
                    raise ValueError(f"{r['id']}: the source document is not in its list")
    return queries, lists


def training_set_lists(lists: dict[str, dict[str, list[dict]]], training_set: str) -> dict[str, dict[str, list[dict]]]:
    """The lists an arm reads: its training set's corpora only, training and validation."""
    corpora = TRAINING_SETS[training_set]
    return {split: {c: rows for c, rows in by.items() if c in corpora} for split, by in lists.items()}


# ---------------------------------------------------------------- the scale s (label-free, before training)


def pair_margins(values: Sequence[float]) -> np.ndarray:
    """v_i - v_j for every pair i < j of one list, in float64, in (i, j) order."""
    v = np.asarray(values, dtype=np.float64)
    i, j = np.triu_indices(v.shape[0], k=1)
    return v[i] - v[j]


def scale_from_lists(teacher: Sequence[Sequence[float]], cosine: Sequence[Sequence[float]]) -> dict[str, Any]:
    """RECIPE["scale"]: s = std(t_i - t_j) / std(c_i - c_j), numpy std (ddof 0, float64), pooled over every pair i < j
    of every list; `teacher` and `cosine` give one list each, candidate for candidate."""
    if len(teacher) != len(cosine):
        raise ValueError(f"{len(teacher)} teacher lists, {len(cosine)} cosine lists")
    t = np.concatenate([pair_margins(x) for x in teacher])
    c = np.concatenate([pair_margins(x) for x in cosine])
    if t.shape != c.shape:
        raise ValueError("teacher and cosine lists differ in length")
    st, sc = float(np.std(t, ddof=0)), float(np.std(c, ddof=0))
    if not sc > 0:
        raise ValueError("the init model's cosine margins have no spread")
    s = st / sc
    return {"s": s, "temperature": 1.0 / s, "std_teacher_margins": st, "std_cosine_margins": sc, "pairs": int(t.shape[0]), "lists": len(teacher)}


# ---------------------------------------------------------------- the model


def tf32_flags() -> dict[str, Any]:
    import torch

    return {"cuda.matmul.allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32), "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32)}


def set_fp32() -> dict[str, Any]:
    """RECIPE["precision"]: no TF32 in matmuls, float32 matmul precision 'highest'; returns the flags as set."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return tf32_flags()


def load_init(init: str, device: str | None = None):
    """An arm's starting model at its pinned revision (a cached snapshot of another revision is refused), in float32."""
    from app.eval.provenance import hf_revision

    spec = INITS[init]
    found = hf_revision(spec["model"])
    if found != spec["revision"]:
        raise RuntimeError(f"{spec['model']}: the cached revision is {found}, the pinned one is {spec['revision']}")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(spec["model"], revision=spec["revision"], device=device)
    check_encoding(model)
    return model


def check_encoding(model) -> dict[str, Any]:
    """The loaded model encodes as ENCODING declares (CLS pooling, Normalize, 512 tokens, lower case, float32)."""
    from app.retrieval.embedders import weights_dtype

    kinds = [type(m).__name__ for m in model]
    pooling = model[1].get_config_dict().get("pooling_mode") if len(model) > 1 else None
    found = {"modules": kinds, "pooling": pooling, "max_seq_length": model.max_seq_length,
             "do_lower_case": bool(getattr(model[0], "do_lower_case", False)), "dtype": weights_dtype(model)}
    expected = {"modules": ["Transformer", "Pooling", "Normalize"], "pooling": ENCODING["pooling"], "max_seq_length": ENCODING["max_seq_length"],
                "do_lower_case": ENCODING["do_lower_case"], "dtype": "torch.float32"}
    if found != expected:
        raise RuntimeError(f"the model does not encode as declared: {found} != {expected}")
    return found


class Tokens:
    """Every unique text tokenized once by the model's own preprocess() (lower case, cut at max_seq_length), stored
    without padding; rows of it are padded again per forward pass. Equal to preprocess() on the batch itself."""

    def __init__(self, model, texts: Sequence[str], block: int = TOKENIZE_BLOCK):
        self.index: dict[str, int] = {}
        self.rows: list[dict[str, np.ndarray]] = []
        self.pad = {"input_ids": int(model.tokenizer.pad_token_id or 0)}
        self.extra: dict[str, Any] = {}
        unique = list(dict.fromkeys(texts))
        for start in range(0, len(unique), block):
            part = unique[start: start + block]
            features = model.preprocess(part)
            mask = features["attention_mask"]
            for i, text in enumerate(part):
                n = int(mask[i].sum())
                self.index[text] = len(self.rows)
                self.rows.append({k: v[i, :n].numpy().astype(np.int64) for k, v in features.items() if hasattr(v, "shape") and v.ndim == 2})
            self.extra.update({k: v for k, v in features.items() if not hasattr(v, "shape")})

    def __len__(self) -> int:
        return len(self.rows)

    def length(self, text: str) -> int:
        return int(self.rows[self.index[text]]["attention_mask"].shape[0])

    def features(self, texts: Sequence[str]) -> dict[str, Any]:
        """preprocess() output for `texts`, rebuilt from the stored rows (padded to the longest)."""
        import torch

        rows = [self.rows[self.index[t]] for t in texts]
        width = max(r["attention_mask"].shape[0] for r in rows)
        out: dict[str, Any] = {}
        for key in rows[0]:
            pad = self.pad.get(key, 0)
            mat = np.full((len(rows), width), pad, dtype=np.int64)
            for i, r in enumerate(rows):
                mat[i, : r[key].shape[0]] = r[key]
            out[key] = torch.from_numpy(mat)
        out.update(self.extra)
        return out


def embed(model, tokens: Tokens, texts: Sequence[str], block: int = FORWARD_TEXTS):
    """Unit embeddings of `texts` (in that order) through the model's forward, with gradients if enabled: sorted by
    token length into forward passes of `block` texts, so a pass pads to its own longest text."""
    import torch
    from sentence_transformers.util import batch_to_device

    order = sorted(range(len(texts)), key=lambda i: tokens.length(texts[i]))
    parts, at = [], []
    for start in range(0, len(order), block):
        idx = order[start: start + block]
        features = batch_to_device(tokens.features([texts[i] for i in idx]), model.device)
        parts.append(model(features)["sentence_embedding"])
        at += idx
    emb = torch.cat(parts, dim=0)
    emb = emb[torch.as_tensor(np.argsort(at), device=emb.device)]
    return torch.nn.functional.normalize(emb.float(), p=2, dim=1)


def list_texts(lists: Sequence[dict]) -> list[str]:
    return [t for r in lists for t in (r["query"], *r["texts"])]


def encode_lists(model, tokens: Tokens, lists: Sequence[dict], block: int = VALIDATION_TEXTS) -> dict[str, np.ndarray]:
    """{text: unit embedding} of every query and candidate of `lists`, without gradients, each unique text once."""
    import torch

    unique = list(dict.fromkeys(list_texts(lists)))
    model.eval()
    out = {}
    with torch.no_grad():
        for start in range(0, len(unique), block * 8):
            part = unique[start: start + block * 8]
            emb = embed(model, tokens, part, block).cpu().numpy().astype(np.float64)
            out.update(zip(part, emb))
    return out


def list_cosines(vectors: dict[str, np.ndarray], r: dict) -> np.ndarray:
    q = vectors[r["query"]]
    return np.asarray([float(q @ vectors[t]) for t in r["texts"]], dtype=np.float64)


def validate_marginmse(model, tokens: Tokens, lists: dict[str, list[dict]], s: float) -> dict[str, Any]:
    """Per corpus the mean MarginMSE of s * cos against the teacher over the validation lists, the macro mean (the
    selection loss), and the diagnostic: the source document's mean 1-based rank inside its list by the student score."""
    vectors = encode_lists(model, tokens, [r for rows in lists.values() for r in rows])
    loss, rank = {}, {}
    for corpus, rows in lists.items():
        losses, ranks = [], []
        for r in rows:
            student = s * list_cosines(vectors, r)
            losses.append(margin_mse(student, r["scores"]))
            src = r["candidates"].index(r["source"])
            ranks.append(1 + int(np.sum(student > student[src])))
        loss[corpus], rank[corpus] = float(np.mean(losses)), float(np.mean(ranks))
    return {"loss": loss, "macro": macro_validation(loss), "source_mean_rank": rank}


# ---------------------------------------------------------------- MNRL (the exploratory arm; the loss not chosen)


def hard_negatives(scores: Sequence[float], k: int | None = None) -> list[int]:
    """RECIPE["mnrl"]: the teacher's bottom half of a list (the 10 lowest-scored of 20 or 21 candidates; numpy's stable
    sort, so an exact tie keeps the earlier candidate), then the first k of those in the list's order."""
    k = RECIPE["mnrl"]["hard_negatives"] if k is None else k
    n = len(scores)
    bottom = np.argsort(np.asarray(scores, dtype=np.float64), kind="stable")[: n // 2]
    return sorted(int(i) for i in bottom)[:k]


def mnrl_examples(lists: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], dict[str, dict[str, int]]]:
    """{corpus: [example]} for the lists where the teacher ranks the source first (positive = the source, the hard
    negatives of hard_negatives()), and per corpus how many lists were eligible and how many were not."""
    out, counts = {}, {}
    for corpus, rows in lists.items():
        kept = []
        for r in rows:
            src = r["candidates"].index(r["source"])
            if int(np.argmax(r["scores"])) != src:
                continue
            neg = hard_negatives(r["scores"])
            if src in neg or len(neg) != RECIPE["mnrl"]["hard_negatives"]:
                raise ValueError(f"{r['id']}: the hard negatives {neg} include the source or are too few")
            kept.append({"id": r["id"], "query": r["query"], "pos": r["source"], "pos_text": r["texts"][src],
                         "neg": [r["candidates"][i] for i in neg], "neg_texts": [r["texts"][i] for i in neg]})
        out[corpus] = kept
        counts[corpus] = {"lists": len(rows), "eligible": len(kept), "not_eligible": len(rows) - len(kept)}
    return out, counts


class MnrlSampler:
    """Batches of queries_per_batch examples from one corpus each (drawn with probability proportional to the square root
    of its eligible lists), with no source document twice in a batch: an example whose source is already in the batch
    stays at the front of its corpus's queue for the next batch (deferred, never dropped). A corpus's queue is extended
    by a fresh permutation of its examples when it runs short."""

    def __init__(self, sizes: dict[str, int], sources: dict[str, list[str]], batch: int, seed: int):
        self.names = sorted(c for c, n in sizes.items() if n > 0)
        weights = np.sqrt([sizes[c] for c in self.names])
        self.p = weights / weights.sum()
        self.sizes, self.sources, self.batch = sizes, sources, batch
        self.rng = np.random.RandomState(seed)
        self.queue: dict[str, list[int]] = {c: [] for c in self.names}
        self.deferred = 0
        for c in self.names:
            if len(set(sources[c])) < batch:
                raise ValueError(f"{c}: {len(set(sources[c]))} distinct source documents, fewer than a batch of {batch}")

    def draw(self) -> tuple[str, list[int]]:
        corpus = self.names[int(self.rng.choice(len(self.names), p=self.p))]
        q, picked, seen, skipped = self.queue[corpus], [], set(), 0
        at = 0
        while len(picked) < self.batch:
            if at == len(q):
                q.extend(int(i) for i in self.rng.permutation(self.sizes[corpus]))
            i = q[at]
            if self.sources[corpus][i] in seen:
                at += 1
                skipped += 1
                continue
            picked.append(i)
            seen.add(self.sources[corpus][i])
            q.pop(at)
        self.deferred += skipped
        return corpus, picked

    def state(self) -> dict[str, Any]:
        return {"rng": self.rng.get_state(), "queue": {c: list(v) for c, v in self.queue.items()}, "deferred": self.deferred}

    def load_state(self, state: dict[str, Any]) -> None:
        self.rng.set_state(state["rng"])
        self.queue = {c: list(v) for c, v in state["queue"].items()}
        self.deferred = state["deferred"]


def validation_batches(examples: dict[str, list[dict]], batch: int, seed: int = 0) -> list[tuple[str, list[int]]]:
    """The MNRL validation batches, built once: per corpus the examples permuted by RandomState(seed), then greedily up
    to `batch` per batch with no source document twice; every example in exactly one batch."""
    out = []
    rng = np.random.RandomState(seed)
    for corpus in sorted(examples):
        left = [int(i) for i in rng.permutation(len(examples[corpus]))]
        while left:
            picked, seen, rest = [], set(), []
            for i in left:
                if len(picked) < batch and examples[corpus][i]["pos"] not in seen:
                    picked.append(i)
                    seen.add(examples[corpus][i]["pos"])
                else:
                    rest.append(i)
            out.append((corpus, picked))
            left = rest
    return out


def mnrl_columns(batch: Sequence[dict]) -> tuple[list[str], list[str], list[str]]:
    """(query texts, candidate texts, candidate doc ids) in the library's column layout: every positive, then every
    first hard negative, and so on (MultipleNegativesRankingLoss: torch.cat(docs))."""
    k = len(batch[0]["neg"])
    texts = [e["pos_text"] for e in batch] + [e["neg_texts"][j] for j in range(k) for e in batch]
    ids = [e["pos"] for e in batch] + [e["neg"][j] for j in range(k) for e in batch]
    return [e["query"] for e in batch], texts, ids


def mnrl_loss(queries, candidates, candidate_ids: Sequence[str], positive_ids: Sequence[str], scale: float | None = None):
    """(loss, masked entries): cross-entropy over scale * cos(query, every in-batch candidate), the anchor's own positive
    (column i) the target. For each anchor every candidate whose doc id is its positive's doc id scores -inf, except
    column i itself. With no such collision it equals sentence-transformers 6.1's MultipleNegativesRankingLoss."""
    import torch

    scale = RECIPE["mnrl"]["scale"] if scale is None else scale
    q = torch.nn.functional.normalize(queries, p=2, dim=1)
    c = torch.nn.functional.normalize(candidates, p=2, dim=1)
    sims = scale * (q @ c.T)
    b = q.shape[0]
    mask = torch.tensor([[candidate_ids[j] == positive_ids[i] and j != i for j in range(len(candidate_ids))] for i in range(b)],
                        dtype=torch.bool, device=sims.device)
    sims = sims.masked_fill(mask, float("-inf"))
    loss = torch.nn.functional.cross_entropy(sims, torch.arange(b, device=sims.device))
    return loss, int(mask.sum())


def validate_mnrl(model, tokens: Tokens, examples: dict[str, list[dict]], batches: list[tuple[str, list[int]]]) -> dict[str, Any]:
    """The masked loss on the validation batches (no gradient): each corpus's mean over its anchors, then the macro mean."""
    import torch

    model.eval()
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    with torch.no_grad():
        for corpus, idx in batches:
            batch = [examples[corpus][i] for i in idx]
            qt, ct, cid = mnrl_columns(batch)
            loss, _ = mnrl_loss(embed(model, tokens, qt, VALIDATION_TEXTS), embed(model, tokens, ct, VALIDATION_TEXTS), cid, [e["pos"] for e in batch])
            sums[corpus] = sums.get(corpus, 0.0) + float(loss) * len(batch)
            counts[corpus] = counts.get(corpus, 0) + len(batch)
    loss = {c: sums[c] / counts[c] for c in sums}
    return {"loss": loss, "macro": macro_validation(loss), "anchors": counts}


# ---------------------------------------------------------------- state for resuming


def rng_states() -> dict[str, Any]:
    import torch

    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_states(state: dict[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_resume(path: Path, state: dict[str, Any]) -> None:
    """Write the resume state atomically: a temporary file in the same directory, then os.replace."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_resume(path: Path) -> dict[str, Any] | None:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False) if path.exists() else None


def _cpu_state(model) -> dict[str, Any]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def sampler_state(sampler: MixedSampler) -> dict[str, Any]:
    """MixedSampler's whole state (its RandomState, every corpus's order and position): a resumed run draws exactly the
    lists an uninterrupted one would."""
    return {"rng": sampler.rng.get_state(), "order": {c: np.asarray(o).copy() for c, o in sampler.order.items()}, "at": dict(sampler.at)}


def load_sampler_state(sampler: MixedSampler, state: dict[str, Any]) -> None:
    sampler.rng.set_state(state["rng"])
    sampler.order = {c: np.asarray(o).copy() for c, o in state["order"].items()}
    sampler.at = dict(state["at"])


class StopFlag:
    """Set by SIGUSR1 (the batch script forwards Slurm's warning signal to this process); read after every step."""

    def __init__(self):
        self.signal = None

    def install(self) -> None:
        signal.signal(signal.SIGUSR1, self._handle)

    def _handle(self, signum, frame) -> None:  # noqa: ARG002
        self.signal = "SIGUSR1"


def _digest(values: Sequence) -> str:
    return hashlib.sha256(json.dumps(list(values), separators=(",", ":")).encode()).hexdigest()


# ---------------------------------------------------------------- training


def train(arm: str, lists: dict[str, dict[str, list[dict]]], s: float, *, device: str | None = None, resume_path: Path | None = None,
          micro_lists: int | None = None, time_limit_min: float | None = DEFAULT_TIME_LIMIT_MIN, max_steps: int | None = None,
          max_epochs: int | None = None, stop_at_step: int | None = None, mnrl_batch: int | None = None, stop_flag: StopFlag | None = None,
          segment: dict[str, Any] | None = None, log: Callable[[str], None] = print) -> tuple[Any, dict[str, Any] | None]:
    """(the model holding the chosen epoch's weights, the training record), or (None, None) when the run stopped early
    and saved its state to resume_path. `lists` are the arm's (training_set_lists). Smoke settings, each recorded and
    refused by `manifest`: max_steps cuts every epoch, max_epochs runs fewer epochs (the schedule over those),
    stop_at_step stops through the same path as the time limit, mnrl_batch replaces queries_per_batch."""
    import torch

    spec = ARMS[arm]
    loss_kind = spec["loss"]
    seed = spec["seed"]
    flags = set_fp32()
    started = time.time()
    model = load_init(spec["init"], device=device)
    from app.eval.models_2026 import check_embedder

    init_check = check_embedder(_wrap(model, spec["init"]))

    train_lists, val_lists = lists["train"], lists["validation"]
    per_step = RECIPE["lists_per_step"]
    epochs_total = RECIPE["epochs"] if max_epochs is None else max_epochs
    if loss_kind == "mnrl":
        examples, eligibility = mnrl_examples(train_lists)
        val_examples, val_eligibility = mnrl_examples(val_lists)
        batch_q = mnrl_batch or RECIPE["mnrl"]["queries_per_batch"]
        sizes = {c: len(rows) for c, rows in examples.items()}
        steps_per_epoch = math.ceil(sum(sizes.values()) / batch_q)
        val_batches = validation_batches(val_examples, batch_q, seed=0)
        texts = [t for rows in [*examples.values(), *val_examples.values()] for e in rows for t in (e["query"], e["pos_text"], *e["neg_texts"])]
    else:
        sizes = {c: len(rows) for c, rows in train_lists.items()}
        steps_per_epoch = math.ceil(sum(sizes.values()) / per_step)
        texts = list_texts([r for by in (train_lists, val_lists) for rows in by.values() for r in rows])
    if max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps)
    total = steps_per_epoch * epochs_total
    t_tok = time.perf_counter()
    tokens = Tokens(model, texts)
    tokenize_seconds = round(time.perf_counter() - t_tok, 1)
    log(f"{arm}: {len(tokens)} unique texts tokenized in {tokenize_seconds} s; {steps_per_epoch} steps per epoch, {total} in all")

    decay = [p for n, p in model.named_parameters() if p.requires_grad and not any(k in n for k in ("bias", "LayerNorm.weight"))]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in ("bias", "LayerNorm.weight"))]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": RECIPE["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
                                  lr=RECIPE["learning_rate"], betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, total, RECIPE["warmup_fraction"]))
    if loss_kind == "mnrl":
        sampler: Any = MnrlSampler(sizes, {c: [e["pos"] for e in rows] for c, rows in examples.items()}, batch_q, seed)
    else:
        sampler = MixedSampler(sizes, seed)

    def validate() -> dict[str, Any]:
        return validate_mnrl(model, tokens, val_examples, val_batches) if loss_kind == "mnrl" else validate_marginmse(model, tokens, val_lists, s)

    state = load_resume(resume_path) if resume_path is not None else None
    smoke = {"max_steps": max_steps, "max_epochs": max_epochs, "stop_at_step": stop_at_step, "mnrl_batch": mnrl_batch}
    shape = {k: v for k, v in smoke.items() if k != "stop_at_step"}  # what a resumed run must share with the run it resumes
    if state is not None:
        if state["arm"] != arm or state["total"] != total or state["s"] != s or state["shape"] != shape:
            raise SystemExit(f"{resume_path} is another run's state ({state['arm']}, {state['total']} steps, s {state['s']}, {state['shape']})")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if loss_kind == "mnrl":
            sampler.load_state(state["sampler"])
        else:
            load_sampler_state(sampler, state["sampler"])
        set_rng_states(state["rng"])
        log(f"resumed at epoch {state['epoch']}, step {state['step_in_epoch']} of {steps_per_epoch} (global step {state['global_step']})")
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        before = validate()  # the init's validation loss, epoch 0
        state = {"arm": arm, "total": total, "s": s, "shape": shape, "smoke_used": {}, "epoch": 1, "step_in_epoch": 0, "global_step": 0, "epochs": [],
                 "running": [], "pairs": 0, "drawn": {c: 0 for c in sizes}, "first_step": None, "before": before, "best": None,
                 "train_seconds": 0.0, "segments": [], "lr": [], "draws": [], "masked": 0, "deferred_at_epoch_start": 0}
        log(f"epoch 0 validation {json.dumps({c: round(v, 4) for c, v in before['loss'].items()})} macro {before['macro']:.4f}")
    seg = {**(segment or {}), "device": _device_name(model), "micro_lists": micro_lists, "tf32": flags, "start_step": state["global_step"],
           "started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "smoke": {k: v for k, v in smoke.items() if v is not None} or None}
    state["smoke_used"].update({k: v for k, v in smoke.items() if v is not None})
    t_seg = time.perf_counter()
    stop_reason = None
    model.train()
    while state["epoch"] <= epochs_total:
        while state["step_in_epoch"] < steps_per_epoch:
            optimizer.zero_grad(set_to_none=True)
            lr = scheduler.get_last_lr()[0]
            if loss_kind == "mnrl":
                corpus, idx = sampler.draw()
                batch = [examples[corpus][i] for i in idx]
                state["draws"].append([corpus, idx])
                state["drawn"][corpus] += len(batch)
                qt, ct, cid = mnrl_columns(batch)
                loss, masked = mnrl_loss(embed(model, tokens, qt), embed(model, tokens, ct), cid, [e["pos"] for e in batch])
                if not torch.isfinite(loss):
                    raise RuntimeError(f"the loss is not finite at step {state['global_step']}")
                loss.backward()
                step_loss = float(loss.item())
                state["masked"] += masked
                state["pairs"] += len(batch) * len(cid)
            else:
                batch, draws = [], []
                for _ in range(per_step):
                    corpus, i = sampler.draw()
                    state["drawn"][corpus] += 1
                    draws.append([corpus, i])
                    batch.append(train_lists[corpus][i])
                state["draws"].append(draws)
                step_loss, numpy_rows, chunks = 0.0, [], micro_chunks(len(batch), micro_lists)
                for start, end in chunks:  # one chunk, the whole step, unless micro_lists
                    chunk = batch[start:end]
                    q = embed(model, tokens, [r["query"] for r in chunk])
                    d = embed(model, tokens, [t for r in chunk for t in r["texts"]])
                    losses, at = [], 0
                    for j, r in enumerate(chunk):
                        n = len(r["texts"])
                        student = s * (d[at: at + n] @ q[j])
                        delta = student - torch.tensor(r["scores"], dtype=torch.float32, device=student.device)
                        diff = delta[:, None] - delta[None, :]
                        losses.append((diff ** 2).sum() / (n * (n - 1)) if n > 1 else diff.sum() * 0.0)
                        if state["first_step"] is None:
                            numpy_rows.append(margin_mse(student.detach().cpu().numpy(), r["scores"]))
                        at += n
                    part = torch.stack(losses).sum() / len(batch)  # divided by the step's lists: the chunks add up to its mean
                    if not torch.isfinite(part):
                        raise RuntimeError(f"the loss is not finite at step {state['global_step']}")
                    part.backward()
                    step_loss += float(part.item())
                    state["pairs"] += at
                if state["first_step"] is None:  # the sum of the tensors backward() ran on, against the numpy definition
                    state["first_step"] = {"torch": step_loss, "numpy": float(np.mean(numpy_rows)), "chunks": len(chunks)}
                    if abs(step_loss - state["first_step"]["numpy"]) > 1e-3 * max(1.0, state["first_step"]["numpy"]):
                        raise RuntimeError(f"the training loss is not the declared MarginMSE: {state['first_step']}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            state["lr"].append(lr)
            state["running"].append(step_loss)
            state["step_in_epoch"] += 1
            state["global_step"] += 1
            epoch_ended = state["step_in_epoch"] == steps_per_epoch
            if epoch_ended:  # the epoch's end: validate, keep the best, save
                val = validate()
                model.train()
                epoch = state["epoch"]
                record = {"epoch": epoch, "step": state["global_step"], "train_loss": round(float(np.mean(state["running"])), 6),
                          "validation_loss": {c: round(v, 6) for c, v in val["loss"].items()}, "validation_macro": round(val["macro"], 6),
                          "pairs": state["pairs"], "seconds": round(state["train_seconds"] + time.perf_counter() - t_seg, 1)}
                if "source_mean_rank" in val:
                    record["source_mean_rank"] = {c: round(v, 4) for c, v in val["source_mean_rank"].items()}
                if loss_kind == "mnrl":
                    record.update(masked_entries=state["masked"], deferred=sampler.deferred - state["deferred_at_epoch_start"], dropped=0)
                    state["masked"], state["deferred_at_epoch_start"] = 0, sampler.deferred
                state["epochs"].append(record)
                log(f"epoch {epoch} train {record['train_loss']:.4f} validation {json.dumps(record['validation_loss'])} macro "
                    f"{record['validation_macro']:.4f} ({record['seconds']} s)")
                if choose_epoch([e["validation_macro"] for e in state["epochs"]]) == epoch:
                    state["best"] = {"epoch": epoch, "weights": _cpu_state(model)}
                state.update(epoch=epoch + 1, step_in_epoch=0, running=[], pairs=0)
                if resume_path is not None:
                    _save(resume_path, state, model, optimizer, scheduler, sampler, loss_kind, t_seg, seg)
            reason = _stop_reason(stop_flag, time_limit_min, started, stop_at_step, state["global_step"])
            if reason and state["global_step"] < total:
                stop_reason = reason
                break
            if epoch_ended:  # on to the next epoch (the outer loop checks whether there is one)
                break
        if stop_reason:
            break
    if stop_reason:
        if resume_path is None:
            raise RuntimeError("a run that can stop early needs a resume path")
        seg.update(end_step=state["global_step"], stopped=stop_reason)
        _save(resume_path, state, model, optimizer, scheduler, sampler, loss_kind, t_seg, seg, close_segment=True)
        log(f"stopped ({stop_reason}) after step {state['global_step']} of {total}; state saved; run the same command to resume")
        return None, None
    state["train_seconds"] += time.perf_counter() - t_seg
    state["segments"].append({**seg, "end_step": state["global_step"], "stopped": None})
    chosen = choose_epoch([e["validation_macro"] for e in state["epochs"]])
    model.load_state_dict(state["best"]["weights"])
    model.eval()
    record = {
        "arm": arm, "attempt": ATTEMPT, "spec": spec, "init": {**INITS[spec["init"]], "load_and_score": init_check}, "encoding": check_encoding(model),
        "recipe": RECIPE, "s": s, "smoke": state["smoke_used"] or None,
        "lists": {"train": {c: len(rows) for c, rows in train_lists.items()}, "validation": {c: len(rows) for c, rows in val_lists.items()},
                  "pairs_train": {c: sum(len(r["texts"]) for r in rows) for c, rows in train_lists.items()}},
        "steps_per_epoch": steps_per_epoch, "steps": total, "lists_drawn": state["drawn"], "unique_texts": len(tokens), "tokenize_seconds": tokenize_seconds,
        "validation_before_training": {"loss": {c: round(v, 6) for c, v in state["before"]["loss"].items()}, "macro": round(state["before"]["macro"], 6),
                                       **({"source_mean_rank": {c: round(v, 4) for c, v in state["before"]["source_mean_rank"].items()}}
                                          if "source_mean_rank" in state["before"] else {})},
        "epochs": state["epochs"], "chosen_epoch": chosen, "chosen_is_last": chosen == len(state["epochs"]),
        "beats_epoch_0": min(e["validation_macro"] for e in state["epochs"]) < round(state["before"]["macro"], 6),
        "first_step_loss": state["first_step"], "train_seconds": round(state["train_seconds"], 1), "segments": state["segments"],
        "schedule": {"lr_sha256": _digest(state["lr"]), "draws_sha256": _digest(state["draws"]), "steps_logged": len(state["lr"])},
        "memory": {"micro_lists_per_segment": [g.get("micro_lists") for g in state["segments"]], "forward_texts": FORWARD_TEXTS,
                   "note": "how a step's lists were split for the GPU's memory; the optimizer steps once per lists_per_step lists either way"},
        "tf32": flags, "device": _device_name(model),
    }
    if loss_kind == "mnrl":
        record["mnrl"] = {"eligibility": eligibility, "validation_eligibility": val_eligibility, "queries_per_batch": batch_q,
                          "validation_batches": len(val_batches), "deferred_total": sampler.deferred}
    if state["smoke_used"]:
        record["schedule"].update(lr=state["lr"], draws=state["draws"])  # a smoke run keeps them, for the resume test
    record["load_and_score"] = check_embedder(_wrap(model, arm))
    return model, record


def _device_name(model) -> str:
    try:
        import torch

        dev = model.device
        return torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
    except Exception:
        return str(getattr(model, "device", None))


def _stop_reason(flag: StopFlag | None, limit_min: float | None, started: float, stop_at_step: int | None, step: int) -> str | None:
    if flag is not None and flag.signal:
        return flag.signal
    if stop_at_step is not None and step == stop_at_step:
        return f"stop_at_step {stop_at_step} (smoke: the time-limit path at a fixed step)"
    if limit_min and (time.time() - started) / 60.0 > limit_min:
        return f"time limit {limit_min} min"
    return None


def _save(path, state, model, optimizer, scheduler, sampler, loss_kind, t_seg, seg, close_segment: bool = False) -> None:
    out = dict(state)
    out["train_seconds"] = state["train_seconds"] + time.perf_counter() - t_seg
    if close_segment:
        out["segments"] = [*state["segments"], dict(seg)]
    else:  # an epoch-end save while the segment runs: a job killed before it closes still shows what it did
        out["segments"] = [*state["segments"], {**seg, "end_step": state["global_step"], "stopped": "open (state saved at an epoch end)"}]
    out.update(model=_cpu_state(model), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
               sampler=sampler.state() if loss_kind == "mnrl" else sampler_state(sampler), rng=rng_states())
    save_resume(path, out)


class _Wrapped:
    """A trained SentenceTransformer seen as the app's embedder (encode() normalizes, float32), for the checks."""

    def __init__(self, model, name: str):
        dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension  # renamed in sentence-transformers 6
        self._model, self.name, self.dim = model, name, int(dim())

    def encode(self, texts):
        return np.asarray(self._model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


def _wrap(model, arm: str) -> _Wrapped:
    return _Wrapped(model, f"ft:{arm}")


# ---------------------------------------------------------------- checkpoint, reference embeddings, manifest


def reference_texts(lists: dict[str, dict[str, list[dict]]]) -> list[dict[str, str]]:
    """The texts whose embeddings the manifest records: the first fixture validation queries and the first candidates of
    the first fixture validation list (the fixture is in the repository, so any machine can embed them again)."""
    rows = sorted(lists["validation"]["fixture"], key=lambda r: r["id"])
    out = [{"kind": "query", "id": r["id"], "text": r["query"]} for r in rows[:REFERENCE_QUERIES]]
    first = rows[0]
    out += [{"kind": "document", "id": c, "text": t} for c, t in list(zip(first["candidates"], first["texts"]))[:REFERENCE_DOCUMENTS]]
    return out


def reference_embeddings(embedder, refs: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    vecs = np.asarray(embedder.encode([r["text"] for r in refs]), dtype=np.float64)
    return [{"kind": r["kind"], "id": r["id"], "embedding": [round(float(x), 7) for x in v]} for r, v in zip(refs, vecs)]


def check_reference(embedder, entry: dict, root: Path | None = None, texts: dict[tuple[str, str], str] | None = None) -> dict[str, Any]:
    """Whether a loaded adapted embedder gives the embeddings its manifest entry records for the reference texts (fixture
    queries and chunks, found again in the repository). Raises when one differs by more than REFERENCE_TOLERANCE."""
    if texts is None:
        from app.eval.distill_queries import fixture_documents

        root = root or _root()
        docs = dict(fixture_documents(root))
        queries = {q["id"]: q["query"] for q in read_jsonl(root / "bench" / "distill" / QUERIES_FILE)}
        texts = {("document", k): v for k, v in docs.items()} | {("query", k): v for k, v in queries.items()}
    refs = entry["reference_embeddings"]
    got = np.asarray(embedder.encode([texts[(r["kind"], r["id"])] for r in refs]), dtype=np.float64)
    worst = float(np.max(np.abs(got - np.asarray([r["embedding"] for r in refs], dtype=np.float64))))
    if worst > REFERENCE_TOLERANCE:
        raise RuntimeError(f"{entry.get('arm')}: the loaded weights differ from the trained ones on the reference texts (max |difference| {worst:.6f})")
    return {"texts": len(refs), "max_abs_difference": round(worst, 7), "tolerance": REFERENCE_TOLERANCE}


def save_checkpoint(model, directory: Path) -> dict[str, dict[str, Any]]:
    """Save in sentence-transformers format (no model card: it would carry this machine's details) and return every
    file's sha256 and size."""
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(directory), create_model_card=False)
    return {p.relative_to(directory).as_posix(): {"sha256": sha256_of(p), "bytes": p.stat().st_size}
            for p in sorted(directory.rglob("*")) if p.is_file() and p.name not in (TRAINING_FILE, SUMS_FILE)}


def path_leaks(directory: Path) -> list[str]:
    """Saved files (other than the weights) that contain an absolute path or this user's name."""
    user = os.environ.get("USER") or ""
    out = []
    for p in sorted(Path(directory).rglob("*")):
        if not p.is_file() or p.suffix in (".safetensors", ".bin", ".pt"):
            continue
        text = p.read_text(errors="replace")
        if any(n in text for n in LEAK_NEEDLES) or (len(user) > 3 and user in text):
            out.append(p.relative_to(directory).as_posix())
    return out


def verify_checkpoint(directory: Path, files: dict[str, dict[str, Any]]) -> None:
    """Every file again by sha256 against what was recorded at save time; then SHA256SUMS beside them."""
    bad = [rel for rel, f in files.items() if sha256_of(Path(directory) / rel) != f["sha256"]]
    if bad:
        raise RuntimeError(f"checkpoint files changed after saving: {bad}")
    (Path(directory) / SUMS_FILE).write_text("".join(f"{f['sha256']}  {rel}\n" for rel, f in sorted(files.items())))


def build_manifest(checkpoint_dir: Path, arms: Sequence[str] | None = None, *, existing: dict | None = None) -> dict[str, Any]:
    """bench/distill/embedders.json from each trained arm's training.json: refuses a smoke run, a dirty tree, weights
    that no longer have the sha256 their training recorded, and an arm trained by other trainer or declaration code
    than the bar row (bge-base may train later only with the same two files). Entries already in `existing` are kept
    unchanged; an arm without a training.json is left out (not trained)."""
    out: dict[str, Any] = dict((existing or {}).get("arms") or {})
    for tag in arms or ARMS:
        if tag in out and arms is None:
            continue
        path = Path(checkpoint_dir) / tag / TRAINING_FILE
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if record.get("smoke"):
            raise SystemExit(f"{tag}: its training.json is a smoke run")
        if record.get("working_tree_dirty"):
            raise SystemExit(f"{tag}: trained with uncommitted changes")
        weights = record["files"][WEIGHTS_FILE]
        found = sha256_of(Path(checkpoint_dir) / tag / WEIGHTS_FILE)
        if found != weights["sha256"]:
            raise SystemExit(f"{tag}: {WEIGHTS_FILE} has sha256 {found}, its training recorded {weights['sha256']}")
        out[tag] = {"directory": tag, "weights_sha256": weights["sha256"], "weights_bytes": weights["bytes"], **record}
    ref = next((out[t]["sources_sha256"] for t in ARMS if t in out), None)
    for tag, entry in out.items():
        entry["same_code_as_first_arm"] = entry.get("sources_sha256") == ref
    return {"about": ("the adapted embedders of app/eval/adapted_embedders.py: what each was trained from and on, its loss after "
                      "every epoch, its resume segments and the sha256 of its checkpoint files. The checkpoints are not in the "
                      "repository; a row that uses one records weights_sha256, and a load is checked against reference_embeddings"),
            "arms": {t: out[t] for t in ARMS if t in out}}


# ---------------------------------------------------------------- the scale step (make embedder-scale)


def compute_scales(root: Path, lists: dict[str, dict[str, list[dict]]], *, device: str | None = None, keys=None, sample: int | None = None,
                   log=print) -> dict[str, Any]:
    """Every SCALE_KEYS entry: the init's cosines over its training set's training lists, s by RECIPE["scale"], and the
    epoch-0 validation loss with that s. `sample` (smoke) takes the first n training lists of each corpus by id."""
    from app.eval.adapted_embedders import SCALE_KEYS

    out = {}
    models: dict[str, Any] = {}
    for init, tset in keys or SCALE_KEYS:
        if init not in models:
            models[init] = load_init(init, device=device)
        model = models[init]
        mine = training_set_lists(lists, tset)
        train_rows = [r for c in sorted(mine["train"]) for r in sorted(mine["train"][c], key=lambda r: r["id"])[: sample or None]]
        tokens = Tokens(model, list_texts(train_rows) + list_texts([r for rows in mine["validation"].values() for r in rows]))
        vectors = encode_lists(model, tokens, train_rows)
        fit = scale_from_lists([r["scores"] for r in train_rows], [list_cosines(vectors, r) for r in train_rows])
        val = validate_marginmse(model, tokens, mine["validation"], fit["s"])
        out[f"{init}/{tset}"] = {"init": init, "model": INITS[init]["model"], "revision": INITS[init]["revision"], "training_set": tset,
                                 "corpora": list(TRAINING_SETS[tset]), **fit,
                                 "validation_epoch_0": {"loss": {c: round(v, 6) for c, v in val["loss"].items()}, "macro": round(val["macro"], 6),
                                                        "source_mean_rank": {c: round(v, 4) for c, v in val["source_mean_rank"].items()}},
                                 "device": _device_name(model)}
        log(f"{init}/{tset}: s {fit['s']:.4f} (1/s {fit['temperature']:.5f}) over {fit['pairs']} pairs of {fit['lists']} lists; "
            f"epoch-0 validation macro {val['macro']:.4f}")
    return out


def read_scale(arm: str, path: Path) -> tuple[float, dict[str, Any]]:
    """(s, its entry) for an arm from the scale file."""
    spec = ARMS[arm]
    blob = json.loads(Path(path).read_text())
    entry = blob["entries"][f"{spec['init']}/{spec['training_set']}"]
    return float(entry["s"]), entry


# ---------------------------------------------------------------- command line


def _parse_args(argv: list[str] | None):
    root = _root()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["scale", "train", "manifest"])
    p.add_argument("--arm", choices=list(ARMS), help="train: which arm")
    p.add_argument("--checkpoint-dir", type=Path, help="train: where the checkpoint is written; manifest: where they are read")
    p.add_argument("--resume-dir", type=Path, default=None, help="train: where resume.pt lives (shared storage, survives the job)")
    p.add_argument("--data-dir", type=Path, default=root / "bench" / "external", help="where the BEIR corpora are")
    p.add_argument("--queries", type=Path, default=root / "bench" / "distill" / QUERIES_FILE)
    p.add_argument("--scores", type=Path, default=root / "bench" / "distill" / SCORES_FILE)
    p.add_argument("--scale-file", type=Path, default=root / SCALE_FILE, help="train: where s is read; scale: where it is written")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--micro-lists", type=int, default=None, help="train: accumulate gradients over chunks of this many lists (a GPU under 40 GB; recorded)")
    p.add_argument("--time-limit-min", type=float, default=DEFAULT_TIME_LIMIT_MIN, help="train: save and stop after this many minutes (0: no limit)")
    p.add_argument("--max-steps", type=int, default=None, help="smoke: cut every epoch to this many steps (never in the manifest)")
    p.add_argument("--max-epochs", type=int, default=None, help="smoke: run this many epochs (never in the manifest)")
    p.add_argument("--stop-at-step", type=int, default=None, help="smoke: stop after this global step through the time-limit path")
    p.add_argument("--mnrl-batch", type=int, default=None, help="smoke: MNRL queries per batch instead of the declared 32")
    p.add_argument("--sample", type=int, default=None, help="scale smoke: the first n training lists of each corpus")
    p.add_argument("--out", type=Path, default=root / MANIFEST_FILE, help="manifest: the committed file")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from app.eval.provenance import command_line, device_info, git_sha, git_state, run_provenance, scrub

    args = _parse_args(argv)
    root = _root()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = None if args.device == "auto" else args.device
    if args.step == "manifest":
        existing = json.loads(args.out.read_text()) if args.out.exists() else None
        manifest = build_manifest(args.checkpoint_dir, existing=existing)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(manifest, indent=2) + "\n")
        for tag, entry in manifest["arms"].items():
            print(tag, entry["weights_sha256"][:12], "epoch", entry["chosen_epoch"], "of", len(entry["epochs"]),
                  [e["validation_macro"] for e in entry["epochs"]], "same code" if entry["same_code_as_first_arm"] else "OTHER CODE")
        print(f"wrote {args.out}")
        return 0
    started = git_state(root)
    data = verify_data(args.queries, args.scores)
    queries, lists = load_lists(root, args.data_dir, args.queries, args.scores)
    header = {"git_sha": git_sha(root), **run_provenance(root=root, started=started), "timestamp": datetime.now(timezone.utc).isoformat(),
              "command": command_line(), "machine": device_info(args.device), "data": data, "sources_sha256": source_sha256(root),
              "teacher_and_query_writer": "not re-run: see bench/distill/queries.meta.json and teacher_scores.meta.json"}
    if args.step == "scale":
        entries = compute_scales(root, lists, device=device, sample=args.sample)
        import torch  # noqa: F401  (the flags below)

        blob = {**header, "tf32": tf32_flags(), "rule": RECIPE["scale"], "sample": args.sample, "entries": entries,
                "use": "the trainer reads s from here; nothing here read a gold question, a BEIR query or a qrel"}
        args.scale_file.parent.mkdir(parents=True, exist_ok=True)
        args.scale_file.write_text(json.dumps(scrub(blob), indent=2) + "\n")
        print(f"wrote {args.scale_file}")
        return 0
    if not args.arm or not args.checkpoint_dir:
        raise SystemExit("train needs --arm and --checkpoint-dir")
    directory = args.checkpoint_dir / args.arm
    if (directory / TRAINING_FILE).exists():
        raise SystemExit(f"{directory} already holds a finished run; it is never overwritten")
    s, scale_entry = read_scale(args.arm, args.scale_file)
    resume = (args.resume_dir / args.arm / RESUME_FILE) if args.resume_dir else None
    flag = StopFlag()
    flag.install()
    segment = {"job": os.environ.get("SLURM_JOB_ID"), "host_class": (device_info(args.device) or {}).get("device_name")}
    model, record = train(args.arm, training_set_lists(lists, ARMS[args.arm]["training_set"]), s, device=device, resume_path=resume,
                          micro_lists=args.micro_lists, time_limit_min=args.time_limit_min or None, max_steps=args.max_steps,
                          max_epochs=args.max_epochs, stop_at_step=args.stop_at_step, mnrl_batch=args.mnrl_batch, stop_flag=flag, segment=segment)
    if model is None:
        return STOPPED
    files = save_checkpoint(model, directory)
    verify_checkpoint(directory, files)
    refs = reference_texts(lists)
    record.update({**header, "scale": {"file": SCALE_FILE, "sha256": sha256_of(args.scale_file), "entry": scale_entry},
                   "files": files, "reference_embeddings": reference_embeddings(_wrap(model, args.arm), refs), "path_leaks": path_leaks(directory)})
    from app.eval.adapted_embedders import load_checkpoint_embedder

    reloaded = load_checkpoint_embedder(directory, args.arm, device=device)
    record["reload_check"] = check_reference(reloaded, record, texts={(r["kind"], r["id"]): r["text"] for r in refs})
    (directory / TRAINING_FILE).write_text(json.dumps(scrub(record), indent=2) + "\n")
    if resume is not None and resume.exists():  # the checkpoint verified by sha256: the run's state is no longer needed
        os.remove(resume)
    print(json.dumps({"arm": args.arm, "chosen_epoch": record["chosen_epoch"], "chosen_is_last": record["chosen_is_last"],
                      "weights_sha256": files[WEIGHTS_FILE]["sha256"], "reload_check": record["reload_check"], "train_seconds": record["train_seconds"],
                      "path_leaks": record["path_leaks"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
