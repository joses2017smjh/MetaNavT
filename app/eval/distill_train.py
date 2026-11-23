"""Train a student reranker on the teacher's scores (M43), by the recipe declared in app/eval/small_rerankers.py.

    python -m app.eval.distill_train train --student student-minilm-l6 --data-dir <BEIR dir> --checkpoint-dir <dir>
    python -m app.eval.distill_train manifest --checkpoint-dir <dir>        # -> bench/distill/students.json

A student is one of the small cross-encoders (STUDENTS names the candidate it starts from), fine-tuned without a
label: its targets are the teacher's logits for the candidates of the synthetic queries
(bench/distill/teacher_scores.jsonl.gz), and its checkpoint is the epoch end with the lowest loss on the synthetic
queries of the held-out source documents. Nothing here opens a gold question, a BEIR query or a qrel: only the two
committed files and the texts of the candidates (the fixture's chunks, each BEIR corpus.jsonl).

Every setting comes from RECIPE; this module has no hyper-parameter of its own. The loss is MarginMSE over every pair
of candidates in one query's list: with d = student logits - teacher logits, the mean over i < j of (d_i - d_j)^2.
Pairs are tokenized and scored by the CrossEncoder's own preprocess() and forward, the path predict() takes when the
model serves, so training and scoring cut a long passage at the same token.

What is written: the checkpoint (sentence-transformers format, float32 safetensors) under --checkpoint-dir/<student>/,
with training.json beside it: the recipe, the files it trained on by sha256, the loss after every epoch, the epoch
kept, the sha256 of every checkpoint file, and the student's logits for a few fixture lists (REFERENCE_LISTS), which
`manifest` copies into the committed bench/distill/students.json so that a later load on any machine can be checked
against the weights that were trained (check_reference). The checkpoints themselves are not committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.eval.distill_queries import QUERIES_FILE, read_jsonl
from app.eval.distill_teacher import SCORES_FILE, corpus_texts
from app.eval.provenance import command_line, device_info, git_sha, git_state, model_device, run_provenance, scrub
from app.eval.small_rerankers import ATTEMPT, CANDIDATES, RECIPE, STUDENT_MANIFEST, STUDENTS, WEIGHTS_FILE, check_revision, sha256_of

TRAINING_FILE = "training.json"
REFERENCE_LISTS = 3  # fixture validation lists whose student logits are recorded: the fixture is in the repository
REFERENCE_TOLERANCE = 1e-3  # a reload on another device may differ in the last bits, not in the third decimal


def margin_mse(student: Sequence[float], teacher: Sequence[float]) -> float:
    """MarginMSE over every pair of one list: the mean over i < j of ((s_i - s_j) - (t_i - t_j))^2. A list of one
    candidate has no pair and no loss."""
    d = np.asarray(student, dtype=np.float64) - np.asarray(teacher, dtype=np.float64)
    n = d.shape[0]
    if n < 2:
        return 0.0
    diff = d[:, None] - d[None, :]
    return float((diff ** 2).sum() / (n * (n - 1)))


def lr_factor(step: int, total: int, warmup_fraction: float) -> float:
    """The learning rate's multiplier at optimizer step `step` (0-based) of `total`: linear warm-up over the first
    warmup_fraction of the steps, then linear decay that reaches zero after the last step."""
    warmup = int(warmup_fraction * total)
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total - step) / max(1, total - warmup))


class MixedSampler:
    """Training lists drawn as RECIPE["corpus_mixing"] says: the corpus of each draw with probability proportional to
    the square root of its number of lists; inside a corpus without replacement, reshuffled when exhausted."""

    def __init__(self, sizes: dict[str, int], seed: int):
        self.names = sorted(c for c, n in sizes.items() if n > 0)
        weights = np.sqrt([sizes[c] for c in self.names])
        self.p = weights / weights.sum()
        self.sizes = sizes
        self.rng = np.random.RandomState(seed)
        self.order = {c: self.rng.permutation(sizes[c]) for c in self.names}
        self.at = {c: 0 for c in self.names}

    def draw(self) -> tuple[str, int]:
        corpus = self.names[int(self.rng.choice(len(self.names), p=self.p))]
        if self.at[corpus] == self.sizes[corpus]:
            self.order[corpus], self.at[corpus] = self.rng.permutation(self.sizes[corpus]), 0
        self.at[corpus] += 1
        return corpus, int(self.order[corpus][self.at[corpus] - 1])


def build_lists(queries: Sequence[dict], scores: Sequence[dict], texts: dict[str, dict[str, str]]) -> dict[str, dict[str, list[dict]]]:
    """{split: {corpus: [list]}}: each list a query, its candidates' texts and the teacher's logits, in id order. Every
    query must have its scored list."""
    by_id = {r["id"]: r for r in scores}
    out: dict[str, dict[str, list[dict]]] = {"train": {}, "validation": {}}
    for q in sorted(queries, key=lambda q: q["id"]):
        row = by_id.get(q["id"])
        if row is None:
            raise KeyError(f"no teacher scores for {q['id']}")
        if len(row["candidates"]) != len(row["scores"]):
            raise ValueError(f"{q['id']}: {len(row['candidates'])} candidates, {len(row['scores'])} scores")
        out[q["split"]].setdefault(q["corpus"], []).append(
            {"id": q["id"], "query": q["query"], "candidates": row["candidates"], "texts": [texts[q["corpus"]][c] for c in row["candidates"]],
             "scores": [float(s) for s in row["scores"]]})
    return out


def choose_epoch(validation: Sequence[float]) -> int:
    """The 1-based epoch with the lowest validation loss; ties go to the earlier epoch."""
    return int(np.argmin(np.asarray(validation, dtype=np.float64))) + 1


def macro_validation(per_corpus: dict[str, float]) -> float:
    """The selection loss: the mean over corpora of each corpus's mean list loss."""
    return float(np.mean([per_corpus[c] for c in sorted(per_corpus)]))


def _forward(model, pairs):
    """The CrossEncoder's logits for (query, text) pairs with gradients: preprocess() and forward, as predict() does."""
    from sentence_transformers.util import batch_to_device

    features = batch_to_device(model.preprocess([list(p) for p in pairs]), model.device)
    scores = model(features)["scores"]
    return scores.squeeze(-1) if scores.ndim > 1 else scores


def _validate(model, lists: dict[str, list[dict]], batch_lists: int = 32) -> dict[str, float]:
    """Mean MarginMSE of the model on each corpus's validation lists (no gradient)."""
    import torch

    model.eval()
    out = {}
    with torch.no_grad():
        for corpus, rows in lists.items():
            losses = []
            for start in range(0, len(rows), batch_lists):
                block = rows[start: start + batch_lists]
                logits = _forward(model, [(r["query"], t) for r in block for t in r["texts"]]).float().cpu().numpy()
                at = 0
                for r in block:
                    losses.append(margin_mse(logits[at: at + len(r["texts"])], r["scores"]))
                    at += len(r["texts"])
            out[corpus] = float(np.mean(losses))
    return out


def train(student: str, lists: dict[str, dict[str, list[dict]]], *, device: str | None = None, max_steps: int | None = None,
          log=print) -> tuple[Any, dict[str, Any]]:
    """(the student CrossEncoder holding the chosen epoch's weights, the training record). max_steps cuts every epoch
    short for a smoke run, which the record says and `manifest` refuses."""
    import torch

    from app.eval.models_2026 import ModelCheckFailed, check_reranker
    from app.eval.provenance import hf_revision
    from app.retrieval.rerank import cross_encoder_provenance, get_cross_encoder

    init = CANDIDATES[STUDENTS[student]]
    check_revision(STUDENTS[student], hf_revision(init["model"]))
    seed = RECIPE["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = get_cross_encoder(init["model"], precision="fp32", max_length=RECIPE["max_length"], device=device, raise_errors=True)
    init_check = check_reranker(model, init["model"])
    train_lists, val_lists = lists["train"], lists["validation"]
    sizes = {c: len(rows) for c, rows in train_lists.items()}
    per_step = RECIPE["lists_per_step"]
    steps_per_epoch = math.ceil(sum(sizes.values()) / per_step)
    if max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps)
    total = steps_per_epoch * RECIPE["epochs"]
    decay = [p for n, p in model.named_parameters() if p.requires_grad and not any(k in n for k in ("bias", "LayerNorm.weight"))]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in ("bias", "LayerNorm.weight"))]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": RECIPE["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
                                  lr=RECIPE["learning_rate"], betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, total, RECIPE["warmup_fraction"]))
    sampler = MixedSampler(sizes, seed)
    before = _validate(model, val_lists)
    log(f"epoch 0 validation {json.dumps({c: round(v, 4) for c, v in before.items()})} macro {macro_validation(before):.4f}")
    epochs, best_state, first_step_check = [], None, None
    drawn = {c: 0 for c in sizes}
    t0 = time.perf_counter()
    for epoch in range(1, RECIPE["epochs"] + 1):
        model.train()
        running, pairs_seen = [], 0
        for _ in range(steps_per_epoch):
            batch = []
            for _ in range(per_step):
                corpus, i = sampler.draw()
                drawn[corpus] += 1
                batch.append(train_lists[corpus][i])
            logits = _forward(model, [(r["query"], t) for r in batch for t in r["texts"]]).float()
            losses, at = [], 0
            for r in batch:
                n = len(r["texts"])
                d = logits[at: at + n] - torch.tensor(r["scores"], dtype=torch.float32, device=logits.device)
                diff = d[:, None] - d[None, :]
                losses.append((diff ** 2).sum() / (n * (n - 1)) if n > 1 else diff.sum() * 0.0)
                at += n
            loss = torch.stack(losses).mean()
            if first_step_check is None:  # the torch loss of the first step against the numpy definition above
                rows, at = [], 0
                for r in batch:
                    rows.append(margin_mse(logits[at: at + len(r["texts"])].detach().cpu().numpy(), r["scores"]))
                    at += len(r["texts"])
                first_step_check = {"torch": float(loss.item()), "numpy": float(np.mean(rows))}
                if abs(first_step_check["torch"] - first_step_check["numpy"]) > 1e-3 * max(1.0, first_step_check["numpy"]):
                    raise RuntimeError(f"the training loss is not the declared MarginMSE: {first_step_check}")
            if not torch.isfinite(loss):
                raise RuntimeError(f"the loss is not finite at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            running.append(float(loss.item()))
            pairs_seen += at
        val = _validate(model, val_lists)
        macro = macro_validation(val)
        epochs.append({"epoch": epoch, "train_loss": round(float(np.mean(running)), 6), "validation_loss": {c: round(v, 6) for c, v in val.items()},
                       "validation_macro": round(macro, 6), "pairs": pairs_seen, "seconds": round(time.perf_counter() - t0, 1)})
        log(f"epoch {epoch} train {epochs[-1]['train_loss']:.4f} validation {json.dumps(epochs[-1]['validation_loss'])} macro {macro:.4f} "
            f"({epochs[-1]['seconds']} s)")
        if choose_epoch([e["validation_macro"] for e in epochs]) == epoch:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    chosen = choose_epoch([e["validation_macro"] for e in epochs])
    model.load_state_dict(best_state)
    model.eval()
    record = {
        "student": student, "attempt": ATTEMPT, "init": {"tag": STUDENTS[student], **init, "load_and_score": init_check},
        "recipe": RECIPE, "smoke_run_max_steps": max_steps,
        "lists": {"train": sizes, "validation": {c: len(rows) for c, rows in val_lists.items()},
                  "pairs_train": {c: sum(len(r["texts"]) for r in rows) for c, rows in train_lists.items()}},
        "steps_per_epoch": steps_per_epoch, "steps": total, "lists_drawn": drawn,
        "validation_before_training": {**{c: round(v, 6) for c, v in before.items()}, "macro": round(macro_validation(before), 6)},
        "epochs": epochs, "chosen_epoch": chosen, "first_step_loss": first_step_check,
        "train_seconds": round(time.perf_counter() - t0, 1), "device": model_device(model), **cross_encoder_provenance(model),
    }
    try:  # recorded, not raised: the checkpoint is saved either way, and a row that loads a failing student stops there
        record["load_and_score"] = check_reranker(model, student)
    except ModelCheckFailed as e:
        record["load_and_score"] = {"passed": False, "error": str(e)}
    return model, record


def reference_logits(model, lists: Sequence[dict]) -> list[dict]:
    """The model's logits for a few lists, through the scorer every benchmark row uses."""
    from app.retrieval.rerank import cross_encoder_logits

    out = []
    for r in lists:
        out.append({"id": r["id"], "candidates": r["candidates"],
                    "logits": [round(x, 6) for x in cross_encoder_logits(model, [(r["query"], t) for t in r["texts"]])]})
    return out


def check_reference(model, entry: dict, root: Path, queries: Sequence[dict] | None = None) -> dict[str, Any]:
    """Whether a loaded student gives the logits its manifest entry records for the reference lists (fixture lists: the
    texts are in the repository). Raises when one differs by more than REFERENCE_TOLERANCE: the weights loaded are not
    the weights that were trained, or they are not read the same way."""
    from app.eval.distill_queries import fixture_documents
    from app.retrieval.rerank import cross_encoder_logits

    texts = dict(fixture_documents(root))
    by_id = {q["id"]: q for q in (queries if queries is not None else read_jsonl(root / "bench" / "distill" / QUERIES_FILE))}
    worst = 0.0
    for ref in entry["reference_logits"]:
        got = cross_encoder_logits(model, [(by_id[ref["id"]]["query"], texts[c]) for c in ref["candidates"]])
        worst = max(worst, float(np.max(np.abs(np.asarray(got) - np.asarray(ref["logits"])))))
    if worst > REFERENCE_TOLERANCE:
        raise RuntimeError(f"{entry.get('student')}: the loaded weights differ from the trained ones on the reference lists (max |difference| {worst:.6f})")
    return {"lists": len(entry["reference_logits"]), "max_abs_difference": round(worst, 6), "tolerance": REFERENCE_TOLERANCE}


def save_checkpoint(model, directory: Path) -> dict[str, dict[str, Any]]:
    """Save in sentence-transformers format (no model card: it would carry this machine's details) and return every
    file's sha256 and size."""
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(directory), create_model_card=False)
    return {p.relative_to(directory).as_posix(): {"sha256": sha256_of(p), "bytes": p.stat().st_size}
            for p in sorted(directory.rglob("*")) if p.is_file() and p.name != TRAINING_FILE}


def build_manifest(checkpoint_dir: Path, students: Sequence[str] | None = None) -> dict[str, Any]:
    """bench/distill/students.json from each student's training.json: refuses a smoke run, a dirty tree and weights
    that no longer have the sha256 their training recorded."""
    out: dict[str, Any] = {}
    for tag in students or STUDENTS:
        path = Path(checkpoint_dir) / tag / TRAINING_FILE
        if not path.exists():
            raise SystemExit(f"{tag}: no {TRAINING_FILE} in the checkpoint directory")
        record = json.loads(path.read_text())
        if record.get("smoke_run_max_steps") is not None:
            raise SystemExit(f"{tag}: its training.json is a smoke run")
        if record.get("working_tree_dirty"):
            raise SystemExit(f"{tag}: trained with uncommitted changes")
        weights = record["files"][WEIGHTS_FILE]
        found = sha256_of(Path(checkpoint_dir) / tag / WEIGHTS_FILE)
        if found != weights["sha256"]:
            raise SystemExit(f"{tag}: {WEIGHTS_FILE} has sha256 {found}, its training recorded {weights['sha256']}")
        out[tag] = {"directory": tag, "weights_sha256": weights["sha256"], "weights_bytes": weights["bytes"], **record}
    return {"about": ("the students of app/eval/small_rerankers.py: what each was trained from and on, its loss after every epoch, "
                      "and the sha256 of its checkpoint files. The checkpoints are not in the repository; a row that uses one "
                      "records weights_sha256, and a load is checked against reference_logits"),
            "students": out}


def _parse_args(argv: list[str] | None):
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["train", "manifest"])
    p.add_argument("--student", choices=sorted(STUDENTS), help="train: which student")
    p.add_argument("--checkpoint-dir", type=Path, required=True, help="where checkpoints are written (train) and read (manifest)")
    p.add_argument("--data-dir", type=Path, default=root / "bench" / "external", help="where the BEIR corpora are")
    p.add_argument("--queries", type=Path, default=root / "bench" / "distill" / QUERIES_FILE)
    p.add_argument("--scores", type=Path, default=root / "bench" / "distill" / SCORES_FILE)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--max-steps", type=int, default=None, help="train: cut every epoch to this many steps (a smoke run; never recorded in the manifest)")
    p.add_argument("--out", type=Path, default=root / STUDENT_MANIFEST, help="manifest: the committed file")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.step == "manifest":
        manifest = build_manifest(args.checkpoint_dir)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(manifest, indent=2) + "\n")
        for tag, entry in manifest["students"].items():
            print(tag, entry["weights_sha256"][:12], entry["weights_bytes"], "bytes; epoch", entry["chosen_epoch"],
                  [e["validation_macro"] for e in entry["epochs"]])
        print(f"wrote {args.out}")
        return 0
    if not args.student:
        raise SystemExit("train needs --student")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    started = git_state(root)
    queries, scores = read_jsonl(args.queries), read_jsonl(args.scores)
    texts = {c: corpus_texts(c, root, args.data_dir) for c in sorted({q["corpus"] for q in queries})}
    lists = build_lists(queries, scores, texts)
    model, record = train(args.student, lists, device=None if args.device == "auto" else args.device, max_steps=args.max_steps)
    directory = args.checkpoint_dir / args.student
    files = save_checkpoint(model, directory)
    from app.retrieval.rerank import get_cross_encoder

    reference = reference_logits(model, lists["validation"]["fixture"][:REFERENCE_LISTS])
    reloaded = get_cross_encoder(str(directory), precision="fp32", max_length=RECIPE["max_length"],
                                 device=None if args.device == "auto" else args.device, raise_errors=True)
    record.update({
        "git_sha": git_sha(root), **run_provenance(root=root, started=started), "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command_line(), "machine": device_info(args.device),
        "data": {QUERIES_FILE: hashlib.sha256(args.queries.read_bytes()).hexdigest(), SCORES_FILE: hashlib.sha256(args.scores.read_bytes()).hexdigest()},
        "files": files, "reference_logits": reference,
    })
    record["reload_check"] = check_reference(reloaded, record, root, queries)  # the saved files, loaded again, score as the trained model
    (directory / TRAINING_FILE).write_text(json.dumps(scrub(record), indent=2) + "\n")
    print(json.dumps({"student": args.student, "chosen_epoch": record["chosen_epoch"], "weights_sha256": files[WEIGHTS_FILE]["sha256"],
                      "reload_check": record["reload_check"], "train_seconds": record["train_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
