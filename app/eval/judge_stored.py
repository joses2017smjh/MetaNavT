"""Judge stored answers with a judge that is validated in the same file (M42).

    python -m app.eval.judge_stored --judge gpt-oss:120b --answers FILE[,FILE...] --out OUT.json [--resume]
    python -m app.eval.judge_stored --check OUT.json

app/eval/llm_eval.py generates answers and judges them in one process. This module is the judge alone, over answers
that already exist, so that a judge can be added without generating anything again, a generator's answers survive a
judge that fails, and answers recorded elsewhere (the served Claude answers) are judged by one command:

1. The gate. The judge labels the calibration answers stored in the committed bench/results/jury_v2.json (the 160
   answers qwen2.5:1.5b gave to the simple-factual questions from the retrieved and from a planted context) with the
   correctness prompt (app/eval/judge.py correctness_prompt: question, gold answer, answer; no retrieved context).
   Cohen's kappa between its verdicts and the bounded exact-match labels must reach KAPPA_GATE (0.6); the interval
   resamples questions, as llm_eval does. Nothing is regenerated: the items are read from the committed file, each
   reference label is recomputed and must equal the stored one, and the file records the sha256 of the items the
   judge saw (`calibration.items_sha256`).
2. The answers. Each FILE is either an answer-evaluation file (llm_eval's rows: jury_v2.json, or a generate-only
   run's jury_v3_answers_<generator>.json) or a JSONL with one answer per line:

       {"id": "q001", "question": "...", "gold": "...", "answer": "...", "cited_paths": ["configs/run_040.yaml"]}

   (`category`, `extractive` and `retrieved_paths`, the paths whose excerpts the model was shown, are optional; an
   id of the gold set must carry that question and gold answer, and takes its category). The judge gives every answer the correctness verdict, under llm_eval's rules: an answer with
   no citation fails loud and is not judged (`uncited_answer`; in a JSONL also an empty answer, `no_answer`), and an
   answer that is NOT IN SOURCES and nothing else, to a question with a gold answer, is wrong by rule without a call.
   Where a row stores the extractive answer the judge also compares the two in both orders (AB/BA), which measures
   its position bias. The M4 composite label needs the retrieved context and is never gated or published: it is not
   collected here.
3. The file. It is written after every phase and every CHECKPOINT verdicts (calibration, then each answer set's
   verdicts, then its pairs), so a run cut short keeps what it judged and `--resume` continues it: stored verdicts
   are kept, a call that failed is asked again. A resumed run must be the same judge (model digest, server version,
   context length), the same calibration items and the same protocol, or it refuses.

A judge's scores of an answer set may be published only if its gate passed in this file (`gate.passes`); the
summary says so either way. The file states what its numbers do not show (`does_not_show`, DOES_NOT_SHOW: text, not a
measurement, kept outside `protocol` so that a file written before it existed can still be resumed). A judge that is not there is not a judge that failed: before any item is judged the
model is loaded, and the run stops with `status: could not load` unless the server reports it wholly on the GPU and
a warm-up prompt returns; five calls failing in a row stop the run as `interrupted`. Empty and unreadable replies
are counted per phase (an unreadable reply reads as partial, as in llm_eval).

`--check` recomputes the gate and every summary from the stored verdicts, and the calibration items' and answer
files' hashes from the files on disk, with no model call and no retrieval: exit 1 if anything differs. `--rescore`
does the same and writes the file back (the header then describes the rescore; `segments` keep the judging runs),
for a file judged by an earlier version of this module.

Where an answers file already stores verdicts of its own judges (jury_v2.json: qwen2.5:7b, which also generated the
answers, and llama3.1:8b), the summary gives Cohen's kappa between this judge and each of them over the answers both
were asked about (`agreement_with_stored_judges`).

The judge is a callable prompt -> reply. The command builds it on Ollama's /api/generate with the request every
committed judge call sends (app/eval/judge.py ollama_payload: temperature 0, seed 0, no `think` key, the server's
context length); run() takes any other callable (tests; another backend) with the block that describes it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from app.agent.generate import NOT_IN_SOURCES, parse_citations, strip_citations
from app.eval.corpus import load_manifest
from app.eval.gold import GoldQuestion, load_gold
from app.eval.harness import project_root
from app.eval.judge import LlmJudge, correctness_prompt, exact_match_label, label_to_score, ollama_payload, pairwise_ab_ba
from app.eval.jury import KAPPA_GATE, cohens_kappa
from app.eval.llm_eval import (
    ABSTENTION_RULE,
    CALIBRATION_CONDITIONS,
    GATED_LABEL,
    OLLAMA_REQUEST,
    REFERENCE_LABELS,
    _confusion,
    _counts,
    abstention_only,
    kappa_interval,
)
from app.eval.provenance import (
    command_line,
    device_info,
    git_sha,
    git_state,
    ollama_base_url,
    ollama_loaded,
    ollama_model_provenance,
    ollama_tag_entry,
    ollama_tags,
    ollama_version,
    run_provenance,
    sanitize_path,
    sha256_file,
)
from app.eval.stats import DEFAULT_N_BOOT, DEFAULT_SEED, RNG_NAME, mean_ci, resample_indices

CALIBRATION_FILE = "bench/results/jury_v2.json"  # the committed calibration answers every judge is gated on
CHECKPOINT = 25  # verdicts between two writes of the file inside a phase
MAX_ERRORS_IN_A_ROW = 5  # calls failing in a row before the run stops as interrupted (the judge is gone, not wrong)
# The warm-up prompt: no calibration item and no answer of any run; its reply only shows that the model answers.
WARMUP_PROMPT = 'Reply JSON {"label": "correct"}.'
LOAD_TIMEOUT_S = 1800.0  # a 65 GB model read from a network filesystem
PAIR_QUESTION = "Which is better grounded and complete? Reply A, B, or TIE."  # app/eval/judge.py pairwise_ab_ba
FAILED_KINDS = {"uncited_answer": "a substantive answer with no citation: failed loud, not judged",
                "no_answer": "an empty answer (nothing was delivered): counted, not judged"}
NOT_COLLECTED = {"composite": "the M4 composite label (faithfulness, relevancy and groundedness with the retrieved context) is "
                              "never gated or published and needs the retrieved chunks: not collected by a judge-only run"}
# What a judge file's numbers do not show, written into every file (run, rescore).
DOES_NOT_SHOW = (
    "that the judge is right on every kind of answer: the gate's reference is an exact-match label on single-value answers "
    "whose wrong class is a planted value of the same field, so a pass shows that the judge compares a single-value answer "
    "with the gold, not that it judges multi-part, paraphrased or computed answers correctly",
    "an unbiased score where the judge's own weights generated the answers: such an answer set is marked `self_judged` "
    "(self-preference risk)",
    "the truth where two judges agree: `agreement_with_stored_judges` is agreement between judges",
    "grounding in the retrieved context: the correctness prompt shows the question, the gold answer and the answer only, "
    "and the M4 composite label is not collected",
    "which answer is better: the AB/BA pairs compare an answer with its row's extractive answer in both orders to measure "
    "position bias, not quality",
    "variance between runs or machines: one verdict per item at temperature 0 and seed 0; `segments` name the commit, GPU "
    "and time of each run that added verdicts, and a model whose prompt template carries the date may reply otherwise on "
    "another day",
    "the judge's reasoning: a verdict stores the reply and the length of the reasoning the server returned, not the reasoning",
)


class CouldNotLoad(RuntimeError):
    """The judge model did not load, is not wholly on the GPU, or did not answer the warm-up prompt."""


class JudgeUnavailable(RuntimeError):
    """MAX_ERRORS_IN_A_ROW calls failed in a row: the run stops and keeps what it has."""


# ---- what the judge is shown: the calibration items and the answer sets ----------------------------------------

def _items_sha256(items: Sequence[dict]) -> str:
    """sha256 of the items as the judge sees them: id, condition, question, gold and the answer without its citation tags."""
    payload = [[r["id"], r["condition"], r["question"], r["gold"], r["answer"]] for r in items]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_calibration(root: Path, rel: str = CALIBRATION_FILE) -> dict[str, Any]:
    """The calibration answers of an answer-evaluation file as a judge is shown them, with today's reference label.

    A planted context where nothing was replaced is not part of the reference (llm_eval), so it is not an item. Each
    label and abstention flag is recomputed from the stored answer and must equal the stored one: the reference a
    judge is measured against is then the committed one and the one this code computes."""
    blob = json.loads((root / rel).read_text())
    cal = blob.get("calibration") or {}
    if not cal.get("rows"):
        raise SystemExit(f"{rel} has no calibration set")
    items = []
    for row in cal["rows"]:
        if not row.get("probe"):
            continue
        text = strip_citations(row["answer"])
        label = exact_match_label(text, row["gold"], bounded=True)
        pure = NOT_IN_SOURCES in row["answer"] and abstention_only(text)
        if label != row["checks"]["exact_match_label"] or pure != bool(row["checks"]["abstention_only"]):
            raise SystemExit(f"{rel}: {row['id']} ({row['condition']}): today's reference label ({label}, abstention only {pure}) "
                             "differs from the stored one; the calibration set is not the committed one")
        items.append({"id": row["id"], "condition": row["condition"], "question": row["question"], "gold": row["gold"],
                      "answer": text, "reference": label, "abstention_only": pure})
    rs = blob.get("rescore") or {}
    return {"file": rel, "file_sha256": sha256_file(root / rel), "rows_git_sha": rs.get("rows_git_sha") or blob.get("git_sha"),
            "generator": cal.get("generator"), "conditions": cal.get("conditions"), "planted_value_rule": cal.get("planted_value_rule"),
            "n": len(items), "items_sha256": _items_sha256(items), "items": items}


def _rel(path: Path, root: Path) -> str:
    return sanitize_path(str(Path(path).resolve()), root=root)


def _gold_row(row: dict, gold: dict[str, GoldQuestion], where: str) -> str | None:
    """The category of a stored answer; an id of the gold set must carry that question and that gold answer."""
    q = gold.get(row["id"])
    if q is None:
        return row.get("category")
    if row["question"] != q.question or row["gold"] != q.answer:
        raise SystemExit(f"{where}: {row['id']} does not carry the question and gold answer of bench/gold/questions.jsonl")
    if row.get("category") not in (None, q.category):
        raise SystemExit(f"{where}: {row['id']} is {q.category} in the gold set, {row['category']} in the file")
    return q.category


def _answer_row(*, qid: str, category: str | None, question: str, gold: str, answer: str, cited_paths: Sequence[str],
                extractive: str | None, failed: str | None, corpus: frozenset[str], retrieved_paths: Sequence[str] | None = None) -> dict[str, Any]:
    """A stored answer as the judge is shown it (citation tags stripped), with the checks no model is needed for:
    whether every cited path is a file of the fixture and, where the row stores what was retrieved, one of those
    paths (None for an answer that cites nothing)."""
    judged = strip_citations(answer or "")
    cited = list(cited_paths)
    row: dict[str, Any] = {"id": qid, "category": category, "question": question, "gold": gold, "answer": judged,
                           "cited_paths": cited, "cited_in_corpus": all(p in corpus for p in cited) if cited else None,
                           "abstention_only": NOT_IN_SOURCES in (answer or "") and abstention_only(judged)}
    if retrieved_paths is not None:
        row["cited_in_retrieved"] = all(p in set(retrieved_paths) for p in cited) if cited else None
    if category == "simple_factual":
        row["exact_match_label"] = exact_match_label(judged, gold, bounded=True)
    if extractive is not None:
        row["extractive"] = extractive
    if failed:
        row["failed"] = failed
    return row


def load_answers(path: Path, root: Path, gold: dict[str, GoldQuestion]) -> dict[str, Any]:
    """One answer set: an answer-evaluation file (its rows) or a JSONL of stored answers (the module docstring)."""
    rel = _rel(path, root)
    corpus = frozenset(entry["path"] for entry in load_manifest(root / "bench" / "corpus" / "MANIFEST.json")["files"])
    out: dict[str, Any] = {"file": rel, "sha256": sha256_file(path)}
    rows: list[dict[str, Any]] = []
    if path.suffix == ".jsonl":
        out["format"] = "stored answers, one JSON object per line"
        for n, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            missing = [k for k in ("id", "question", "gold", "answer", "cited_paths") if k not in obj]
            if missing or not isinstance(obj["cited_paths"], list):
                raise SystemExit(f"{rel}:{n}: a stored answer needs id, question, gold, answer and cited_paths (a list); missing or malformed: {missing or ['cited_paths']}")
            answer = obj["answer"] or ""
            failed = "no_answer" if not answer.strip() else ("uncited_answer" if not obj["cited_paths"] and NOT_IN_SOURCES not in answer else None)
            rows.append(_answer_row(qid=str(obj["id"]), category=_gold_row(obj, gold, f"{rel}:{n}"), question=obj["question"], gold=obj["gold"],
                                    answer=answer, cited_paths=[str(p) for p in obj["cited_paths"]], extractive=obj.get("extractive"), failed=failed,
                                    corpus=corpus, retrieved_paths=obj.get("retrieved_paths")))
    else:
        blob = json.loads(path.read_text())
        if "rows" not in blob or "models" not in blob:
            raise SystemExit(f"{rel} is neither a JSONL of stored answers nor an answer-evaluation file")
        rs = blob.get("rescore") or {}
        out.update({"format": "answer-evaluation file (app/eval/llm_eval.py rows)", "generator": blob["models"].get("generator"),
                    "rows_git_sha": rs.get("rows_git_sha") or blob.get("git_sha")})
        for stored in blob["rows"]:
            cited = list(dict.fromkeys(c.path for c in parse_citations(stored["answer"])))
            row = _answer_row(qid=stored["id"], category=_gold_row(stored, gold, rel), question=stored["question"], gold=stored["gold"],
                              answer=stored["answer"], cited_paths=cited, extractive=stored.get("extractive"), failed=stored.get("failed"),
                              corpus=corpus)
            checks = stored.get("checks") or {}
            if "cited_in_retrieved" in checks:  # the file's own check, made against the retrieved chunks and their byte ranges
                row["cited_in_retrieved"] = checks["cited_in_retrieved"]
            if "abstention_only" in checks and bool(checks["abstention_only"]) != row["abstention_only"]:
                raise SystemExit(f"{rel}: {stored['id']}: the abstention rule would label other rows than the file's checks say")
            if "exact_match_label" in checks and checks["exact_match_label"] != row.get("exact_match_label"):
                raise SystemExit(f"{rel}: {stored['id']}: today's exact-match label differs from the stored one")
            earlier = {name: v["label"] for name, v in (stored.get("correctness") or {}).items() if v.get("by") == "judge"}
            if earlier:  # the verdicts of the file's own judges on this answer, to compare this judge with
                row["stored_verdicts"] = earlier
            rows.append(row)
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{rel}: an id appears more than once ({sorted(k for k, v in Counter(ids).items() if v > 1)[:5]})")
    out["n"] = len(rows)
    out["rows"] = rows
    return out


# ---- the judge -------------------------------------------------------------------------------------------------

class OllamaJudge:
    """prompt -> reply through Ollama's /api/generate with the committed request (judge.ollama_payload); `last` holds
    what the server reported for the last call: tokens generated (reasoning included), the length of the reasoning
    the server returned beside the reply, and the error when the call failed."""

    def __init__(self, model: str):
        self.model = model
        self.last: dict[str, Any] = {}

    def __call__(self, prompt: str) -> str:
        payload, error = ollama_payload(prompt, model=self.model)
        self.last = {"eval_count": payload.get("eval_count"), "thinking_chars": len(payload.get("thinking") or "")}
        if error:
            self.last = {"error": error}
        return payload.get("response") or payload.get("text") or ""


class _Asker:
    """Calls the judge, times it, and counts failures: MAX_ERRORS_IN_A_ROW in a row raise JudgeUnavailable."""

    def __init__(self, complete: Callable[[str], str]):
        self.complete = complete
        self.calls = 0
        self.errors_in_a_row = 0

    def __call__(self, prompt: str) -> tuple[str, dict[str, Any]]:
        t0 = time.perf_counter()
        try:
            reply = self.complete(prompt) or ""
            meta = dict(getattr(self.complete, "last", None) or {})
        except Exception as e:  # noqa: BLE001 - a judge that raises is counted like one that does not answer
            reply, meta = "", {"error": f"{type(e).__name__}: {e}"}
        self.calls += 1
        meta["seconds"] = round(time.perf_counter() - t0, 3)
        if meta.get("error"):
            self.errors_in_a_row += 1
            if self.errors_in_a_row >= MAX_ERRORS_IN_A_ROW:
                raise JudgeUnavailable(f"{self.errors_in_a_row} judge calls failed in a row; the last: {meta['error']}")
        else:
            self.errors_in_a_row = 0
        return reply, meta


def _verdict(asker: _Asker, question: str, answer: str, gold: str) -> dict[str, Any]:
    """The correctness verdict through LlmJudge.judge_correctness, the path llm_eval's judges take."""
    meta: dict[str, Any] = {}

    def complete(prompt: str) -> str:
        reply, got = asker(prompt)
        meta.update(got)
        return reply

    return {**LlmJudge(complete=complete).judge_correctness(question, answer, gold), "by": "judge", **meta}


def _pair(asker: _Asker, question: str, answer: str, extractive: str) -> dict[str, Any]:
    replies: list[str] = []
    errors: list[str] = []
    t0 = time.perf_counter()

    def complete(prompt: str) -> str:
        reply, got = asker(prompt)
        replies.append(reply)
        if got.get("error"):
            errors.append(got["error"])
        return reply

    out = pairwise_ab_ba(complete, question, answer, extractive) | {"raw": replies, "seconds": round(time.perf_counter() - t0, 3)}
    return out | ({"error": errors[0]} if errors else {})


def _done(record: dict | None) -> bool:
    """A stored verdict or pair that a resumed run keeps: anything but a failed call."""
    return bool(record) and not record.get("error")


def ollama_judge_block(model: str) -> dict[str, Any]:
    """Load `model` on the Ollama server at OLLAMA_BASE_URL (the address every judge call uses) and describe how it is
    served; CouldNotLoad unless the server lists the tag, loads it wholly on the GPU and returns a reply to the warm-up
    prompt. A tag without a digest stops the run, as in llm_eval."""
    tags = ollama_tags()
    entry = ollama_tag_entry(model, tags)
    if not entry:
        raise CouldNotLoad(f"the Ollama server does not list {model}; it has {sorted(tags)}")
    if not entry.get("digest"):
        raise SystemExit(f"Ollama lists no digest for {model}; refusing to record weights by tag alone")
    block = ollama_model_provenance(model, "judge", tags) | {"server_version": ollama_version(), "request": OLLAMA_REQUEST}
    t0 = time.perf_counter()
    try:  # a request without a prompt loads the model and generates nothing
        body = json.dumps({"model": model, "stream": False, "keep_alive": "30m"}).encode()
        req = urllib.request.Request(ollama_base_url() + "/api/generate", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=LOAD_TIMEOUT_S) as resp:
            resp.read()
    except Exception as e:  # noqa: BLE001 - the reason is the result
        raise CouldNotLoad(f"{model} did not load after {round(time.perf_counter() - t0, 1)} s: {type(e).__name__}: {e}") from e
    loaded = ollama_loaded(model)
    block["loaded"] = loaded | {"load_seconds": round(time.perf_counter() - t0, 1)}
    if not loaded or not loaded.get("size_bytes") or loaded.get("size_vram_bytes") != loaded.get("size_bytes"):
        raise CouldNotLoad(f"{model} is not wholly on the GPU: the server reports {loaded or 'no loaded model'}")
    payload, error = ollama_payload(WARMUP_PROMPT, model=model)
    block["warmup"] = {"prompt": WARMUP_PROMPT, "reply": payload.get("response") or "", "eval_count": payload.get("eval_count"),
                       "thinking_chars": len(payload.get("thinking") or ""), "error": error}
    if error:
        raise CouldNotLoad(f"{model} loaded but the warm-up call failed: {error}")
    return block


# ---- the gate and the summaries (pure: also what --check recomputes) -------------------------------------------

def _health(records: Sequence[dict], replies: Callable[[dict], Sequence[str]]) -> dict[str, int]:
    texts = [t for r in records for t in replies(r)]
    return {"n": len(texts), "empty": sum(1 for t in texts if not (t or "").strip()), "errors": sum(1 for r in records if r.get("error"))}


def gate_summary(rows: Sequence[dict]) -> dict[str, Any]:
    """Cohen's kappa of the judge's correctness verdicts against the bounded exact-match labels of the calibration
    answers, as llm_eval.summarize computes it: rule-labelled abstentions left out and counted, the interval over
    questions. `passes` needs every reference answer labelled, a reference with two classes, and kappa >= KAPPA_GATE."""
    ref_rows = [r for r in rows if not r["abstention_only"]]
    labelled = [r for r in ref_rows if r.get("verdict")]
    pairs = [(r["reference"], r["verdict"]["label"]) for r in labelled]
    ref_labels = [r["reference"] for r in ref_rows]
    single_class = bool(ref_labels) and len(set(ref_labels)) == 1
    evaluable = bool(ref_labels) and not single_class
    k = cohens_kappa([a for a, _ in pairs], [b for _, b in pairs])
    asked = [r["verdict"] for r in labelled if r["verdict"].get("by") == "judge"]
    complete = len(labelled) == len(ref_rows)
    return {
        "threshold": KAPPA_GATE,
        "kappa": round(k, 4),
        "interval": kappa_interval(pairs, [r["id"] for r in labelled]),
        "n": len(pairs),
        "complete": complete,
        "passes": complete and evaluable and bool(pairs) and k >= KAPPA_GATE,
        "label_counts": _counts([b for _, b in pairs]),
        "confusion_reference_by_judge": _confusion(pairs),
        "reference": {"labels": REFERENCE_LABELS, "source": "calibration", "n": len(ref_rows), "label_counts": _counts(ref_labels),
                      "by_condition": {c: _counts([r["reference"] for r in ref_rows if r["condition"] == c]) for c in CALIBRATION_CONDITIONS},
                      "left_out_abstention_only": len(rows) - len(ref_rows), "single_class": single_class, "gate_evaluable": evaluable},
        "replies": _health(asked, lambda v: [v.get("raw")]) | {"unparsed": sum(1 for v in asked if not v.get("parsed"))},
        "interval_method": ("percentile bootstrap over the calibration questions: each resample draws questions with replacement and "
                            "keeps a drawn question's reference answers together (its retrieved and its planted answer)"),
    }


def _agreement(labelled: Sequence[dict]) -> dict[str, Any]:
    """Cohen's kappa between this judge's verdicts and those of each judge the answers file already stores, over the
    answers both were asked about (rule-labelled abstentions agree by construction and are left out). Agreement
    between two judges is not agreement with the truth."""
    out: dict[str, Any] = {}
    for name in sorted({n for r in labelled for n in (r.get("stored_verdicts") or {})}):
        both = [(r["stored_verdicts"][name], r["correctness"]["label"]) for r in labelled
                if name in (r.get("stored_verdicts") or {}) and r["correctness"].get("by") == "judge"]
        out[name] = {"kappa": round(cohens_kappa([a for a, _ in both], [b for _, b in both]), 4), "agree": sum(1 for a, b in both if a == b),
                     "n": len(both), "confusion_stored_by_this_judge": _confusion(both)}
    return out


def _rate(flags: Sequence[bool | None]) -> dict[str, Any]:
    """The share of true flags among those that are not None (an answer that cites nothing has none)."""
    known = [bool(f) for f in flags if f is not None]
    return {"rate": round(sum(known) / len(known), 4) if known else None, "n": len(known)}


def _mean(labels: Sequence[str]) -> dict[str, Any]:
    """Mean correctness (correct 1, partial 0.5, wrong 0) with a percentile bootstrap interval over the answers."""
    scores = [label_to_score(lab) for lab in labels]
    if not scores:
        return {"mean": None, "interval": None, "n": 0}
    mean, lo, hi = mean_ci(scores, resample_indices(len(scores), DEFAULT_N_BOOT, DEFAULT_SEED))
    return {"mean": round(mean, 4), "interval": {"lo": round(lo, 4), "hi": round(hi, 4)}, "n": len(scores)}


def set_summary(rows: Sequence[dict], judge_model: str, generator: dict | None, passes: bool) -> dict[str, Any]:
    """One answer set under one judge: what was judged, the verdict counts, mean correctness with its interval, the
    agreement with exact match on the set's own simple-factual answers, reply health and the AB/BA pairs."""
    judged = [r for r in rows if not r.get("failed")]
    labelled = [r for r in judged if r.get("correctness")]
    asked = [r["correctness"] for r in labelled if r["correctness"].get("by") == "judge"]
    sf = [r for r in judged if r.get("category") == "simple_factual"]
    own = [(r["exact_match_label"], r["correctness"]["label"]) for r in sf if (r.get("correctness") or {}).get("by") == "judge"]
    pairs = [r["pairwise"] for r in judged if r.get("pairwise")]
    with_second = [r for r in judged if r.get("extractive") is not None]
    exact = [r["exact_match_label"] == "correct" for r in sf]
    out: dict[str, Any] = {
        "n": len(rows),
        "n_judged": len(judged),
        "failed": {kind: sum(1 for r in rows if r.get("failed") == kind) for kind in FAILED_KINDS},
        "abstention_only": {"count": sum(1 for r in judged if r["abstention_only"]), "labelled": "wrong, by rule",
                            "ids": [r["id"] for r in judged if r["abstention_only"] and r["gold"].strip()]},
        "exact_match_simple_factual": {"rate": round(sum(exact) / len(exact), 4) if exact else None, "n": len(exact)},
        "cited_in_corpus": _rate([r["cited_in_corpus"] for r in judged]),
        "cited_in_retrieved": _rate([r.get("cited_in_retrieved") for r in judged]),
        "complete": len(labelled) == len(judged),
        "label_counts": _counts([r["correctness"]["label"] for r in labelled]),
        "mean_correctness": _mean([r["correctness"]["label"] for r in labelled]),
        "mean_correctness_by_category": {cat: _mean([r["correctness"]["label"] for r in labelled if r.get("category") == cat])
                                         for cat in sorted({str(r.get("category")) for r in labelled}) if cat != "None"},
        "simple_factual_answers": {"agree": sum(1 for a, b in own if a == b), "n": len(own), "confusion_reference_by_judge": _confusion(own)},
        "agreement_with_stored_judges": _agreement(labelled),
        "replies": {"correctness": _health(asked, lambda v: [v.get("raw")]) | {"unparsed": sum(1 for v in asked if not v.get("parsed"))},
                    "pairwise": _health(pairs, lambda p: p.get("raw") or [])},
        "pairwise_llm_vs_extractive": ({
            "p_llm_better_mean": round(sum(p["p_a"] for p in pairs) / len(pairs), 4),
            "position_gap_mean": round(sum(p["position_gap"] for p in pairs) / len(pairs), 4),
            "position_flips": sum(1 for p in pairs if p["p_a_ab"] != p["p_a_ba"]),
            "n": len(pairs), "complete": len(pairs) == len(with_second),
        } if pairs else None),
        "pairs_not_collected": None if with_second else "the file stores no second (extractive) answer",
        "self_judged": bool(generator) and generator.get("model") == judge_model,
        "publishable": bool(passes) and len(labelled) == len(judged) and bool(labelled),
    }
    return out


def _verdict_text(blob: dict) -> str:
    judge, gate = blob["judge"]["model"], blob.get("gate")
    if blob["status"] == "could not load":
        return f"{judge} could not load: no item was judged, and this is not a gate result"
    if not gate or not gate["complete"]:
        return f"{judge}: the calibration set is not fully judged ({(gate or {}).get('n', 0)} of {blob['calibration']['n']}); nothing may be published"
    if not gate["reference"]["gate_evaluable"]:
        return f"gate not evaluable for {judge}: the calibration reference is one class or empty; nothing may be published"
    if gate["passes"]:
        return f"kappa >= {KAPPA_GATE} for {judge} on the calibration set: its correctness scores of the answer sets it finished may be published"
    return f"{judge} did not reach kappa >= {KAPPA_GATE} on the calibration set; its scores stay in this file and off the README"


def summarize(blob: dict) -> dict:
    """Fill `gate`, each answer set's `summary` and `verdict` from the stored verdicts (run() and --check)."""
    if blob["status"] != "could not load":
        blob["gate"] = gate_summary(blob["calibration"]["rows"])
        for answers in blob["answer_sets"]:
            answers["summary"] = set_summary(answers["rows"], blob["judge"]["model"], answers.get("generator"), blob["gate"]["passes"])
    blob["verdict"] = _verdict_text(blob)
    return blob


# ---- self-preference or leniency (M42): a judge on its own generator's answers and on other generators' --------

SELF_PREFERENCE_METHOD = (
    "for each independent judge I that passed its gate: per answer judged by both (verdicts by the judge in both files; "
    "rule-labelled and failed rows left out), gap = score of the judge under test - score of I (correct 1, partial 0.5, "
    "wrong 0); own = the answer set the judge's own weights generated, others = every other answer set it judged, pooled; "
    "per gold question, own_q = the gap on its own answer and other_q = the mean gap over the other sets' answers, and "
    "d_q = own_q - other_q where a question has both; means with percentile bootstrap 95% intervals over questions. "
    "Reading: d above 0 = self-preference; d covering 0 with the others' gap above 0 = leniency; d covering 0 otherwise = "
    "the intervals cannot tell; d below 0 = no self-preference")


def _gap_stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "interval": None, "n": 0}
    mean, lo, hi = mean_ci(values, resample_indices(len(values), DEFAULT_N_BOOT, DEFAULT_SEED))
    # + 0.0: a bound below zero by less than the rounding prints as 0.0, not -0.0 (it still does not lie above zero)
    return {"mean": round(mean, 4) + 0.0, "interval": {"lo": round(lo, 4) + 0.0, "hi": round(hi, 4) + 0.0}, "n": len(values)}


def _judged_scores(answer_set: dict) -> dict[str, float]:
    """Row id -> score of the rows a judge itself labelled (not by rule, not failed)."""
    return {r["id"]: label_to_score(r["correctness"]["label"]) for r in answer_set.get("rows") or []
            if not r.get("failed") and (r.get("correctness") or {}).get("by") == "judge"}


def self_preference(judge_file: dict, independent: Sequence[dict]) -> dict[str, Any] | None:
    """Does the judge of `judge_file` favour the answers its own weights generated, or is it lenient with any
    generator's? Compared, per independent judge file that passed its gate, on the answers both judged
    (SELF_PREFERENCE_METHOD). None when the judge file has no complete answer set of its own generator or none of
    another generator. The judge's own gate is part of the result: a judge that failed it in this file answers nothing."""
    me = (judge_file.get("judge") or {}).get("model")
    sets = [s for s in judge_file.get("answer_sets") or [] if s.get("summary")]
    own = [s for s in sets if s["summary"].get("self_judged")]
    others = [s for s in sets if not s["summary"].get("self_judged") and (s.get("generator") or {}).get("model")]
    if not own or not others:
        return None
    gate = judge_file.get("gate") or {}
    out: dict[str, Any] = {"judge": me, "gate_passes": bool(gate.get("passes")), "own_files": [s["file"] for s in own],
                           "other_files": [s["file"] for s in others], "method": SELF_PREFERENCE_METHOD, "by_independent_judge": {}}
    for ref in independent:
        if not ref or ref.get("status") != "complete" or not (ref.get("gate") or {}).get("passes"):
            continue
        name = ref["judge"]["model"]
        if name == me:
            continue
        theirs = {s["file"]: _judged_scores(s) for s in ref.get("answer_sets") or []}

        def gaps(sets_: Sequence[dict]) -> dict[str, list[float]]:
            per_q: dict[str, list[float]] = {}
            for s in sets_:
                mine, ref_scores = _judged_scores(s), theirs.get(s["file"]) or {}
                for qid in mine.keys() & ref_scores.keys():
                    per_q.setdefault(qid, []).append(mine[qid] - ref_scores[qid])
            return per_q

        own_q = {q: sum(v) / len(v) for q, v in gaps(own).items()}
        other_q = {q: sum(v) / len(v) for q, v in gaps(others).items()}
        if not own_q or not other_q:
            continue
        both = sorted(own_q.keys() & other_q.keys())
        own_gap, other_gap = _gap_stats([own_q[q] for q in sorted(own_q)]), _gap_stats([other_q[q] for q in sorted(other_q)])
        d = _gap_stats([own_q[q] - other_q[q] for q in both])
        lo, hi = (d["interval"] or {}).get("lo"), (d["interval"] or {}).get("hi")
        others_above = (other_gap["interval"] or {}).get("lo", 0) > 0
        if d["n"] and lo > 0:
            reading = "self-preference"
        elif d["n"] and hi < 0:
            reading = "no self-preference"
        elif others_above:
            reading = "leniency"
        else:
            reading = "cannot tell"
        out["by_independent_judge"][name] = {"own_gap": own_gap, "others_gap": other_gap, "difference": d, "reading": reading,
                                             "also_lenient_with_others": reading == "self-preference" and others_above}
    return out if out["by_independent_judge"] else None


# ---- the run -----------------------------------------------------------------------------------------------------

def _protocol() -> dict[str, Any]:
    return {"gated_label": GATED_LABEL, "correctness_prompt": correctness_prompt("{question}", "{answer}", "{gold}"),
            "reference": REFERENCE_LABELS, "abstention_rule": ABSTENTION_RULE, "failed": FAILED_KINDS,
            "pairwise": f"AB and BA, the answer against the row's extractive answer: {PAIR_QUESTION}", "not_collected": NOT_COLLECTED,
            "interval": {"rng": RNG_NAME, "seed": DEFAULT_SEED, "n_boot": DEFAULT_N_BOOT, "level": 0.95, "gate_reads": "the point estimate"}}


def _write(out: Path, blob: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(blob, indent=2) + "\n")
    os.replace(tmp, out)


def _same_judge(a: dict, b: dict) -> bool:
    keys = ("model", "revision", "server_version", "backend")
    return all(a.get(k) == b.get(k) for k in keys) and (a.get("loaded") or {}).get("context_length") == (b.get("loaded") or {}).get("context_length")


def run(*, judge_model: str, answers: Sequence[Path], out: Path, root: Path | None = None, calibration_file: str = CALIBRATION_FILE,
        complete: Callable[[str], str] | None = None, judge_block: dict | None = None, resume: bool = False, pairs: bool = True,
        limit: int | None = None) -> dict[str, Any]:
    """Gate `judge_model` on the calibration set, then judge each file of `answers`; write `out` as it goes and return
    the blob. `complete` (prompt -> reply) with `judge_block` replaces Ollama (tests; another backend). `limit` judges
    the first items of each phase only (a smoke run): such a file is never complete, so nothing in it is publishable."""
    root = root or project_root()
    started = git_state(root)  # before the run writes anything (run_provenance)
    gold = {q.id: q for q in load_gold(root / "bench" / "gold" / "questions.jsonl")}
    cal = load_calibration(root, calibration_file)
    sets = [load_answers(Path(p), root, gold) for p in answers]
    prior = json.loads(out.read_text()) if resume and out.exists() else None
    t_start = time.perf_counter()

    blob: dict[str, Any] = {
        "git_sha": git_sha(root),
        **run_provenance(root=root, started=started),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command_line(),
        "device": device_info("auto"),
        "what": "one judge over stored answers, gated on stored calibration answers in this file (app/eval/judge_stored.py)",
        "does_not_show": list(DOES_NOT_SHOW),
        "kappa_gate": KAPPA_GATE,
        "protocol": _protocol(),
        "judge": judge_block or {"role": "judge", "model": judge_model, "backend": "callable"},
        "status": "running",
        "limit": limit,
        "segments": list((prior or {}).get("segments") or []),
        "calibration": {k: v for k, v in cal.items() if k != "items"} | {"rows": []},
        "gate": None,
        "answer_sets": [],
        "verdict": None,
    }
    segment = {"git_head": blob["git_head"], "working_tree_dirty": blob["working_tree_dirty"], "timestamp": blob["timestamp"],
               "command": blob["command"], "device_name": blob["device"].get("device_name"), "resumed": prior is not None}
    blob["segments"].append(segment)
    if complete is None:
        try:
            blob["judge"] = ollama_judge_block(judge_model)
        except CouldNotLoad as e:
            if prior is not None:  # a file with verdicts is not replaced by a failed start
                raise SystemExit(f"could not load: {e}; {out} is left as it was")
            blob.update(status="could not load", could_not_load=str(e))
            segment["calls"] = 0
            _write(out, summarize(blob))
            return blob
        complete = OllamaJudge(judge_model)
    if prior is not None:
        if prior.get("status") == "could not load":
            prior = None  # nothing was judged: start over in the same file
        elif not _same_judge(prior["judge"], blob["judge"]) or prior["calibration"]["items_sha256"] != cal["items_sha256"] or prior["protocol"] != blob["protocol"]:
            raise SystemExit(f"--resume: {out} was written by another judge (model digest, server version or context length), another "
                             "calibration set or another protocol; write a new file")
        elif prior.get("limit") != limit:
            raise SystemExit(f"--resume: {out} was written with --limit {prior.get('limit')}; write a new file")
    asker = _Asker(complete)
    since = 0

    def save() -> None:  # every write carries the calls and the time so far, so a run cut short still says them
        segment.update(calls=asker.calls, wall_s=round(time.perf_counter() - t_start, 1))
        blob["wall_s"] = round(sum(s.get("wall_s", 0) for s in blob["segments"]), 1)
        _write(out, summarize(blob))

    def checkpoint(force: bool = False) -> None:
        nonlocal since
        since += 1
        if force or since >= CHECKPOINT:
            since = 0
            save()

    def finish(status: str) -> dict:
        blob["status"] = status
        save()
        return blob

    kept = {(r["id"], r["condition"]): r.get("verdict") for r in (prior or {}).get("calibration", {}).get("rows", [])}
    prior_sets = {s["file"]: s for s in (prior or {}).get("answer_sets", [])}
    for s in sets:
        old = prior_sets.pop(s["file"], None)
        if old is not None and old["sha256"] != s["sha256"]:
            raise SystemExit(f"--resume: {s['file']} changed since {out} judged it; write a new file")
        old_rows = {r["id"]: r for r in (old or {}).get("rows", [])}
        for row in s["rows"]:
            for key in ("correctness", "pairwise"):
                if _done(old_rows.get(row["id"], {}).get(key)):
                    row[key] = old_rows[row["id"]][key]
        blob["answer_sets"].append(s)
    blob["answer_sets"] += list(prior_sets.values())  # sets judged before and not named now are kept as they are
    try:
        # Phase 1: the gate.
        blob["calibration"]["rows"] = [dict(item) for item in cal["items"]]  # every item; a verdict as each is judged
        for row in blob["calibration"]["rows"][:limit] if limit else blob["calibration"]["rows"]:
            if _done(kept.get((row["id"], row["condition"]))):
                row["verdict"] = kept[(row["id"], row["condition"])]
            elif row["abstention_only"] and row["gold"].strip():
                row["verdict"] = {"label": "wrong", "by": "rule"}
            else:
                row["verdict"] = _verdict(asker, row["question"], row["answer"], row["gold"])
                checkpoint()
        checkpoint(force=True)
        # Phase 2: the correctness verdict of every answer; phase 3: the AB/BA pairs.
        for s in sets:
            todo = [r for r in s["rows"] if not r.get("failed")]
            for row in todo[:limit] if limit else todo:
                if _done(row.get("correctness")):
                    continue
                if row["abstention_only"] and row["gold"].strip():
                    row["correctness"] = {"label": "wrong", "by": "rule"}
                    continue
                row["correctness"] = _verdict(asker, row["question"], row["answer"], row["gold"])
                checkpoint()
            checkpoint(force=True)
        for s in sets if pairs else []:
            todo = [r for r in s["rows"] if not r.get("failed") and r.get("extractive") is not None]
            for row in todo[:limit] if limit else todo:
                if _done(row.get("pairwise")):
                    continue
                row["pairwise"] = _pair(asker, row["question"], row["answer"], row["extractive"])
                checkpoint()
            checkpoint(force=True)
    except JudgeUnavailable as e:
        blob["interrupted"] = str(e)
        return finish("interrupted")
    blob.pop("interrupted", None)
    return finish("complete" if not limit else "partial (limit)")


# ---- re-reading a file without a model -------------------------------------------------------------------------

def check(path: Path, root: Path | None = None) -> list[str]:
    """What differs between a judge file and what its stored verdicts, the committed calibration set and the answer
    files on disk give today ([] when nothing does). No model call, no retrieval."""
    root = root or project_root()
    blob = json.loads(path.read_text())
    problems: list[str] = []
    cal = load_calibration(root, blob["calibration"]["file"])
    if cal["items_sha256"] != blob["calibration"]["items_sha256"]:
        problems.append(f"calibration items: {blob['calibration']['file']} hashes to {cal['items_sha256']}, the file says {blob['calibration']['items_sha256']}")
    items = {(r["id"], r["condition"]): r for r in cal["items"]}
    if blob["status"] != "could not load" and len(blob["calibration"]["rows"]) != cal["n"]:
        problems.append(f"calibration rows: {len(blob['calibration']['rows'])} in the file, {cal['n']} items in {blob['calibration']['file']}")
    for row in blob["calibration"]["rows"]:
        item = items.get((row["id"], row["condition"]))
        if item is None or any(row[k] != item[k] for k in ("question", "gold", "answer", "reference", "abstention_only")):
            problems.append(f"calibration row {row['id']} ({row['condition']}) is not the committed item")
    gold = {q.id: q for q in load_gold(root / "bench" / "gold" / "questions.jsonl")}
    for answers in blob["answer_sets"]:
        source = root / answers["file"]
        if not source.exists():
            problems.append(f"{answers['file']}: not found, so its answers cannot be compared")
            continue
        fresh = load_answers(source, root, gold)
        if fresh["sha256"] != answers["sha256"]:
            problems.append(f"{answers['file']}: sha256 {fresh['sha256']} on disk, {answers['sha256']} in the file")
        stored = [{k: v for k, v in r.items() if k not in ("correctness", "pairwise")} for r in answers["rows"]]
        if stored != fresh["rows"]:
            problems.append(f"{answers['file']}: the rows the judge was shown differ from the file on disk")
    again = summarize(json.loads(json.dumps({**blob, "gate": None})))
    if again["gate"] != blob.get("gate"):
        problems.append("gate: the stored verdicts give another gate block than the file's")
    for old, new in zip(blob["answer_sets"], again["answer_sets"]):
        if old.get("summary") != new["summary"]:
            problems.append(f"{old['file']}: the stored verdicts give another summary than the file's")
    if again["verdict"] != blob.get("verdict"):
        problems.append("verdict: differs from what the stored verdicts give")
    return problems


def rescore(path: Path, root: Path | None = None) -> dict[str, Any]:
    """Recompute a judge file's gate and summaries from its stored verdicts, with today's code and the inputs on disk:
    the calibration items and every answers file are read again, must hash to what the file recorded, and take their
    stored verdicts by id. No model call. The header then describes the rescore; `segments` keep the judging runs."""
    root = root or project_root()
    started = git_state(root)
    blob = json.loads(path.read_text())
    if blob["status"] != "complete":
        raise SystemExit(f"{path.name} is {blob['status']}: finish it with --resume before rescoring")
    cal = load_calibration(root, blob["calibration"]["file"])
    if cal["items_sha256"] != blob["calibration"]["items_sha256"]:
        raise SystemExit(f"{blob['calibration']['file']}: the calibration items are not the ones this file was judged on")
    kept = {(r["id"], r["condition"]): r["verdict"] for r in blob["calibration"]["rows"]}
    rows = [dict(item) | {"verdict": kept[(item["id"], item["condition"])]} for item in cal["items"]]
    gold = {q.id: q for q in load_gold(root / "bench" / "gold" / "questions.jsonl")}
    sets = []
    for old in blob["answer_sets"]:
        fresh = load_answers(root / old["file"], root, gold)
        if fresh["sha256"] != old["sha256"]:
            raise SystemExit(f"{old['file']} changed since it was judged ({fresh['sha256'][:12]} on disk, {old['sha256'][:12]} in the file)")
        verdicts = {r["id"]: r for r in old["rows"]}
        for row in fresh["rows"]:
            for key in ("correctness", "pairwise"):
                if key in verdicts[row["id"]]:
                    row[key] = verdicts[row["id"]][key]
        sets.append(fresh)
    prior = blob.get("rescore") or {}
    blob["calibration"] = {k: v for k, v in cal.items() if k != "items"} | {"rows": rows}
    blob["answer_sets"] = sets
    blob["rescore"] = {"rows_git_sha": prior.get("rows_git_sha") or blob["git_sha"], "rows_git_head": prior.get("rows_git_head") or blob.get("git_head"),
                       "rows_working_tree_dirty": prior["rows_working_tree_dirty"] if "rows_working_tree_dirty" in prior else blob.get("working_tree_dirty"),
                       "rows_command": prior.get("rows_command") or blob["command"], "rows_timestamp": prior.get("rows_timestamp") or blob["timestamp"],
                       "checks_git_sha": git_sha(root), "command": command_line(), "timestamp": datetime.now(timezone.utc).isoformat(),
                       "what": "gate and summaries recomputed from the stored verdicts, the calibration items and the answers files read again; every verdict untouched"}
    blob.update(git_sha=blob["rescore"]["checks_git_sha"], command=blob["rescore"]["command"], **run_provenance(root=root, started=started))
    if "does_not_show" not in blob:  # a file judged before the field existed: it goes where run() writes it
        blob = {k: v for key, value in blob.items() for k, v in ([(key, value), ("does_not_show", None)] if key == "what" else [(key, value)])}
    blob["does_not_show"] = list(DOES_NOT_SHOW)
    return summarize(blob)


def markdown_table(blob: dict) -> str:
    judge = blob["judge"]
    lines = [f"Judge {judge['model']} ({(judge.get('revision') or 'no digest')[:12]}), status {blob['status']}; wall {blob.get('wall_s')} s."]
    gate = blob.get("gate")
    if gate:
        ci, ref = gate["interval"] or {}, gate["reference"]
        lines += ["", f"Gate on {blob['calibration']['n']} calibration answers of {blob['calibration']['file']} (items sha256 {blob['calibration']['items_sha256'][:12]}), "
                  f"reference c/p/w {ref['label_counts']['correct']}/{ref['label_counts']['partial']}/{ref['label_counts']['wrong']}:", "",
                  "| judge | kappa vs exact match | interval | n | complete | passes | verdicts c/p/w | empty / unreadable / failed calls |", "|---|---:|---|---:|:---:|:---:|---|---|",
                  f"| {judge['model']} | {gate['kappa']} | [{ci.get('lo')}, {ci.get('hi')}] | {gate['n']} | {gate['complete']} | {gate['passes']} | "
                  f"{gate['label_counts']['correct']}/{gate['label_counts']['partial']}/{gate['label_counts']['wrong']} | "
                  f"{gate['replies']['empty']} / {gate['replies']['unparsed']} / {gate['replies']['errors']} of {gate['replies']['n']} |"]
    if blob["answer_sets"] and gate:
        lines += ["", "| answers | generator | judged | failed loud | by rule | mean correctness | interval | verdicts c/p/w | simple-factual agreement | P(LLM better) / position gap | publishable |",
                  "|---|---|---:|---:|---:|---:|---|---|---|---|:---:|"]
        for answers in blob["answer_sets"]:
            s = answers["summary"]
            m, pw = s["mean_correctness"], s["pairwise_llm_vs_extractive"]
            ci = m["interval"] or {}
            lines.append(f"| {answers['file']} | {(answers.get('generator') or {}).get('model') or '—'} | {m['n']} of {s['n_judged']} | {sum(s['failed'].values())} | "
                         f"{s['abstention_only']['count']} | {m['mean']} | [{ci.get('lo')}, {ci.get('hi')}] | "
                         f"{s['label_counts']['correct']}/{s['label_counts']['partial']}/{s['label_counts']['wrong']} | "
                         f"{s['simple_factual_answers']['agree']} of {s['simple_factual_answers']['n']} | "
                         + (f"{pw['p_llm_better_mean']} / {pw['position_gap_mean']} (n {pw['n']})" if pw else "not collected") + f" | {s['publishable']} |")
    lines.append(f"\nVerdict: {blob['verdict']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    root = project_root()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judge", default=os.environ.get("JUDGE_MODEL"), help="the Ollama model that judges")
    p.add_argument("--answers", default="", help="comma-separated files: answer-evaluation files or JSONL of stored answers; none gates the judge only")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--calibration", default=CALIBRATION_FILE, help="the answer-evaluation file whose stored calibration answers are the gate's reference")
    p.add_argument("--resume", action="store_true", help="continue OUT: keep its verdicts, ask what is missing or failed")
    p.add_argument("--no-pairs", action="store_true", help="skip the AB/BA pairs")
    p.add_argument("--limit", type=int, default=None, help="judge only the first N items of each phase (a smoke run; never publishable)")
    p.add_argument("--rescore", type=Path, default=None, metavar="FILE",
                   help="recompute FILE's gate and summaries from its stored verdicts and its inputs read again (no model call) and write it back")
    p.add_argument("--check", type=Path, default=None, metavar="FILE",
                   help="recompute FILE's gate and summaries from its stored verdicts and its inputs' hashes (no model call); exit 1 if anything differs")
    args = p.parse_args(argv)
    if args.check is not None:
        target = args.check if args.check.is_absolute() else root / args.check
        problems = check(target, root)
        for line in problems:
            print(f"DIFFERS {line}")
        print(f"{_rel(target, root)}: {'OK' if not problems else f'{len(problems)} difference(s)'} (gate and summaries recomputed from the stored verdicts)")
        return 1 if problems else 0
    if args.rescore is not None:
        target = args.rescore if args.rescore.is_absolute() else root / args.rescore
        blob = rescore(target, root)
        _write(target, blob)
        print(markdown_table(blob))
        print(f"rescored {_rel(target, root)} (verdicts from {blob['rescore']['rows_git_sha']}, summaries at {blob['rescore']['checks_git_sha']})")
        return 0
    if not args.judge or args.out is None:
        p.error("--judge and --out are required (or --check FILE, or --rescore FILE)")
    files = [Path(f) if Path(f).is_absolute() else root / f for f in (x.strip() for x in args.answers.split(",")) if f]
    out = args.out if args.out.is_absolute() else root / args.out
    blob = run(judge_model=args.judge, answers=files, out=out, root=root, calibration_file=args.calibration, resume=args.resume,
               pairs=not args.no_pairs, limit=args.limit)
    print(markdown_table(blob))
    print(f"wrote {_rel(out, root)} ({blob['status']})")
    return {"complete": 0, "partial (limit)": 0, "could not load": 3, "interrupted": 4}.get(blob["status"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
