"""Phase 6.1–6.3: LLM-as-judge for RAG e2e (pointwise, pairwise AB/BA, self-consistency).

Retrieval metrics stay BEIR-style. These scores are extra columns.
A HeuristicJudge keeps CI offline; plug in ollama / any callable for the real row.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

from app.retrieval.bm25 import tokenize

LABELS = ("correct", "partial", "wrong")
JudgeFn = Callable[[str], str]


def parse_label(raw: str) -> str:
    text = (raw or "").strip().lower()
    for lab in LABELS:
        if re.search(rf"\b{lab}\b", text):
            return lab
    if "yes" in text or "faithful" in text:
        return "correct"
    if "no" in text or "unfaith" in text:
        return "wrong"
    return "partial"


def parse_score(raw: str) -> float:
    """Extract a 0–1 score from JSON, SCORE: x, or a label."""
    text = (raw or "").strip()
    try:
        blob = json.loads(text)
        if isinstance(blob, dict):
            for key in ("score", "faithfulness", "relevancy", "groundedness"):
                if key in blob:
                    return max(0.0, min(1.0, float(blob[key])))
            if "label" in blob:
                return label_to_score(parse_label(str(blob["label"])))
    except Exception:
        pass
    m = re.search(r"(?:score|faithfulness|relevancy)\s*[:=]\s*([0-9.]+)", text, re.I)
    if m:
        val = float(m.group(1))
        return max(0.0, min(1.0, val if val <= 1.0 else val / 10.0))
    return label_to_score(parse_label(text))


def label_to_score(label: str) -> float:
    return {"correct": 1.0, "partial": 0.5, "wrong": 0.0}.get(label, 0.5)


def score_to_label(score: float) -> str:
    if score >= 0.75:
        return "correct"
    if score >= 0.35:
        return "partial"
    return "wrong"


def pointwise_prompts(question: str, answer: str, contexts: Sequence[str], gold: str = "") -> dict[str, str]:
    ctx = "\n---\n".join((c or "")[:400] for c in contexts[:6])
    gold_line = f"\nGold answer: {gold}" if gold else ""
    shared = f"Question: {question}\nAnswer: {answer}{gold_line}\nContext:\n{ctx}\n"
    return {
        "faithfulness": shared
        + "Is every claim in the answer supported by the context? Reply JSON {\"score\": 0-1, \"label\": correct|partial|wrong}.",
        "relevancy": shared
        + "Does the answer address the question? Reply JSON {\"score\": 0-1, \"label\": correct|partial|wrong}.",
        "groundedness": shared
        + "Does the answer match the gold when gold is given, and stay inside the context? Reply JSON {\"score\": 0-1, \"label\": correct|partial|wrong}.",
    }


NO_ANSWER = "NOT IN SOURCES"  # the generator's abstention (app/agent/generate.py NOT_IN_SOURCES)


def correctness_prompt(question: str, answer: str, gold: str) -> str:
    """Is the answer the gold answer? The judge sees no retrieved context, so a wrong value copied faithfully
    from a retrieved chunk is still wrong; the pointwise prompts above mix that faithfulness into their label."""
    return (
        f"Question: {question}\nGold answer: {gold}\nAnswer: {answer}\n"
        "Does the answer give the gold answer? Judge only that, and accept another wording or number format "
        "(7e-6 is 0.000007); ignore style, length and sources.\n"
        "correct: the answer gives the gold answer.\n"
        "partial: the gold answer has several parts and the answer gives some of them.\n"
        f"wrong: the answer gives something else, or no answer ({NO_ANSWER} gives no answer).\n"
        'Reply JSON {"label": "correct" | "partial" | "wrong"}.'
    )


_VERDICT_WORDS = {"correct": "correct", "partial": "partial", "wrong": "wrong", "incorrect": "wrong"}


def parse_verdict(raw: str) -> tuple[str, bool]:
    """(label, parsed) from a correctness reply: the label of the JSON object, else a `label: x` pair, else a reply
    that is nothing but one label word. Anything else (prose such as "not correct", an empty reply) is unparsed and
    reads as partial; callers count it, so it is never silent."""
    text = (raw or "").strip()
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            blob = json.loads(m.group(0))
        except ValueError:
            blob = None
        label = _VERDICT_WORDS.get(str(blob.get("label", "")).strip().lower()) if isinstance(blob, dict) else None
        if label:
            return label, True
    m = re.search(r"label['\"]?\s*[:=]\s*['\"]?(correct|partial|wrong|incorrect)\b", text, re.I)
    if m:
        return _VERDICT_WORDS[m.group(1).lower()], True
    bare = re.sub(r"[^a-z]", "", text.lower())
    if bare in _VERDICT_WORDS:
        return _VERDICT_WORDS[bare], True
    return "partial", False


@dataclass
class PointwiseVerdict:
    faithfulness: float
    relevancy: float
    groundedness: float
    label: str
    raw: dict

    def as_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "relevancy": round(self.relevancy, 4),
            "groundedness": round(self.groundedness, 4),
            "label": self.label,
        }


class HeuristicJudge:
    """Offline stand-in: token overlap with gold / context. Not RAGAS-official."""

    name = "heuristic"

    def __call__(self, prompt: str) -> str:
        # Used when the caller still goes through a prompt; prefer judge_answer.
        return json.dumps({"score": 0.5, "label": "partial"})

    def judge_answer(self, question: str, answer: str, contexts: Sequence[str], gold: str = "") -> PointwiseVerdict:
        ans_tok = set(tokenize(answer))
        q_tok = set(tokenize(question))
        ctx_tok = set(tokenize(" ".join(contexts)))
        gold_tok = set(tokenize(gold)) if gold else set()
        faith = (len(ans_tok & ctx_tok) / len(ans_tok)) if ans_tok else 0.0
        rel = (len(ans_tok & q_tok) / len(q_tok)) if q_tok else 0.0
        if gold_tok:
            ground = len(gold_tok & ans_tok) / len(gold_tok)
        else:
            ground = faith
        label = score_to_label(0.5 * ground + 0.3 * faith + 0.2 * rel)
        return PointwiseVerdict(faith, rel, ground, label, {"backend": self.name})


def ollama_complete(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
) -> str | None:
    model = model or os.environ.get("JUDGE_MODEL", "qwen2.5:7b")
    url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/api/generate"
    # Same option keys as app/agent/generate.OllamaGenerator: a different num_ctx would
    # make Ollama reload the model on every alternation between generator and judge.
    body = json.dumps(
        {"model": model, "prompt": prompt, "stream": False, "keep_alive": "30m", "options": {"temperature": temperature, "seed": 0}}
    ).encode()
    try:
        import urllib.request

        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=float(os.environ.get("OLLAMA_TIMEOUT", "600"))) as resp:
            payload = json.loads(resp.read().decode())
        return payload.get("response") or payload.get("text")
    except Exception:
        return None


class LlmJudge:
    def __init__(self, complete: JudgeFn | None = None, model: str = "qwen2.5:14b", name: str | None = None):
        self.complete = complete
        self.model = model
        self.name = name or model

    def _ask(self, prompt: str, temperature: float = 0.0) -> str:
        if self.complete:
            return self.complete(prompt) or ""
        return ollama_complete(prompt, model=self.model, temperature=temperature) or ""

    def judge_answer(
        self, question: str, answer: str, contexts: Sequence[str], gold: str = ""
    ) -> PointwiseVerdict:
        prompts = pointwise_prompts(question, answer, contexts, gold)
        raw = {}
        scores = {}
        for dim, prompt in prompts.items():
            text = self._ask(prompt, temperature=0.0)
            raw[dim] = text
            scores[dim] = parse_score(text)
        mean = sum(scores.values()) / max(len(scores), 1)
        return PointwiseVerdict(
            faithfulness=scores.get("faithfulness", 0.0),
            relevancy=scores.get("relevancy", 0.0),
            groundedness=scores.get("groundedness", 0.0),
            label=score_to_label(mean),
            raw=raw,
        )

    def judge_correctness(self, question: str, answer: str, gold: str) -> dict:
        """The correctness verdict (correctness_prompt): label, whether the reply could be read, the reply itself."""
        raw = self._ask(correctness_prompt(question, answer, gold), temperature=0.0)
        label, parsed = parse_verdict(raw)
        return {"label": label, "parsed": parsed, "raw": raw}


def pairwise_ab_ba(
    complete: JudgeFn,
    question: str,
    answer_a: str,
    answer_b: str,
) -> dict:
    """Position-swap pairwise: average AB and BA. Returns P(A better)."""

    def one(left: str, right: str) -> float:
        prompt = (
            f"Question: {question}\nAnswer A:\n{left}\nAnswer B:\n{right}\n"
            "Which is better grounded and complete? Reply A, B, or TIE."
        )
        raw = (complete(prompt) or "").strip().upper()
        if re.search(r"\bTIE\b", raw):
            return 0.5
        # first standalone A/B after ignoring the prompt letters in "ANSWER A"
        if re.search(r"\bB\b", raw) and not re.search(r"\bA\b", raw):
            return 0.0
        if re.search(r"\bA\b", raw) and not re.search(r"\bB\b", raw):
            return 1.0
        if raw.startswith("B") or "WINNER: B" in raw or "PREFER B" in raw:
            return 0.0
        if raw.startswith("A") or "WINNER: A" in raw or "PREFER A" in raw:
            return 1.0
        # both mentioned: last wins
        a_pos = raw.rfind(" A")
        b_pos = raw.rfind(" B")
        if a_pos == b_pos == -1:
            return 0.5
        return 1.0 if a_pos > b_pos else 0.0

    ab = one(answer_a, answer_b)
    ba = 1.0 - one(answer_b, answer_a)  # BA prompt: A=answer_b, so P(original A)
    # one(answer_b, answer_a) is P(answer_b better when listed first) = P(original B | BA)
    # P(original A | BA) = 1 - that if no ties... handle ties: if BA says TIE, one returns 0.5, 1-0.5=0.5
    return {
        "p_a_ab": ab,
        "p_a_ba": ba,
        "p_a": 0.5 * (ab + ba),
        "position_gap": abs(ab - ba),
    }


def self_consistency_vote(
    complete: JudgeFn,
    prompt: str,
    *,
    k: int = 3,
    temperature: float = 0.3,
) -> dict:
    """Wang et al. self-consistency: majority label over k samples."""
    labels = []
    for i in range(k):
        # temperature is the caller's job; we just call k times
        labels.append(parse_label(complete(prompt) or ""))
    counts = Counter(labels)
    winner, n = counts.most_common(1)[0]
    return {
        "label": winner,
        "votes": dict(counts),
        "k": k,
        "agreement": n / k,
        "score": label_to_score(winner),
    }


_NUM_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _standalone(text: str, start: int, end: int) -> bool:
    """The number at text[start:end] is not part of a word or a longer number (the 2 in dinov2, the 50 in resnet50)."""
    before, after, after2 = text[start - 1:start], text[end:end + 1], text[end + 1:end + 2]
    if before and (before.isalnum() or before in "_."):
        return False
    if after and (after.isalnum() or after == "_"):
        return False
    return not (after == "." and after2.isdigit())


def _numbers(text: str, *, bounded: bool = False) -> set[float]:
    out = set()
    for m in _NUM_RE.finditer(text or ""):
        if bounded and not _standalone(text, m.start(), m.end()):
            continue
        try:
            out.add(float(m.group(0)))
        except ValueError:
            continue
    return out


def _bounded_pattern(value: str) -> str:
    head = r"(?<![\w.])" if re.match(r"\w", value) else ""
    tail = r"(?!\w|\.\d)" if re.match(r"\w", value[-1]) else ""
    return head + re.escape(value) + tail


def contains_bounded(text: str, value: str) -> bool:
    """`value` occurs in `text` and is not part of a longer number or word at an end where it starts or ends with
    a letter or digit: "2" is not in "run 42" or "2.5", "0.053" is not in "0.0533", "oak" is not in "oaks"."""
    value = value.strip()
    return bool(value) and re.search(_bounded_pattern(value), text) is not None


def single_number(value: str) -> float | None:
    """The number `value` is, when the whole string is one number ("1e-04", "0.0555", "3"), else None."""
    found = _NUM_RE.findall(value or "")
    return float(found[0]) if len(found) == 1 and value.strip() == found[0] else None


def replace_bounded(text: str, value: str, new: str) -> tuple[str, int]:
    """`text` with every occurrence of `value` that contains_bounded finds (any case) replaced by `new`, and, when
    `value` is one number, every standalone number equal to it (0.0001 for 1e-04); with the count of replacements."""
    value = value.strip()
    if not value:
        return text, 0
    target = single_number(value)
    if target is not None:
        count = 0

        def swap(m: re.Match) -> str:
            nonlocal count
            if _standalone(text, m.start(), m.end()) and abs(float(m.group(0)) - target) <= 1e-9 * max(1.0, abs(target)):
                count += 1
                return new
            return m.group(0)

        return _NUM_RE.sub(swap, text), count
    return re.subn(_bounded_pattern(value), lambda m: new, text, flags=re.I)


def exact_match_label(answer: str, gold: str, *, bounded: bool = False) -> str:
    """correct: the gold string is in the answer, or the gold is a number and the answer
    states the same number in any format (1e-04 == 0.0001); partial: token overlap;
    wrong: none. This is the deterministic label the judges are measured against.
    bounded=True (the answer evaluation's reference since M29) counts the gold string, and a
    number in the answer, only where it is not part of a longer number or word
    (contains_bounded, _standalone); plain containment finds gold "8" in "runs 45, 46, 47, 48"
    and the number comparison finds 2 in "dinov2"."""
    if not gold.strip():
        return "partial"
    a, g = answer.lower(), gold.lower()
    if (contains_bounded(a, g) if bounded else g in a) or a.strip() == g.strip():
        return "correct"
    gold_nums = _numbers(gold)
    if gold_nums and len(_NUM_RE.findall(gold)) == 1 and gold.strip() == _NUM_RE.findall(gold)[0]:
        if any(abs(x - y) <= 1e-9 * max(1.0, abs(y)) for y in gold_nums for x in _numbers(answer, bounded=bounded)):
            return "correct"
    if set(tokenize(gold)) & set(tokenize(answer)):
        return "partial"
    return "wrong"
