"""Derived probe files outside the frozen fixture (bench/probes/). They are not gold labels.

    python -m app.eval.probes            # write the files under bench/probes/
    python -m app.eval.probes --check    # exit 1 if a committed file differs from what this script writes

paraphrases.jsonl: rule-based rewrites of a sample of the gold questions (bench/gold/questions.jsonl,
read only). Each probe names the question it rewrites (`source_id`) and inherits its relevant paths, so
retrieval for the probe can be scored and a semantic-cache hit can be told apart as served from the
rewritten question's entry or from another question's. The paths are inherited, not judged: they are
right only as far as a rewrite keeps the question's meaning, which is why the rules below only drop
filler words or wrap the question in a request.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.eval.gold import GoldQuestion, load_gold

ROOT = Path(__file__).resolve().parents[2]
GOLD = "bench/gold/questions.jsonl"
PROBE_DIR = "bench/probes"
PARAPHRASES = f"{PROBE_DIR}/paraphrases.jsonl"
PER_CATEGORY = 3  # the first, the middle and the last question of every category, in gold order

# Words the keyword rule drops: question words, auxiliaries, articles and prepositions. Content words
# (run ids, file names, settings, numbers) are never in this set.
KEYWORD_STOP = frozenset({
    "what", "which", "who", "where", "when", "how", "did", "does", "do", "is", "are", "was", "were", "be",
    "the", "a", "an", "of", "for", "in", "to", "with", "by", "use", "used", "uses", "me", "i", "it", "that",
})


def keywords(question: str) -> str:
    """'what learning rate did run 40 use' -> 'learning rate run 40' (how a user types into a search box)."""
    text = question.lower().replace("how many", "number of").replace("what's", "what is")
    words = [w.strip("?,.:") for w in text.split()]
    return " ".join(w for w in words if w and w not in KEYWORD_STOP)


IMPERATIVES = frozenset({"open", "find", "show", "read", "locate", "explain", "compare", "list", "get"})


def ask(question: str) -> str:
    """'what encoder did run 40 use' -> 'can you tell me what encoder did run 40 use?'; 'open x' -> 'can you open x?'."""
    first = question.split()[0].lower() if question.split() else ""
    return f"can you {question}?" if first in IMPERATIVES else f"can you tell me {question}?"


def prefix(question: str) -> str:
    return f"quick question: {question}"


RULES = (("keywords", keywords), ("ask", ask), ("prefix", prefix))


@dataclass(frozen=True)
class Probe:
    id: str
    source_id: str
    question: str
    rule: str
    category: str
    relevant_paths: list[str]


def sample(gold: list[GoldQuestion], per_category: int = PER_CATEGORY) -> list[GoldQuestion]:
    """Evenly spaced questions of every category (categories in name order, questions in gold order)."""
    out: list[GoldQuestion] = []
    for category in sorted({q.category for q in gold}):
        rows = [q for q in gold if q.category == category]
        k = min(per_category, len(rows))
        picks = sorted({round(i * (len(rows) - 1) / (k - 1)) for i in range(k)}) if k > 1 else [0]
        out.extend(rows[i] for i in picks)
    return out


def paraphrases(gold: list[GoldQuestion]) -> list[Probe]:
    """One rewrite per sampled question; the rules take turns, and a rule that changes nothing passes to the next."""
    probes = []
    for i, q in enumerate(sample(gold)):
        for step in range(len(RULES)):
            rule, fn = RULES[(i + step) % len(RULES)]
            text = fn(q.question)
            if text.strip().lower() != q.question.strip().lower():
                break
        probes.append(Probe(f"p{i + 1:03d}", q.id, text, rule, q.category, list(q.relevant_ids())))
    return probes


def load_probes(path: Path) -> list[Probe]:
    return [Probe(**json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]


def render(root: Path = ROOT) -> dict[str, str]:
    """{relative path: file text} for every probe file this script writes."""
    gold = load_gold(root / GOLD)
    return {PARAPHRASES: "".join(json.dumps(asdict(p)) + "\n" for p in paraphrases(gold))}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="compare the committed files with what would be written")
    args = p.parse_args(argv)
    drift = []
    for rel, text in render().items():
        path = ROOT / rel
        if args.check:
            if not path.exists() or path.read_text() != text:
                drift.append(rel)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote {rel} ({text.count(chr(10))} probes)")
    for rel in drift:
        print(f"drift: {rel} differs from `python -m app.eval.probes` output")
    if args.check and not drift:
        print("probe files in sync")
    return 1 if drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
