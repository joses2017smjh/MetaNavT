"""Synthetic training queries for the distilled rerankers (M43): documents sampled from the three corpora, queries
written by a local open LLM, near-duplicates of test queries dropped.

    python -m app.eval.distill_queries generate --data-dir <BEIR dir> --work <dir> [--ollama-url http://127.0.0.1:11434]
    python -m app.eval.distill_queries filter   --data-dir <BEIR dir> --work <dir> [--out-dir bench/distill]

The students of app/eval/small_rerankers.py are trained without a label: on a teacher's scores for queries nobody
judged. Those queries must not be the benchmark's, so they are written here:

  sample    documents of each corpus (CORPORA), seeded: every chunk of fixture v1, and a fixed number of BEIR SciFact
            and FiQA documents drawn from corpus.jsonl. This step opens corpus.jsonl and the fixture's files only:
            never queries.jsonl (it holds the test queries) and never the qrels.
  generate  one prompt per document (PROMPT) to an Ollama model, asking for questions, keyword queries and one-sentence
            statements that the document answers. The prompt names the kind of corpus and shows the document; it
            contains no example query, so no benchmark query can reach the model. Every raw completion is kept
            (completions.jsonl), with how it was read.
  filter    the completions parsed and cleaned (CLEAN), exact repeats dropped, then every synthetic query compared
            with every test query of the three benchmarks: fixture v1's 136 gold questions (both halves) and the
            SciFact and FiQA test queries. This is the only step that reads test queries, and it only removes: a
            synthetic query is dropped when it matches a test query after normalisation, or its token-set Jaccard with
            one reaches NEAR_DUPLICATE["jaccard"], or the cosine of their bge-small embeddings reaches
            NEAR_DUPLICATE["cosine"]. What was dropped, the counts at other thresholds and the nearest kept queries
            are recorded (queries.meta.json). Then 5% of each corpus's source documents are set aside, by document, as
            the validation set a student's checkpoint is chosen on.

What this does not make: a judged query. No query here has a relevance label; the teacher's scores are the only
signal (app/eval/distill_teacher.py). The documents are the benchmarks' own corpora, so a student trained on these
queries has seen the documents its test questions are about, though none of the questions and none of the labels.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from app.eval.provenance import command_line, device_info, git_sha, git_state, ollama_model_provenance, ollama_tags, ollama_version, run_provenance

FORMS = ("questions", "keywords", "statements")
# What is sampled from each corpus, and how many queries of each form one document is asked for. The fixture is 71
# chunks, so each is asked for more; a BEIR document is asked for one of each form.
CORPORA: dict[str, dict[str, Any]] = {
    "fixture": {"description": "files of a machine-learning research project (configs, logs, job scripts, code, notes, results)",
                "documents": None, "per_form": 4},
    "scifact": {"description": "abstracts of scientific papers", "documents": 1500, "per_form": 1},
    "fiqa": {"description": "posts from a personal-finance question-and-answer site", "documents": 1800, "per_form": 1},
}
SAMPLE_SEED = 0
MIN_WORDS = 30  # a sampled BEIR document has at least this many words: a one-line post gives the model nothing to ask about
MAX_CHARS = 3000  # of the document shown to the model
PROMPT = """You write search queries for testing a document search engine.
Corpus: {description}
Document:
\"\"\"
{text}
\"\"\"
Write search queries that this document answers, as a person who has not seen the document would type them:
- {n} natural-language questions
- {n} keyword queries (3 to 8 words, no question words)
- {n} one-sentence statements of a fact that the document supports
Each query must be answerable from this document alone and specific enough to prefer it over similar documents. Do not copy a whole sentence from the document.
Reply with JSON only: {{"questions": [...], "keywords": [...], "statements": [...]}}"""
LLM_MODEL = "gemma4:12b"
LLM_OPTIONS = {"temperature": 0.0, "seed": 0, "num_ctx": 4096, "num_predict": 600}
LLM_REQUEST = {"think": False, "format": "json"}  # no reasoning tokens; the reply constrained to a JSON object
CLEAN = {"min_words": 2, "max_words": 60}  # a query outside this range is dropped; whitespace is collapsed
NEAR_DUPLICATE = {"jaccard": 0.7, "cosine": 0.90, "embedder": "BAAI/bge-small-en-v1.5"}
REPORT_COSINE = (0.80, 0.85, 0.90, 0.95)
REPORT_JACCARD = (0.5, 0.6, 0.7, 0.8)
VALIDATION = {"fraction": 0.05, "min_documents": 4, "seed": 1}
PARSE_STATUSES = ("ok", "no_completion", "not_json", "no_queries")
MAX_FAILED_SHARE = 0.02  # of the calls: above it the generate step exits non-zero (a few failed calls are counted, not hidden)
QUERIES_FILE, COMPLETIONS_FILE, META_FILE = "queries.jsonl.gz", "completions.jsonl.gz", "queries.meta.json"


# ---------------------------------------------------------------- sample


def fixture_documents(root: Path) -> list[tuple[str, str]]:
    """(chunk id, text) of every chunk of fixture v1, as the benchmark indexes them."""
    from app.eval.index_loader import load_chunks

    return [(c.chunk_id, c.text) for c in load_chunks(root / "bench" / "corpus" / "files", mtimes="manifest")]


def sample_ids(ids: Iterable[str], texts: dict[str, str], n: int, *, seed: int = SAMPLE_SEED, min_words: int = MIN_WORDS) -> list[str]:
    """`n` document ids drawn without replacement: the ids sorted, permuted by numpy's RandomState(seed), and the first
    `n` whose text has at least `min_words` words kept, in that order."""
    ordered = sorted(ids)
    picked = []
    for i in np.random.RandomState(seed).permutation(len(ordered)):
        if len(texts[ordered[i]].split()) >= min_words:
            picked.append(ordered[i])
            if len(picked) == n:
                break
    return picked


def sample_documents(root: Path, data_dir: Path, corpora: dict[str, dict] | None = None) -> dict[str, list[tuple[str, str]]]:
    """{corpus: [(document id, text)]}: every fixture chunk, and the seeded sample of each BEIR corpus. Reads the
    fixture's files and each dataset's corpus.jsonl (app.eval.beir.load_corpus), nothing else."""
    from app.eval.beir import load_corpus

    out: dict[str, list[tuple[str, str]]] = {}
    for name, spec in (corpora or CORPORA).items():
        if name == "fixture":
            out[name] = fixture_documents(root)
            continue
        texts = load_corpus(Path(data_dir) / name)
        out[name] = [(did, texts[did]) for did in sample_ids(texts, texts, spec["documents"])]
    return out


def documents_sha256(docs: Sequence[tuple[str, str]]) -> str:
    return hashlib.sha256(json.dumps([[did, text] for did, text in docs], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


# ---------------------------------------------------------------- generate


def prompt_for(corpus: str, text: str, corpora: dict[str, dict] | None = None) -> str:
    spec = (corpora or CORPORA)[corpus]
    return PROMPT.format(description=spec["description"], text=text[:MAX_CHARS], n=spec["per_form"])


def ollama_chat(prompt: str, model: str, base_url: str, *, timeout: float = 600.0) -> dict[str, Any]:
    """One /api/chat call; {"content", "eval_count", "done_reason", "error"}, content None when the call failed."""
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, "keep_alive": "30m",
                       "options": LLM_OPTIONS, **LLM_REQUEST}).encode()
    try:
        req = urllib.request.Request(base_url.rstrip("/") + "/api/chat", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read().decode())
    except Exception as e:  # noqa: BLE001 - a failed call is recorded and counted, never retried silently
        return {"content": None, "eval_count": None, "done_reason": None, "error": type(e).__name__}
    return {"content": (out.get("message") or {}).get("content"), "eval_count": out.get("eval_count"),
            "done_reason": out.get("done_reason"), "error": None}


def generate(docs: dict[str, list[tuple[str, str]]], complete: Callable[[str], dict[str, Any]], out_path: Path, *, workers: int = 8,
             corpora: dict[str, dict] | None = None, limit: int | None = None) -> int:
    """One completion per document, appended to `out_path` as JSON lines; a document already there is skipped, so an
    interrupted run resumes. Returns the number of calls made."""
    done = {(r["corpus"], r["doc_id"]) for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [(c, did, text) for c, rows in docs.items() for did, text in rows if (c, did) not in done]
    if limit is not None:
        todo = todo[:limit]

    def one(item):
        corpus, did, text = item
        t0 = time.perf_counter()
        reply = complete(prompt_for(corpus, text, corpora))
        return {"corpus": corpus, "doc_id": did, "ms": round((time.perf_counter() - t0) * 1000.0, 1), **reply}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(max(1, workers)) as pool:
        for row in pool.map(one, todo):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
    return len(todo)


# ---------------------------------------------------------------- parse and clean


def parse_completion(content: str | None, per_form: int) -> tuple[dict[str, list[str]], str]:
    """({form: cleaned queries}, status): at most `per_form` per form, in the order written; a query is cleaned by
    collapsing whitespace and dropped when it is not a string, repeats an earlier one of the same document, or has
    fewer than CLEAN["min_words"] or more than CLEAN["max_words"] words. status is one of PARSE_STATUSES."""
    empty: dict[str, list[str]] = {form: [] for form in FORMS}
    if not content:
        return empty, "no_completion"
    try:
        blob = json.loads(content)
    except ValueError:
        return empty, "not_json"
    if not isinstance(blob, dict):
        return empty, "not_json"
    seen: set[str] = set()
    out = {form: [] for form in FORMS}
    for form in FORMS:
        items = blob.get(form)
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, str):
                continue
            query = " ".join(item.split())
            if not CLEAN["min_words"] <= len(query.split()) <= CLEAN["max_words"] or normalise(query) in seen:
                continue
            seen.add(normalise(query))
            out[form].append(query)
            if len(out[form]) == per_form:
                break
    return out, "ok" if any(out.values()) else "no_queries"


def query_id(corpus: str, doc_id: str, form: str, k: int) -> str:
    return f"{corpus}:{doc_id}:{form[0]}{k}"


# ---------------------------------------------------------------- near-duplicates of test queries


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalise(text: str) -> str:
    """Lowercase, every run of characters that are not a-z or 0-9 turned into one space."""
    return _NON_ALNUM.sub(" ", text.lower()).strip()


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def benchmark_queries(root: Path, data_dir: Path) -> list[dict[str, str]]:
    """Every test query a synthetic query is compared with: fixture v1's gold questions (both halves of the split) and
    the test split of BEIR SciFact and FiQA. The filter step is the only caller."""
    from app.eval.beir import load_beir
    from app.eval.gold import load_gold

    out = [{"set": "fixture", "id": q.id, "text": q.question} for q in load_gold(root / "bench" / "gold" / "questions.jsonl")]
    for name in ("scifact", "fiqa"):
        _, queries, _ = load_beir(Path(data_dir) / name, "test")
        out += [{"set": name, "id": qid, "text": text} for qid, text in queries.items()]
    return out


def nearest_test_queries(queries: Sequence[str], tests: Sequence[dict[str, str]], embed: Callable[[Sequence[str]], np.ndarray]) -> list[dict[str, Any]]:
    """Per synthetic query: whether it equals a test query after normalisation, its highest token-set Jaccard with a
    test query and its highest embedding cosine with one, each with the test query it is nearest to. `embed` returns
    unit rows (bge-small in the committed run)."""
    if not queries:
        return []
    test_norm = [normalise(t["text"]) for t in tests]
    test_sets = [set(n.split()) for n in test_norm]
    exact = set(test_norm)
    cos = np.asarray(embed(list(queries)), dtype=np.float64) @ np.asarray(embed([t["text"] for t in tests]), dtype=np.float64).T
    out = []
    for i, query in enumerate(queries):
        norm = normalise(query)
        tokens = set(norm.split())
        scores = [jaccard(tokens, s) for s in test_sets]
        j = max(range(len(tests)), key=scores.__getitem__)
        c = int(np.argmax(cos[i]))
        out.append({"equal": norm in exact,
                    "jaccard": round(scores[j], 4), "jaccard_nearest": {"set": tests[j]["set"], "id": tests[j]["id"]},
                    "cosine": round(float(cos[i, c]), 4), "cosine_nearest": {"set": tests[c]["set"], "id": tests[c]["id"]}})
    return out


def flag_rules(nearest: dict[str, Any], rule: dict[str, Any] | None = None) -> list[str]:
    """Which of the near-duplicate rules a synthetic query trips (empty: kept)."""
    rule = rule or NEAR_DUPLICATE
    return [name for name, hit in (("equal", nearest["equal"]), ("jaccard", nearest["jaccard"] >= rule["jaccard"]),
                                   ("cosine", nearest["cosine"] >= rule["cosine"])) if hit]


def validation_documents(doc_ids: Iterable[str], spec: dict[str, Any] | None = None) -> set[str]:
    """The documents of one corpus whose queries are the validation set: the ids sorted, permuted (RandomState(seed)),
    the first max(min_documents, ceil(fraction * n)) taken."""
    spec = spec or VALIDATION
    ordered = sorted(set(doc_ids))
    if not ordered:
        return set()
    n = min(len(ordered), max(spec["min_documents"], math.ceil(spec["fraction"] * len(ordered))))
    return {ordered[i] for i in np.random.RandomState(spec["seed"]).permutation(len(ordered))[:n]}


def filter_queries(completions: Sequence[dict], tests: Sequence[dict[str, str]], embed: Callable[[Sequence[str]], np.ndarray], *,
                   corpora: dict[str, dict] | None = None) -> tuple[list[dict], dict[str, Any]]:
    """(the kept synthetic queries with their split, the report): parse every completion, drop repeats inside a corpus,
    flag and drop near-duplicates of `tests`, then split by source document."""
    corpora = corpora or CORPORA
    parse = {c: {s: 0 for s in PARSE_STATUSES} for c in corpora}
    candidates: list[dict] = []
    repeats = {c: 0 for c in corpora}
    seen: dict[str, set[str]] = {c: set() for c in corpora}
    for row in completions:
        corpus = row["corpus"]
        forms, status = parse_completion(row.get("content"), corpora[corpus]["per_form"])
        parse[corpus][status] += 1
        for form in FORMS:
            for k, query in enumerate(forms[form], start=1):
                if normalise(query) in seen[corpus]:
                    repeats[corpus] += 1
                    continue
                seen[corpus].add(normalise(query))
                candidates.append({"id": query_id(corpus, row["doc_id"], form, k), "corpus": corpus, "doc_id": row["doc_id"],
                                   "form": form, "query": query})
    nearest = nearest_test_queries([c["query"] for c in candidates], tests, embed)
    kept, flagged = [], []
    for cand, near in zip(candidates, nearest):
        rules = flag_rules(near)
        (flagged if rules else kept).append({**cand, **({"rules": rules, **near} if rules else {})})
    by_corpus: dict[str, Any] = {}
    for corpus in corpora:
        mine = [(c, n) for c, n in zip(candidates, nearest) if c["corpus"] == corpus]
        mine_kept = [(c, n) for c, n in mine if not flag_rules(n)]
        val = validation_documents(c["doc_id"] for c, _ in mine_kept)
        for q in kept:
            if q["corpus"] == corpus:
                q["split"] = "validation" if q["doc_id"] in val else "train"
        top = sorted(mine_kept, key=lambda cn: cn[1]["cosine"], reverse=True)[:10]
        by_corpus[corpus] = {
            "completions": parse[corpus],
            "queries_parsed": len(mine) + repeats[corpus],
            "repeats_dropped": repeats[corpus],
            "checked": len(mine),
            "near_duplicates_dropped": len(mine) - len(mine_kept),
            "dropped_by_rule": {name: sum(1 for _, n in mine if name in flag_rules(n)) for name in ("equal", "jaccard", "cosine")},
            "at_other_thresholds": {
                "cosine_at_least": {f"{t:.2f}": sum(1 for _, n in mine if n["cosine"] >= t) for t in REPORT_COSINE},
                "jaccard_at_least": {f"{t:.1f}": sum(1 for _, n in mine if n["jaccard"] >= t) for t in REPORT_JACCARD},
            },
            "kept": len(mine_kept),
            "kept_by_form": {form: sum(1 for c, _ in mine_kept if c["form"] == form) for form in FORMS},
            "kept_by_split": {split: sum(1 for q in kept if q["corpus"] == corpus and q["split"] == split) for split in ("train", "validation")},
            "source_documents": len({c["doc_id"] for c, _ in mine_kept}),
            "validation_documents": len(val),
            "nearest_kept": [{"id": c["id"], "query": c["query"], "cosine": n["cosine"], "cosine_nearest": n["cosine_nearest"],
                              "jaccard": n["jaccard"]} for c, n in top],
        }
    report = {"rule": {**NEAR_DUPLICATE, "equal": "the normalised strings are the same",
                       "normalise": "lowercase, every run of characters outside a-z and 0-9 becomes one space",
                       "jaccard_of": "the sets of normalised tokens", "cosine_of": "unit embeddings of the bare queries",
                       "dropped_when": "any of the three holds against any test query of any of the three benchmarks"},
              "test_queries": {name: sum(1 for t in tests if t["set"] == name) for name in sorted({t["set"] for t in tests})},
              "by_corpus": by_corpus, "flagged": flagged}
    return kept, report


# ---------------------------------------------------------------- files


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl_gz(path: Path, rows: Iterable[dict]) -> str:
    """Rows as gzipped JSON lines with no timestamp or file name in the gzip header, so the same rows give the same
    bytes; returns the file's sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as fh:
        for row in rows:
            fh.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _embedder():
    from app.retrieval.embedders import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(NEAR_DUPLICATE["embedder"])


def _parse_args(argv: list[str] | None):
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["generate", "filter"])
    p.add_argument("--data-dir", type=Path, default=root / "bench" / "external", help="where the BEIR datasets are (scifact/, fiqa/)")
    p.add_argument("--work", type=Path, required=True, help="the working directory: completions.jsonl and generation.meta.json")
    p.add_argument("--out-dir", type=Path, default=root / "bench" / "distill", help="filter: where the committed files go")
    p.add_argument("--ollama-url", default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    p.add_argument("--model", default=LLM_MODEL)
    p.add_argument("--workers", type=int, default=8, help="generate: calls in flight")
    p.add_argument("--limit", type=int, default=None, help="generate: at most this many documents (a smoke run; never the committed set)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    started = git_state(root)
    raw_path, gen_meta_path = args.work / "completions.jsonl", args.work / "generation.meta.json"
    if args.step == "generate":
        tags = ollama_tags(args.ollama_url)
        model = ollama_model_provenance(args.model, "query writer", tags)
        if model["revision"] is None:
            raise SystemExit(f"{args.model} is not served at {args.ollama_url}")
        docs = sample_documents(root, args.data_dir)
        t0 = time.perf_counter()
        calls = generate(docs, lambda prompt: ollama_chat(prompt, args.model, args.ollama_url), raw_path, workers=args.workers, limit=args.limit)
        meta = {"git_sha": git_sha(root), **run_provenance(root=root, started=started), "timestamp": datetime.now(timezone.utc).isoformat(),
                "command": command_line(), "device": device_info(args.device),
                "llm": {**model, "ollama_version": ollama_version(args.ollama_url), "options": LLM_OPTIONS, "request": LLM_REQUEST,
                        "endpoint": "/api/chat, one user message"},
                "prompt": {"template": PROMPT, "sha256": hashlib.sha256(PROMPT.encode()).hexdigest(), "max_chars_of_document": MAX_CHARS,
                           "contains_example_queries": False},
                "sample": {"seed": SAMPLE_SEED, "min_words": MIN_WORDS, "reads": "the fixture's files and each BEIR corpus.jsonl; never queries.jsonl or qrels",
                           "corpora": {c: {**CORPORA[c], "n_documents": len(rows), "documents_sha256": documents_sha256(rows)} for c, rows in docs.items()}},
                "calls_this_run": calls, "seconds_this_run": round(time.perf_counter() - t0, 1), "workers": args.workers, "limit": args.limit}
        gen_meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        rows = read_jsonl(raw_path)
        empty = sum(1 for r in rows if not r.get("content"))
        print(f"{calls} calls in {meta['seconds_this_run']} s; {len(rows)} completions in {raw_path.name}; {empty} without content")
        if empty > MAX_FAILED_SHARE * len(rows):  # a server that went away fails every later call at once: not a set to filter
            raise SystemExit(f"{empty} of {len(rows)} calls returned nothing (more than {MAX_FAILED_SHARE:.0%}): nothing is retried; "
                             "find the cause and generate again into an empty working directory")
        return 0

    completions = read_jsonl(raw_path)
    generation = json.loads(gen_meta_path.read_text())
    if generation.get("limit") is not None:
        raise SystemExit("these completions come from a --limit run: not the committed set")
    expected = sum(c["n_documents"] for c in generation["sample"]["corpora"].values())
    if len(completions) != expected:
        raise SystemExit(f"{len(completions)} completions for {expected} sampled documents: generation is not finished")
    embedder = _embedder()
    tests = benchmark_queries(root, args.data_dir)
    kept, report = filter_queries(completions, tests, embedder.encode)
    kept.sort(key=lambda q: (q["corpus"], q["doc_id"], FORMS.index(q["form"]), q["id"]))
    statuses = {(c["corpus"], c["doc_id"]): parse_completion(c.get("content"), CORPORA[c["corpus"]]["per_form"])[1] for c in completions}
    completions = sorted(completions, key=lambda c: (c["corpus"], c["doc_id"]))
    queries_sha = write_jsonl_gz(args.out_dir / QUERIES_FILE, kept)
    completions_sha = write_jsonl_gz(args.out_dir / COMPLETIONS_FILE, ({**c, "parse": statuses[(c["corpus"], c["doc_id"])]} for c in completions))
    from app.eval.provenance import model_device, model_provenance

    meta = {"git_sha": git_sha(root), **run_provenance(root=root, started=started), "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": command_line(), "device": device_info(args.device),
            "files": {QUERIES_FILE: {"sha256": queries_sha, "n": len(kept), "fields": "id, corpus, doc_id, form, query, split"},
                      COMPLETIONS_FILE: {"sha256": completions_sha, "n": len(completions),
                                         "fields": "corpus, doc_id, ms, content (the raw completion), eval_count, done_reason, error, parse"}},
            "generation": generation,
            "clean": CLEAN,
            "near_duplicates": {"embedder": {**model_provenance("st:" + NEAR_DUPLICATE["embedder"], "embedder"), **embedder.provenance(),
                                             "device": model_device(embedder._model)}, **report},
            "validation": {**VALIDATION, "rule": "by source document, per corpus: the ids sorted, permuted (RandomState(seed)), the first "
                                                 "max(min_documents, ceil(fraction * n)) are validation; every query of a document goes with it"},
            "what_this_is_not": ("no query has a relevance label; the documents are the benchmarks' own corpora, so a student "
                                 "trained on these queries has seen the documents its test questions are about, though no test "
                                 "question and no label")}
    (args.out_dir / META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    for corpus, block in report["by_corpus"].items():
        print(corpus, json.dumps({k: block[k] for k in ("completions", "checked", "near_duplicates_dropped", "dropped_by_rule", "kept", "kept_by_split")}))
    print(f"wrote {args.out_dir / QUERIES_FILE} ({len(kept)} queries, sha256 {queries_sha[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
