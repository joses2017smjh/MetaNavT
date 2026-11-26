"""M46, part (d): the arm wiring, and the fixture and BEIR runners.

    python -m app.eval.llm_query_arms --dataset fixture --out bench/results/llm_query_arms.json      (make bench-llm-query-arms)
    python -m app.eval.llm_query_arms --dataset scifact --data-dir <beir> --out ...                  (make bench-beir-llm-query-arms)
    python -m app.eval.llm_query_arms --dataset fiqa --data-dir <beir> --out ...                     (make bench-beir-fiqa-llm-query-arms)

Scoring needs no LLM: every LLM output comes from bench/llm_query (a missing key stops the run and lists the keys; no
fallback); this module never imports an Ollama client. It needs bge-small and bge-reranker-v2-m3 (a GPU, or a CPU
slowly); reranker scores are cached on disk by (sha256 of the query text, unit, reranker revision, GPU name) so a
killed run resumes, and an entry is reused only on the same GPU type.

The wiring (llm_query_spec.ARMS; "unit" is a chunk on the fixture, a document on BEIR):

  first stage   BM25 + bge-small fused by RRF k 60 with the reranker OFF (P0-shaped), for the query, an alternative,
                a sub-question or a rewrite; the HyDE vector replaces the dense leg's query vector (BM25 reads the query)
  head_rerank   bge-reranker-v2-m3 over a list's top 20 against the ORIGINAL query, the tail in list order (P1's shape)
  union         RRF k 60 over the first-stage lists of [query + alternatives or sub-questions], truncated to k
  per-sub       each list's top 20 reranked against its own (sub-)question, units merged by their highest reranker
                score (ties in union order), then the rest in union order
  HyDE post     P1's list re-sorted by RRF with the HyDE dense list (the wiring of the rows recorded before M18)
  corrective    grade the input list's top G units; "yes" units in order, the rest of the list, the "no" units in
                order; when no unit is "yes" and the rewrite is non-empty and differs from the query (case-insensitive):
                the placement's first stage with the rewrite (post-rerank: its top 20 reranked against the original
                query), without the old top G, then those G units
  every arm     truncated to k (50 chunks on the fixture, 100 documents on BEIR); on BEIR a document whose id equals
                the query id is then dropped, as beir.run_config does

The arms never call heuristic_hypothesis, llm_hypothesis, hyde_embed, heuristic_decompose, llm_decompose,
expand_queries, rrf_union_paths or multi_query_retrieve (each substitutes a heuristic, pads, or reranks every list
against its own query); the heuristic stand-in rows (fixture only, tagged [heuristic]) call the keyword helpers
through HeuristicSource and nothing else. A failed, empty, unparseable or prompt-overflowed HyDE, multi-query or
decomposition call makes the arm behave as its parent for that query; such a grade counts as "yes"; such a rewrite
means no rewrite (llm_query_parse.read_call).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from app.eval import llm_query_bar as bar
from app.eval import llm_query_spec as spec
from app.eval.llm_query_parse import parse

ROOT = Path(__file__).resolve().parents[2]
LLM_DIR = "bench/llm_query"
PARENTS_FILE = "bench/llm_query/parents.json.gz"
QUERY_IDS = "bench/llm_query/query_ids.json"
FIXTURE_METRICS = ("ndcg@10", "recall@10", "recall@50", "mrr@10", "input_recall@20")
BEIR_METRICS = ("ndcg@10", "recall@10", "recall@100", "mrr@10", "input_recall@20")


def cache_file(dataset: str, tag: str) -> str:
    return f"{LLM_DIR}/{dataset}.{tag}.jsonl.gz"


def list_sha256(units: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(units).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- reading a cached call (the fallback rule)


def read_call(entry: dict | None, kind: str, query: str) -> tuple[Any, str, bool]:
    """(value, status, truncated) of one cached call. status: ok | empty | unparseable | unchanged | failed |
    overflowed. A failed or prompt-overflowed call has no value; the parsers handle the rest (llm_query_parse)."""
    if entry is None:
        raise KeyError("no cache entry")
    truncated = entry.get("done_reason") == "length"
    if entry.get("error"):
        return None, "failed", truncated
    if entry.get("overflowed") or spec.overflowed(entry.get("prompt_eval_count"), kind):
        return None, "overflowed", truncated
    value, status = parse(kind, entry.get("reply"), query, truncated)
    return value, status, truncated


# ---------------------------------------------------------------- what a source gives the arms for one query


@dataclass
class View:
    """One query's LLM step, parsed: the hypothesis, the alternatives, the sub-questions, the rewrite and the grades
    (unit -> "yes" or "no"; an unusable grade is "yes"). None or empty: the arm behaves as its parent."""
    hypothesis: str | None = None
    alternatives: list[str] = field(default_factory=list)
    subs: list[str] = field(default_factory=list)
    rewrite: str | None = None
    grades: dict[str, str] = field(default_factory=dict)


class CacheSource:
    """A model's merged cache (bench/llm_query/<dataset>.<tag>.jsonl.gz): calls are found by their cache key, rebuilt
    from the model's digest, the options and the prompt (the template filled with the query and, for the grader, the
    unit's passage), so an entry of another model or another prompt can never answer."""

    def __init__(self, tag: str, entries: Iterable[dict], passage: Callable[[str], str], digest: str | None = None):
        self.tag = tag
        self.digest = digest or spec.MODELS[tag]["digest"]
        self.by_key = {e["key"]: e for e in entries}
        self.passage = passage

    def key(self, kind: str, query: str, unit: str | None = None) -> str:
        prompt = spec.build_prompt(kind, query, self.passage(unit) if kind == "grade" else None)
        return spec.cache_key(self.digest, kind, prompt)

    def missing(self, query: str, units: Sequence[str]) -> list[str]:
        keys = [self.key(k, query) for k in spec.QUERY_KINDS] + [self.key("grade", query, u) for u in units]
        return [k for k in keys if k not in self.by_key]

    def view(self, query: str, units: Sequence[str]) -> View:
        def value(kind, unit=None):
            return read_call(self.by_key[self.key(kind, query, unit)], kind, query)[0]

        grades = {}
        for u in units:
            g = value("grade", u)
            grades[u] = g if g in ("yes", "no") else "yes"
        return View(hypothesis=value("hyde"), alternatives=list(value("multiquery") or []), subs=list(value("decompose") or []),
                    rewrite=value("rewrite"), grades=grades)


class HeuristicSource:
    """The keyword stand-ins (fixture only), each with its row's fallback label: heuristic_hypothesis; expand_queries
    without an LLM, its padding copies of the query dropped; heuristic_decompose behind its should_decompose gate (the
    LLM arm has no gate); classify_chunk (CORRECT or AMBIGUOUS = yes); _rewrite_query."""

    tag = spec.HEURISTIC_TAG

    def __init__(self, chunk_of: Callable[[str], Any]):
        self.chunk_of = chunk_of

    def view(self, query: str, units: Sequence[str]) -> View:
        from app.agent.corrective import RelevanceLabel, _rewrite_query, classify_chunk
        from app.agent.decompose import heuristic_decompose, should_decompose
        from app.agent.deep_research import expand_queries
        from app.retrieval.hyde import heuristic_hypothesis

        q = query.strip().lower()
        alts = []
        for a in expand_queries(query, n=3, llm=None)[1:]:
            if a.strip().lower() != q and a.strip().lower() not in {x.lower() for x in alts}:
                alts.append(a)
        subs = []
        if should_decompose(query):
            found = heuristic_decompose(query)
            if len(found) > 1:
                for s in found:
                    if s.strip() and s.strip().lower() != q and s.strip().lower() not in {x.lower() for x in subs}:
                        subs.append(s)
        grades = {}
        for u in units:
            label = classify_chunk(query, self.chunk_of(u), 0.0).label
            grades[u] = "yes" if label in (RelevanceLabel.CORRECT, RelevanceLabel.AMBIGUOUS) else "no"
        hyp = heuristic_hypothesis(query) or None
        return View(hypothesis=hyp, alternatives=alts[: spec.N_ALTERNATIVES], subs=subs[: spec.MAX_SUBS], rewrite=_rewrite_query(query),
                    grades=grades)


class StoredHypothesisSource:
    """qwen2.5:7b's committed HyDE hypotheses (frontier_generations.json; no new call): HyDE only."""

    tag = spec.QWEN25_TAG

    def __init__(self, generations: dict):
        self.by_question = {i["question"]: i["text"] for i in generations["hyde"]["items"]}

    def view(self, query: str, units: Sequence[str]) -> View:
        if query not in self.by_question:
            raise KeyError("no stored qwen2.5:7b hypothesis for this question")
        text = (self.by_question[query] or "").strip()
        return View(hypothesis=text or None)


# ---------------------------------------------------------------- the wiring (pure: a backend and a view)


class Recorder:
    """Wraps a backend's rerank to record the units it scored for the current arm (input recall@20, added work)."""

    def __init__(self, backend):
        self.backend = backend
        self.units: list[str] = []
        self.pairs = 0
        self.searches = 0

    def rerank(self, query: str, units: Sequence[str]) -> list[tuple[str, float]]:
        self.units += [u for u in units if u not in self.units]
        self.pairs += len(units)
        return self.backend.rerank(query, list(units))

    def first_stage(self, query: str, query_vec=None) -> list[str]:
        self.searches += 1
        return self.backend.first_stage(query, query_vec)


def head_rerank(rec: Recorder, query: str, units: Sequence[str]) -> list[str]:
    """P1's shape: the top RERANK['top'] reranked against `query`, the tail in list order."""
    top = spec.RERANK["top"]
    return [u for u, _ in rec.rerank(query, units[:top])] + list(units[top:])


def union(lists: Sequence[Sequence[str]], k: int) -> list[str]:
    """RRF (k 60) over ranked unit lists, descending, ties in first-appearance order (stable), truncated to k."""
    from app.retrieval.fuse import rrf_score_map

    fused = rrf_score_map([list(x) for x in lists], k=spec.RRF_K)
    return [u for u, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)][:k]


def per_sub(rec: Recorder, queries: Sequence[str], lists: Sequence[Sequence[str]], k: int) -> list[str]:
    """Each list's top 20 reranked against its own (sub-)question; the scored units by their highest score (ties in
    union order), then the rest in union order."""
    order = union(lists, 10**9)
    position = {u: i for i, u in enumerate(order)}
    best: dict[str, float] = {}
    for q, units in zip(queries, lists):
        for u, s in rec.rerank(q, list(units[: spec.RERANK["top"]])):
            if u not in best or s > best[u]:
                best[u] = s
    scored = sorted(best, key=lambda u: (-best[u], position[u]))
    return (scored + [u for u in order if u not in best])[:k]


def hyde_resort(p1: Sequence[str], hyde_dense: Sequence[str]) -> list[str]:
    """P1's list re-sorted by RRF with the HyDE dense list (harness.py's after_rerank): no unit is added."""
    from app.retrieval.fuse import rrf_score_map

    boosted = rrf_score_map([list(p1), list(hyde_dense)], k=spec.RRF_K)
    return sorted(p1, key=lambda u: boosted[u], reverse=True)


def corrective(rec: Recorder, query: str, parent: Sequence[str], view: View, rerun: Callable[[str], list[str]], k: int) -> tuple[list[str], dict]:
    """CRAG-style, one round, at rank level (llm_query_spec.G)."""
    top = list(parent[: spec.G])
    labels = {u: view.grades[u] for u in top}  # every unit of the top G was graded (the worklist's candidates)
    yes = [u for u in top if labels[u] == "yes"]
    no = [u for u in top if labels[u] == "no"]
    out = yes + list(parent[spec.G:]) + no
    rewritten = False
    rw = (view.rewrite or "").strip()
    if not yes and rw and rw.lower() != query.strip().lower():
        new = rerun(rw)
        out = [u for u in new if u not in top] + top
        rewritten = True
    return out[:k], {"demoted": len(no), "rewritten": rewritten}


def arm_lists(backend, query: str, view: View, p0: list[str], p1: list[str], *, placements: Sequence[tuple[str, str]] | None = None) -> dict[tuple[str, str], dict[str, Any]]:
    """Every arm of one source for one query: (technique, placement) -> {"units" (before the BEIR id filter, at most
    k), "input" (the units the reranker scored), "searches" and "pairs" (the first-stage searches, a dense-only search
    included, and the reranked pairs the arm's whole pipeline runs, its parent's own included), "fired"}. P0 and P1
    are the parent lists (P0 is one search; P1 adds its reranked top 20)."""
    k = backend.k
    wanted = set(placements) if placements is not None else {(t, p) for t, p, _, _ in spec.ARMS}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    memo: dict[str, list[str]] = {}
    p1_pairs = min(spec.RERANK["top"], len(p0))

    def fs(rec: Recorder, q: str) -> list[str]:  # the first stage of a text (the query's own is P0)
        if q == query:
            rec.searches += 1
            return list(p0)
        if q not in memo:
            memo[q] = rec.first_stage(q)
        else:
            rec.searches += 1  # the work is counted per arm even when this run reuses the list
        return list(memo[q])

    def with_p1(rec: Recorder) -> Recorder:  # an arm that starts from P1's list runs P0's search and P1's rerank
        rec.units, rec.searches, rec.pairs = list(p0[: spec.RERANK["top"]]), rec.searches + 1, rec.pairs + p1_pairs
        return rec

    def done(key, rec, units, fired):
        out[key] = {"units": list(units)[:k], "input": list(rec.units) if spec.reranked(key[1]) else list(units)[: spec.RERANK["top"]],
                    "searches": rec.searches, "pairs": rec.pairs, "fired": fired}

    vec = backend.hyde_vector(view.hypothesis, query) if view.hypothesis and any(t == "hyde" for t, _ in wanted) else None
    hyde_first = None
    if ("hyde", "none") in wanted or ("hyde", "pre") in wanted:
        rec = Recorder(backend)
        hyde_first = rec.first_stage(query, vec) if vec is not None else fs(rec, query)
        if ("hyde", "none") in wanted:
            done(("hyde", "none"), rec, hyde_first, {"hypothesis": vec is not None})
        if ("hyde", "pre") in wanted:
            rec2 = Recorder(backend)
            rec2.searches = rec.searches
            done(("hyde", "pre"), rec2, head_rerank(rec2, query, hyde_first), {"hypothesis": vec is not None})
    if ("hyde", "post") in wanted:
        rec = with_p1(Recorder(backend))
        rec.searches += int(vec is not None)  # the HyDE dense search
        units = hyde_resort(p1, backend.dense(vec)) if vec is not None else list(p1)
        done(("hyde", "post"), rec, units, {"hypothesis": vec is not None})
    for technique, texts in (("multiquery", view.alternatives), ("decompose", view.subs)):
        if not any(t == technique for t, _ in wanted):
            continue
        rec = Recorder(backend)
        lists = [fs(rec, query)] + [fs(rec, x) for x in texts]
        merged = union(lists, k)
        fired = {"lists": len(lists), "more_than_one_sub": len(texts) > 1} if technique == "decompose" else {"lists": len(lists)}
        if (technique, "none") in wanted:
            done((technique, "none"), rec, merged, fired)
        pre = "pre" if technique == "multiquery" else "union"
        if (technique, pre) in wanted:
            r2 = Recorder(backend)
            r2.searches = rec.searches
            done((technique, pre), r2, head_rerank(r2, query, merged), fired)
        if technique == "decompose" and ("decompose", "per-sub") in wanted:
            r3 = Recorder(backend)
            r3.searches = rec.searches
            done(("decompose", "per-sub"), r3, per_sub(r3, [query, *texts], lists, k), fired)
    if ("corrective", "none") in wanted:
        rec = Recorder(backend)
        rec.searches += 1  # P0
        units, fired = corrective(rec, query, p0, view, lambda rw: fs(rec, rw), k)
        done(("corrective", "none"), rec, units, fired)
    if ("corrective", "post") in wanted:
        rec = with_p1(Recorder(backend))
        units, fired = corrective(rec, query, p1, view, lambda rw: head_rerank(rec, query, fs(rec, rw)), k)
        done(("corrective", "post"), rec, units, fired)
    return out


def parents(backend, query: str) -> tuple[list[str], list[str]]:
    """(P0, P1) of one query, before the BEIR id filter."""
    p0 = backend.first_stage(query, None)
    return p0, head_rerank(Recorder(backend), query, p0)


# ---------------------------------------------------------------- the reranker-score cache (resumable, per GPU type)


class RerankCache:
    """(sha256 of the query text, unit, reranker revision, GPU name) -> score, appended to a JSONL file as computed.
    Missing pairs of one call are scored together, in list order (one batch, as the reranker scores P1's top 20)."""

    def __init__(self, path: Path | None, revision: str, gpu: str):
        self.path, self.revision, self.gpu = path, revision, gpu
        self.scores: dict[tuple[str, str], float] = {}
        self.hits = self.computed = 0
        if path is not None and path.exists():
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue  # a line cut by a killed job
                    if e.get("rev") == revision and e.get("gpu") == gpu:
                        self.scores.setdefault((e["q"], e["u"]), e["s"])

    def get(self, query: str, units: Sequence[str], score: Callable[[str, list[str]], list[float]]) -> list[float]:
        qh = hashlib.sha256(query.encode("utf-8")).hexdigest()
        todo = [u for u in dict.fromkeys(units) if (qh, u) not in self.scores]
        self.hits += len(units) - len(todo)
        if todo:
            fresh = score(query, todo)
            self.computed += len(todo)
            lines = []
            for u, s in zip(todo, fresh):
                self.scores[(qh, u)] = float(s)
                lines.append(json.dumps({"q": qh, "u": u, "rev": self.revision, "gpu": self.gpu, "s": float(s)}))
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + "\n")
                    fh.flush()
        return [self.scores[(qh, u)] for u in units]


def _sorted_by_score(units: Sequence[str], scores: Sequence[float]) -> list[tuple[str, float]]:
    ranked = list(zip(units, scores))
    ranked.sort(key=lambda item: item[1], reverse=True)  # stable: ties keep the input order, as CrossEncoderReranker
    return ranked


# ---------------------------------------------------------------- backends


class FixtureBackend:
    """Fixture v1: the bge-small index of small_rerankers.run_fixture (reranker off), units are chunk ids."""

    k = spec.K["fixture"]

    def __init__(self, index, score_pairs: Callable[[str, list[str]], list[float]] | None, cache: RerankCache | None = None):
        self.index = index
        self.by_id = index.by_id
        self.score_pairs = score_pairs
        self.cache = cache

    def first_stage(self, query: str, query_vec=None) -> list[str]:
        return [h.chunk.chunk_id for h in self.index.retrieve(query, k=self.k, query_vec=query_vec).hits]

    def dense(self, query_vec) -> list[str]:
        return [c.chunk_id for c, _ in self.index.search_dense("", k=self.k, query_vec=query_vec)]

    def hyde_vector(self, hypothesis: str, query: str):
        return _hyde_vector(self.index.embedder, hypothesis, query)

    def rerank(self, query: str, units: list[str]) -> list[tuple[str, float]]:
        if not units:
            return []
        scores = self.cache.get(query, units, self.score_pairs) if self.cache else self.score_pairs(query, units)
        return _sorted_by_score(units, scores)

    def finish(self, qid: str, units: Sequence[str]) -> list[str]:
        return list(units)

    def passage(self, unit: str) -> str:
        c = self.by_id[unit]
        return spec.fixture_passage(c.path, c.text)

    def text(self, unit: str) -> str:
        return self.by_id[unit].text

    def paths(self, units: Sequence[str]) -> list[str]:
        out: list[str] = []
        for u in units:
            p = self.by_id[u].path
            if p not in out:
                out.append(p)
        return out


def _hyde_vector(embedder, hypothesis: str, query: str):
    """normalise(mean(embed([hypothesis, query]))): hyde_embed's formula, one encode call of the two texts."""
    import numpy as np

    vecs = embedder.encode([hypothesis, query])
    mean = vecs.mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean


class BeirBackend:
    """A BEIR corpus through beir.Engine at the committed rows' settings: BM25 with the "beir" analyzer (k1 0.9, b 0.4),
    the bge-small unit matrix, RRF k 60 over the top 100 of each leg; units are document ids."""

    k = spec.K["beir"]

    def __init__(self, engine, corpus: dict[str, str], score_pairs: Callable[[str, list[str]], list[float]] | None,
                 cache: RerankCache | None = None, embedder: str = "st:BAAI/bge-small-en-v1.5"):
        self.engine, self.corpus, self.score_pairs, self.cache, self.embedder_name = engine, corpus, score_pairs, cache, embedder

    def _dense_ids(self, qv) -> list[str]:
        from app.retrieval.hybrid import dense_scores, rank_by_score

        scores = dense_scores(qv, self.engine.unit_matrix(self.embedder_name), unit=True)
        return [self.engine.chunks[i].chunk_id for i in rank_by_score(scores, self.k)]

    def query_vector(self, query: str):
        from app.retrieval.embedders import encode_queries

        return encode_queries(self.engine.embedder(self.embedder_name), [query])[0]

    def first_stage(self, query: str, query_vec=None) -> list[str]:
        """beir.retrieve's hybrid branch with the reranker off; `query_vec` replaces the dense leg's query vector."""
        from app.retrieval.fuse import rrf_score_map

        bm25_ids = [d for d, _ in self.engine.bm25("beir", 0.9, 0.4).search(query, k=self.k)]
        dense_ids = self._dense_ids(self.query_vector(query) if query_vec is None else query_vec)
        fused = rrf_score_map([bm25_ids, dense_ids], k=spec.RRF_K)
        return [d for d, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[: self.k]]

    def dense(self, query_vec) -> list[str]:
        return self._dense_ids(query_vec)

    def dense_row(self, query: str) -> list[str]:
        return self._dense_ids(self.query_vector(query))

    def hyde_vector(self, hypothesis: str, query: str):
        return _hyde_vector(self.engine.embedder(self.embedder_name), hypothesis, query)

    def rerank(self, query: str, units: list[str]) -> list[tuple[str, float]]:
        if not units:
            return []
        scores = self.cache.get(query, units, self.score_pairs) if self.cache else self.score_pairs(query, units)
        return _sorted_by_score(units, scores)

    def finish(self, qid: str, units: Sequence[str]) -> list[str]:
        return [d for d in units if d != qid]  # BEIR's ignore_identical_ids, as beir.run_config

    def passage(self, unit: str) -> str:
        return spec.beir_passage(self.corpus[unit])

    def text(self, unit: str) -> str:
        return self.corpus[unit]


# ---------------------------------------------------------------- one dataset's rows (rankings only: no qrels)


ARM_KINDS = ("arm", "stand-in", "qwen2.5-7b", "stretch", "stretch-reference")


def row_specs(models: Sequence[str], *, fixture: bool, beir: bool, stretch: bool = False) -> list[dict[str, Any]]:
    """Every row of a run, in table order: P0, P1, (BEIR) dense@bge-small, then each source's arms. In a stretch run
    qwen3.5:27b's arms are recomputed beside qwen3.5:122b's as references only (their bar cells are the main files')."""
    rows = [{"config": spec.P0, "kind": "parent"}, {"config": spec.P1, "kind": "parent", "parent": spec.P0}]
    if beir:
        rows.append({"config": spec.DENSE, "kind": "reference"})
    tags = list(models)
    if fixture and not stretch:
        tags += [spec.HEURISTIC_TAG, spec.QWEN25_TAG]
    for tag in tags:
        for technique, placement, _, _ in spec.ARMS:
            if tag == spec.QWEN25_TAG and technique != "hyde":
                continue
            kind = ("stand-in" if tag == spec.HEURISTIC_TAG else "qwen2.5-7b" if tag == spec.QWEN25_TAG
                    else "stretch" if tag == spec.STRETCH_MODEL else "stretch-reference" if stretch else "arm")
            rows.append({"config": spec.arm_name(technique, placement, tag), "kind": kind, "model": tag, "technique": technique,
                         "placement": placement, "parent": spec.parent_name(technique, placement)})
    return rows


def comparison_pairs(rows: Sequence[dict], *, beir: bool) -> list[tuple[str, str, str]]:
    """(row, other, kind) of every paired interval of a run: each arm against its parent (the bar cells, or the
    stand-in, qwen2.5:7b and stretch rows' words), then the secondaries (never a decision)."""
    names = {r["config"] for r in rows}
    pairs: list[tuple[str, str, str]] = [(spec.P1, spec.P0, "reference")]
    for r in rows:
        if r["kind"] in ("arm", "stand-in", "qwen2.5-7b", "stretch"):
            pairs.append((r["config"], r["parent"], r["kind"]))
    arm_rows = [r for r in rows if r["kind"] in ("arm", "stretch")]
    for r in arm_rows:
        t, p, m = r["technique"], r["placement"], r["model"]
        if r["kind"] == "stretch":
            other = spec.arm_name(t, p, spec.MAIN_MODEL)
            if other in names:
                pairs.append((r["config"], other, "stretch"))
            continue
        h = spec.arm_name(t, p, spec.HEURISTIC_TAG)
        if h in names:
            pairs.append((r["config"], h, "secondary:llm-minus-heuristic"))
        if m == spec.REPLICATION_MODEL and spec.arm_name(t, p, spec.MAIN_MODEL) in names:
            pairs.append((r["config"], spec.arm_name(t, p, spec.MAIN_MODEL), "secondary:gemma-minus-qwen"))
        if (t, p) == ("hyde", "pre"):
            pairs.append((r["config"], spec.arm_name("hyde", "post", m), "secondary:pre-minus-post"))
        if (t, p) == ("decompose", "per-sub"):
            pairs.append((r["config"], spec.arm_name("decompose", "union", m), "secondary:per-sub-minus-union"))
        if p == "none":
            pairs.append((r["config"], spec.P1, "secondary:no-rerank-minus-P1"))
            if beir:
                pairs.append((r["config"], spec.DENSE, "secondary:no-rerank-minus-dense"))
    for r in rows:
        if r["kind"] == "qwen2.5-7b" and spec.arm_name(r["technique"], r["placement"], spec.HEURISTIC_TAG) in names:
            pairs.append((r["config"], spec.arm_name(r["technique"], r["placement"], spec.HEURISTIC_TAG), "qwen2.5-7b"))
    return [(a, b, kind) for a, b, kind in pairs if a in names and b in names]


def rank_dataset(backend, queries: Sequence[tuple[str, str]], sources: dict[str, Any], *, beir: bool,
                 expected: dict[str, dict] | None = None, progress: Callable[[int], None] | None = None) -> dict[str, Any]:
    """Every row's ranked units per query (after the BEIR id filter), with the reranker's inputs and the fired counts.
    `sources`: tag -> a source with view(query, grader units). `expected`: qid -> {p0_sha256, p1_sha256} (the
    worklist's): a query whose parent ranking differs stops the run. Reads no qrels."""
    rows = {spec.P0: [], spec.P1: []}
    if beir:
        rows[spec.DENSE] = []
    inputs = {spec.P0: [], spec.P1: []}
    if beir:
        inputs[spec.DENSE] = []
    fired: dict[str, list[dict]] = {}
    work: dict[str, list[tuple[int, int]]] = {}
    ms: dict[str, list[float]] = {}
    parent_lists: dict[str, dict[str, Any]] = {}
    drift = []
    for n, (qid, query) in enumerate(queries):
        p0, p1 = parents(backend, query)
        f0, f1 = backend.finish(qid, p0), backend.finish(qid, p1)
        parent_lists[qid] = {"p0_sha256": list_sha256(f0), "p1_sha256": list_sha256(f1), "p0_top": p0[: spec.G], "p1_top": p1[: spec.G]}
        if expected is not None:
            want = expected.get(qid) or {}
            if any(want.get(key) != parent_lists[qid][key] for key in ("p0_sha256", "p1_sha256", "p0_top", "p1_top")):
                drift.append(qid)
                continue
        rows[spec.P0].append(f0)
        rows[spec.P1].append(f1)
        inputs[spec.P0].append(p0[: spec.RERANK["top"]])
        inputs[spec.P1].append(p0[: spec.RERANK["top"]])
        if beir:
            d = backend.dense_row(query)
            rows[spec.DENSE].append(backend.finish(qid, d))
            inputs[spec.DENSE].append(d[: spec.RERANK["top"]])
        graded = spec.grader_candidates(p0, p1)
        for tag, source in sources.items():
            view = source.view(query, graded)
            placements = [(t, p) for t, p, _, _ in spec.ARMS if tag != spec.QWEN25_TAG or t == "hyde"]
            t0 = time.perf_counter()
            arms = arm_lists(backend, query, view, p0, p1, placements=placements)
            elapsed = (time.perf_counter() - t0) * 1000.0 / max(1, len(arms))
            for (t, p), a in arms.items():
                name = spec.arm_name(t, p, tag)
                rows.setdefault(name, []).append(backend.finish(qid, a["units"]))
                inputs.setdefault(name, []).append(a["input"])
                fired.setdefault(name, []).append(a["fired"])
                work.setdefault(name, []).append((a["searches"], a["pairs"]))
                ms.setdefault(name, []).append(elapsed)
        if progress:
            progress(n)
    if drift:
        raise ParentDrift(drift, parent_lists)
    return {"rows": rows, "inputs": inputs, "fired": fired, "work": work, "ms": ms, "parents": parent_lists}


class ParentDrift(RuntimeError):
    """The scoring run's P0 or P1 ranking of some queries differs from the worklist's: the run stops (Leakage risk 5)."""

    def __init__(self, qids: list[str], parent_lists: dict):
        super().__init__(f"{len(qids)} queries rank P0 or P1 differently from the worklist: {qids[:20]}")
        self.qids, self.parent_lists = qids, parent_lists


def fired_summary(name: str, parent: str, ranked: dict[str, Any]) -> dict[str, Any]:
    """Fired counts of one arm row: lists changed vs the parent, top-10 differs, and per technique the sub-question,
    demotion and rewrite counts; the added work (first-stage searches and reranked pairs per query)."""
    import numpy as np

    rows, base = ranked["rows"][name], ranked["rows"][parent]
    fired = ranked["fired"].get(name, [])
    out: dict[str, Any] = {"n_queries": len(rows), "lists_changed": sum(1 for a, b in zip(rows, base) if a != b),
                           "top10_differs": sum(1 for a, b in zip(rows, base) if a[:10] != b[:10])}
    if fired and "more_than_one_sub" in fired[0]:
        out["more_than_one_sub"] = sum(1 for f in fired if f["more_than_one_sub"])
    if fired and "lists" in fired[0]:
        out["queries_with_extra_lists"] = sum(1 for f in fired if f["lists"] > 1)
    if fired and "demoted" in fired[0]:
        out["queries_with_demotions"] = sum(1 for f in fired if f["demoted"])
        out["units_demoted"] = sum(f["demoted"] for f in fired)
        out["rewrites"] = sum(1 for f in fired if f["rewritten"])
    if fired and "hypothesis" in fired[0]:
        out["queries_with_hypothesis"] = sum(1 for f in fired if f["hypothesis"])
    work = ranked["work"].get(name) or []
    if work:
        searches, pairs = float(np.mean([w[0] for w in work])), float(np.mean([w[1] for w in work]))
        p_pairs = float(np.mean([min(spec.RERANK["top"], len(x)) for x in ranked["inputs"][spec.P1]])) if parent == spec.P1 else 0.0
        out["work"] = {"first_stage_searches_per_query": round(searches, 3), "reranked_pairs_per_query": round(pairs, 3),
                       "added_vs_parent": {"first_stage_searches_per_query": round(searches - 1.0, 3), "reranked_pairs_per_query": round(pairs - p_pairs, 3)},
                       "note": "the arm's whole pipeline per query (a dense-only search counts as a search); P0 is one search, P1 adds "
                               "its reranked top 20"}
    times = ranked["ms"].get(name) or []
    if times:
        out["ranking_ms"] = {"p50": round(float(np.percentile(times, 50)), 2), "p95": round(float(np.percentile(times, 95)), 2),
                             "note": "indicative: the source's arms per query divided evenly, LLM output from the cache, reranker "
                                     "scores partly from the on-disk score cache"}
    return out


# ---------------------------------------------------------------- the LLM and cost blocks (from bench/llm_query alone)


def load_cache(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_parents(root: Path) -> dict[str, Any]:
    path = root / PARENTS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{PARENTS_FILE} is missing (make llm-query-merge writes it)")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


_KIND_NAMES = {"hyde": "HyDE", "multiquery": "multi-query", "decompose": "decomposition", "rewrite": "rewrite", "grade": "grader"}


def llm_block(technique: str, placement: str, queries: Sequence[dict], by_call: dict[tuple, dict]) -> dict[str, Any]:
    """The llm counts, the cost and the LLM fallback labels of one arm of one model, from the cached calls it reads:
    the technique's query call, or (corrective) the grades of the parent's top G and, when none is yes, the rewrite."""
    import numpy as np

    counts = {s: 0 for s in ("calls", "failed", "empty", "unparseable", "unchanged", "truncated", "overflowed", "thinking")}
    per_query_ms, prompt_tok, gen_tok, gpus = [], 0, 0, set()
    parent_behaviour: dict[str, int] = {}
    grade_yes_fallback: dict[str, int] = {}
    no_rewrite: dict[str, int] = {}
    rewrites_read = 0
    for q in queries:
        calls = []
        if technique == "corrective":
            top = q["p0_top"] if placement == "none" else q["p1_top"]
            grades = []
            for u in top:
                e = by_call[("grade", q["id"], u)]
                value, status, trunc = read_call(e, "grade", q["text"])
                calls.append((e, status, trunc))
                grades.append(value)
                if status != "ok" or value not in ("yes", "no"):
                    grade_yes_fallback[status] = grade_yes_fallback.get(status, 0) + 1
            if not any(g in ("yes", None) for g in grades) and grades:
                e = by_call[("rewrite", q["id"], None)]
                value, status, trunc = read_call(e, "rewrite", q["text"])
                calls.append((e, status, trunc))
                rewrites_read += 1
                if value is None:
                    no_rewrite[status] = no_rewrite.get(status, 0) + 1
                elif value.strip().lower() == q["text"].strip().lower():
                    no_rewrite["equal to the query"] = no_rewrite.get("equal to the query", 0) + 1
        else:
            e = by_call[(technique, q["id"], None)]
            value, status, trunc = read_call(e, technique, q["text"])
            calls.append((e, status, trunc))
            if not value:
                parent_behaviour[status] = parent_behaviour.get(status, 0) + 1
        ms = 0.0
        for e, status, trunc in calls:
            counts["calls"] += 1
            if status in counts:
                counts[status] += 1
            counts["truncated"] += int(trunc)
            counts["thinking"] += int((e.get("thinking_chars") or 0) > 0)
            prompt_tok += int(e.get("prompt_eval_count") or 0)
            gen_tok += int(e.get("eval_count") or 0)
            ms += float(e.get("wall_ms") or 0.0)
            if e.get("gpu"):
                gpus.add(e["gpu"])
        per_query_ms.append(ms)
    n = len(queries)
    labels = []
    name = _KIND_NAMES[technique if technique != "corrective" else "grade"]
    for status, c in sorted(parent_behaviour.items()):
        if status == "unchanged":
            labels.append(f"parent behaviour on {c} of {n} queries (the {name} reply repeated the query)")
        else:
            labels.append(f"parent behaviour on {c} of {n} queries ({status} {name} reply)")
    for status, c in sorted(grade_yes_fallback.items()):
        labels.append(f"grade counted as yes for {c} of {n * spec.G} graded units ({status} grader reply)")
    for status, c in sorted(no_rewrite.items()):
        why = "the rewrite equals the query" if status == "equal to the query" else f"{status} rewrite reply"
        labels.append(f"no rewrite on {c} of {rewrites_read} queries with no yes ({why})")
    if counts["truncated"]:
        labels.append(f"{counts['truncated']} of {counts['calls']} calls truncated (the declared truncation rule)")
    if counts["thinking"]:
        labels.append(f"{counts['thinking']} of {counts['calls']} calls returned thinking text")
    cost = {"llm_calls": counts["calls"], "llm_calls_per_query": round(counts["calls"] / n, 4) if n else None,
            "prompt_tokens_per_query": round(prompt_tok / n, 2) if n else None,
            "generated_tokens_per_query": round(gen_tok / n, 2) if n else None,
            "llm_ms_per_query": ({"p50": round(float(np.percentile(per_query_ms, 50)), 1), "p95": round(float(np.percentile(per_query_ms, 95)), 1),
                                  "mean": round(float(np.mean(per_query_ms)), 1)} if per_query_ms else None),
            "gpu": sorted(gpus),
            "note": "the sum of the cached calls the arm reads per query, sequential, client wall time on the GPU named"}
    return {"llm_counts": counts, "cost": cost, "llm_fallbacks": labels}


def llm_blocks_from_cache(root: Path, blob: dict) -> dict[str, dict[str, Any]]:
    """Every LLM arm row's llm counts, cost and LLM fallback labels, from bench/llm_query (no model, no dataset):
    the merged caches, and parents.json.gz (each query's text and P0/P1 top G, from the worklist)."""
    dataset = blob["dataset"]
    parents_all = load_parents(root)
    queries = parents_all[dataset]["queries"]
    out: dict[str, dict[str, Any]] = {}
    tags = sorted({r["model"] for r in blob.get("rows") or [] if r.get("model") in spec.MODELS})
    for tag in tags:
        path = root / cache_file(dataset, tag)
        if not path.exists():
            raise FileNotFoundError(f"{cache_file(dataset, tag)} is missing")
        by_call = {(e["kind"], e["qid"], e.get("unit")): e for e in load_cache(path)}
        for technique, placement, _, _ in spec.ARMS:
            name = spec.arm_name(technique, placement, tag)
            if any(r["config"] == name for r in blob.get("rows") or []):
                out[name] = llm_block(technique, placement, queries, by_call)
    return out


# ---------------------------------------------------------------- scoring (the qrels readers)


def fixture_scores(backend: FixtureBackend, gold, ranked: dict[str, Any], n_files: int) -> dict[str, dict[str, list[float]]]:
    """Per row, per metric, one score per gold question (file paths of the ranked chunks), plus input recall@20."""
    from app.eval.metrics import aggregate_retrieval

    out = {}
    for name, lists in ranked["rows"].items():
        rows = [{"id": q.id, "category": q.category, "relevant": q.relevant_ids(), "retrieved": backend.paths(units)} for q, units in zip(gold, lists)]
        per = aggregate_retrieval(rows, n_files=n_files).per_query
        scores = {m: [float(p[m]) for p in per] for m in FIXTURE_METRICS if m != "input_recall@20"}
        scores["input_recall@20"] = [_recall(backend.paths(inp), q.relevant_ids()) for q, inp in zip(gold, ranked["inputs"][name])]
        out[name] = scores
    return out


def _recall(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    rel = set(relevant)
    return len(rel & set(retrieved)) / len(rel) if rel else 0.0


def beir_scores(qids: Sequence[str], qrels: dict[str, dict[str, int]], ranked: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    from app.eval.beir import graded_ndcg_at_k
    from app.eval.metrics import mrr_at_k, recall_at_k

    out = {}
    for name, lists in ranked["rows"].items():
        cols = {m: [] for m in BEIR_METRICS}
        for qid, ids, inp in zip(qids, lists, ranked["inputs"][name]):
            relevant = list(qrels[qid])
            cols["ndcg@10"].append(graded_ndcg_at_k(ids, qrels[qid], 10))
            cols["recall@10"].append(recall_at_k(ids, relevant, 10))
            cols["recall@100"].append(recall_at_k(ids, relevant, 100))
            cols["mrr@10"].append(mrr_at_k(ids, relevant, 10))
            cols["input_recall@20"].append(_recall(inp, relevant))
        out[name] = {m: [float(x) for x in v] for m, v in cols.items()}
    return out


def trec_sha256(name: str, qids: Sequence[str], lists: Sequence[Sequence[str]]) -> str:
    """The sha256 of the TREC run file beir.write_trec_outputs would write for this row (its ranking_sha256)."""
    from app.eval.beir import BeirConfig, trec_run_text

    return hashlib.sha256(trec_run_text(BeirConfig(name=name, mode="hybrid"), dict(zip(qids, lists))).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- reproduction (declared reading, Leakage risk 5)


def reproduction_status(checks: list[dict[str, Any]]) -> dict[str, Any]:
    """'reproduces' (every per-query score and every available hash equal); 'reproduces in every metric; near-tied units
    ordered differently in N queries' (scores equal, a hash differs); 'failed' (a per-query score or a committed mean
    differs): that dataset's cells are then withheld until it is explained."""
    scores_equal = all(c.get("scores_equal", True) and c.get("means_equal", True) for c in checks)
    hashes_equal = all(c.get("ranking_equal", True) for c in checks)
    if not scores_equal:
        return {"status": "failed", "checks": checks}
    if not hashes_equal:
        n = max(int(c.get("queries_ordered_differently") or 0) for c in checks)  # the same queries differ in every file that has the run
        known = all(c.get("queries_ordered_differently") is not None for c in checks if not c.get("ranking_equal", True))
        return {"status": f"reproduces in every metric; near-tied units ordered differently in {n if known else 'an unknown number of'} queries",
                "checks": checks}
    return {"status": "reproduces", "checks": checks}


def _committed_rows(root: Path, rel: str) -> dict[str, dict]:
    path = root / rel
    return {r["config"]: r for r in json.loads(path.read_text()).get("results", [])} if path.exists() else {}


def _sidecar_scores(root: Path, rel: str) -> dict | None:
    from app.eval import per_question

    side = per_question.sidecar_path(root / rel)
    return json.loads(side.read_text()) if side.exists() else None


def check_scores(name: str, rel: str, mine: dict[str, list[float]], ids: Sequence[str], sidecar: dict | None, committed_name: str | None = None) -> dict:
    """Per-query scores of a recomputed row against a committed sidecar's unrounded scores (same ids, same order)."""
    committed_name = committed_name or name
    if sidecar is None or committed_name not in sidecar.get("scores", {}):
        return {"row": name, "file": rel, "per_query": "not available"}
    if list(sidecar["ids"]) != list(ids):
        return {"row": name, "file": rel, "scores_equal": False, "differs": "query ids or order differ"}
    differs = {}
    for m in sidecar["metrics"]:
        if m in mine and list(sidecar["scores"][committed_name][m]) != list(mine[m]):
            differs[m] = sum(1 for a, b in zip(sidecar["scores"][committed_name][m], mine[m]) if a != b)
    return {"row": name, "file": rel, "compared_as": committed_name, "scores_equal": not differs, **({"queries_differ": differs} if differs else {})}


# ---------------------------------------------------------------- results


def _git_info(root: Path) -> dict[str, Any]:
    from app.eval.provenance import git_sha

    return {"git_sha": git_sha(root)}


def declaration(root: Path) -> dict[str, Any]:
    """D0 and D1 (bench/llm_query/declaration.json, written after D1) and the attempt."""
    path = root / LLM_DIR / "declaration.json"
    block = json.loads(path.read_text()) if path.exists() else {"D0": None, "D1": None, "note": "declaration.json missing"}
    return {**block, "attempt": spec.ATTEMPT}


def build_results(*, dataset: str, rows: list[dict], ranked: dict[str, Any], scores: dict[str, dict[str, list[float]]], metrics: Sequence[str],
                  scopes: dict[str, list[int]], ids: Sequence[str], header: dict[str, Any], reproduction: dict[str, Any],
                  llm: dict[str, dict], categories: Sequence[str] | None = None) -> tuple[dict, dict]:
    """(the results blob, its per-query sidecar). Every interval goes through llm_query_bar.scope_blocks."""
    names = [r["config"] for r in rows if r["config"] in scores]
    pairs = comparison_pairs([r for r in rows if r["config"] in scores], beir=dataset != "fixture")
    blob: dict[str, Any] = {"schema": bar.SCHEMA, "dataset": dataset, **header}
    blob["reproduction"] = reproduction
    out_rows = []
    for r in rows:
        if r["config"] not in scores:
            continue
        row = dict(r)
        if r["config"] in ranked["rows"]:
            lists = ranked["rows"][r["config"]]
            row["ranking_sha256"] = trec_sha256(r["config"], ids, lists)
        if r.get("kind") in ARM_KINDS:
            row["fired"] = fired_summary(r["config"], r["parent"], ranked)
        fallbacks = list(row.pop("fallbacks", []))
        if r["config"] in llm:
            row.update(llm[r["config"]])
            fallbacks += llm[r["config"]]["llm_fallbacks"]
        row["fallbacks"] = fallbacks
        out_rows.append(row)
    blob["rows"] = out_rows
    blob["scopes"] = {}
    for scope_name, subset in scopes.items():
        blob["scopes"][scope_name] = bar.scope_blocks(scores, subset, names, pairs, metrics)
    blob["n_intervals"] = bar.count_intervals(blob)
    blob["does_not_show"] = list(spec.DOES_NOT_SHOW)
    sidecar = {"schema": bar.SIDECAR_SCHEMA,
               "about": "per-query scores of the results file with the same name (.json), unrounded: `make llm-query-arms-rescore` "
                        "recomputes every interval, mean and verdict of its scopes from them",
               "git_sha": blob.get("git_sha"), "timestamp": blob.get("timestamp"), "dataset": dataset, "metrics": list(metrics),
               "ids": list(ids), **({"categories": list(categories)} if categories is not None else {}),
               "scopes": {k: list(v) for k, v in scopes.items()}, "scores": {n: scores[n] for n in names}}
    return blob, sidecar


def heuristic_fallbacks(technique: str) -> list[str]:
    from app.eval.harness import fallback_labels

    flag = {"hyde": "hyde", "multiquery": "multi_query", "decompose": "decompose", "corrective": "corrective"}[technique]
    return fallback_labels({"settings": {flag: True}}) + [spec.HEURISTIC_LABEL]


# ---------------------------------------------------------------- the runs


def score_fixture(root: Path, *, models: Sequence[str], load_reranker=None, embedder: str | None = None, rerank_cache: Path | None = None,
                  gpu: str = "cpu", stretch: bool = False, llm_dir: Path | None = None, reproduce_only: bool = False) -> tuple[dict, dict]:
    """Fixture v1: every row on all 136 questions, scored on the held-out half (the bar), the tune half and all 136.
    `load_reranker` (tests) -> (score_pairs(query, texts), revision). reproduce_only: P0 and P1 alone, against the
    committed rows (no cache, no arm); returns ({"reproduction": ...}, {})."""
    from datetime import datetime, timezone

    from app.eval.corpus import load_manifest, verify_manifest
    from app.eval.frontier_llm import load_generations
    from app.eval.gold import load_gold
    from app.eval.harness import BGE_SMALL
    from app.eval.index_loader import build_index
    from app.eval.provenance import command_line, device_info, git_state, model_provenance, run_provenance, sha256_file
    from app.eval.rerank_arms import split_questions

    started = git_state(root)
    files_root = root / "bench" / "corpus" / "files"
    gold_path = root / "bench" / "gold" / "questions.jsonl"
    gold = load_gold(gold_path)
    manifest = load_manifest(root / "bench" / "corpus" / "MANIFEST.json")
    mismatches = verify_manifest(files_root, manifest)
    if mismatches:
        raise RuntimeError("corpus drift:\n" + "\n".join(mismatches))
    embedder = embedder or BGE_SMALL
    _check_embedder_revision(embedder)
    index = build_index(files_root, embedder_name=embedder, retrieve_k=spec.K["fixture"], enable_router=False,
                        enable_rerank=False, mtimes="manifest")
    score_texts, revision, reranker_info = (load_reranker or _load_bge_reranker)()
    cache = RerankCache(rerank_cache, revision, gpu) if rerank_cache else None
    backend = FixtureBackend(index, lambda q, units: score_texts(q, [index.by_id[u].text for u in units]), cache)
    if reproduce_only:
        ranked = rank_dataset(backend, [(q.id, q.question) for q in gold], {}, beir=False)
        scores = fixture_scores(backend, gold, ranked, manifest["n_files"])
        return {"dataset": "fixture", "reproduce_only": True, "reproduction": _fixture_reproduction(root, gold, backend, ranked, scores, manifest["n_files"])}, {}
    parents_all = load_parents(root)["fixture"]
    expected = {q["id"]: q for q in parents_all["queries"]}
    queries = [(q.id, q.question) for q in gold]
    sources = _cache_sources(root, "fixture", models, backend.passage, llm_dir)
    if not stretch:
        sources[spec.HEURISTIC_TAG] = HeuristicSource(lambda u: index.by_id[u])
        sources[spec.QWEN25_TAG] = StoredHypothesisSource(load_generations(root / "bench" / "results" / "frontier_generations.json"))
    _stop_on_missing(sources, queries, expected)
    ranked = rank_dataset(backend, queries, sources, beir=False, expected=expected)
    scores = fixture_scores(backend, gold, ranked, manifest["n_files"])
    tune, held = split_questions(gold)
    rows = row_specs([m for m in models], fixture=True, beir=False, stretch=stretch)
    for r in rows:
        if r.get("model") == spec.HEURISTIC_TAG:
            r["fallbacks"] = heuristic_fallbacks(r["technique"])
    reproduction = _fixture_reproduction(root, gold, backend, ranked, scores, manifest["n_files"])
    header = {**_git_info(root), **run_provenance(root=root, started=started), "timestamp": datetime.now(timezone.utc).isoformat(),
              "command": command_line(), "device": device_info(), "corpus_sha256": manifest["aggregate_sha256"],
              "gold_sha256": sha256_file(gold_path), "n_queries": len(gold), "n_chunks": len(index.chunks),
              "embedder": {**model_provenance(embedder, "embedder"), **(index.embedder.provenance() if hasattr(index.embedder, "provenance") else {})},
              **_common_header(root, models, reranker_info, cache, stretch, llm_dir)}
    header["split"] = {"rule": "rerank_arms.split_questions (alternate inside each category, gold order)", "held_out": "the bar",
                       "tune": "reported; no choice made on it", "held_out_ids": [gold[i].id for i in held], "tune_ids": [gold[i].id for i in tune]}
    llm = _llm_for(root, "fixture", rows, llm_dir)
    return build_results(dataset="fixture", rows=rows, ranked=ranked, scores=scores, metrics=FIXTURE_METRICS,
                         scopes={"held_out": held, "tune": tune, "all_questions": list(range(len(gold)))}, ids=[q.id for q in gold],
                         header=header, reproduction=reproduction, llm=llm, categories=[q.category for q in gold])


def score_beir(root: Path, data_dir: Path, dataset: str, *, models: Sequence[str], load_reranker=None, rerank_cache: Path | None = None,
               gpu: str = "cpu", stretch: bool = False, llm_dir: Path | None = None, device: str = "auto", engine=None,
               corpus: dict | None = None, qrels: dict | None = None, embedder: str = "st:BAAI/bge-small-en-v1.5",
               reproduce_only: bool = False) -> tuple[dict, dict]:
    """BEIR SciFact or FiQA test: every row on every test query (query ids from bench/llm_query/query_ids.json, the
    text from queries.jsonl, the qrels read here to score)."""
    from datetime import datetime, timezone

    from app.eval import beir
    from app.eval.llm_query_gen import load_queries
    from app.eval.provenance import command_line, device_info, git_state, run_provenance

    started = git_state(root)
    _check_embedder_revision(embedder)
    ids = json.loads((root / QUERY_IDS).read_text())[dataset]["test"]
    droot = Path(data_dir) / dataset
    if corpus is None:
        corpus = beir.load_corpus(droot)
    queries = list(load_queries(droot, ids).items())
    if engine is None:
        engine = beir.Engine(beir.chunks_from_corpus(corpus), cache_dir=Path(data_dir) / "cache", device=device)
    score_texts, revision, reranker_info = (load_reranker or (lambda: _load_bge_reranker(device)))()
    cache = RerankCache(rerank_cache, revision, gpu) if rerank_cache else None
    backend = BeirBackend(engine, corpus, lambda q, units: score_texts(q, [corpus[u] for u in units]), cache, embedder=embedder)
    if reproduce_only:  # a named qrels reader: P0, P1 and dense@bge-small against the committed rows, nothing else
        ranked = rank_dataset(backend, queries, {}, beir=True)
        if qrels is None:
            _, _, qrels = beir.load_beir(droot, "test")
        qids = [q for q, _ in queries]
        return {"dataset": dataset, "reproduce_only": True,
                "reproduction": _beir_reproduction(root, dataset, qids, ranked, beir_scores(qids, qrels, ranked))}, {}
    parents_all = load_parents(root)[dataset]
    expected = {q["id"]: q for q in parents_all["queries"]}
    sources = _cache_sources(root, dataset, models, backend.passage, llm_dir)
    _stop_on_missing(sources, queries, expected)
    ranked = rank_dataset(backend, queries, sources, beir=True, expected=expected)
    if qrels is None:
        _, _, qrels = beir.load_beir(droot, "test")  # scoring: a named qrels reader
    qids = [q for q, _ in queries]
    scores = beir_scores(qids, qrels, ranked)
    rows = row_specs(models, fixture=False, beir=True, stretch=stretch)
    reproduction = _beir_reproduction(root, dataset, qids, ranked, scores)
    header = {**_git_info(root), **run_provenance(root=root, started=started), "timestamp": datetime.now(timezone.utc).isoformat(),
              "command": command_line(), "device": device_info(device), "corpus_sha256": getattr(engine, "corpus_sha256", None),
              "n_queries": len(qids), "n_docs": len(corpus), "split": "test", "document_text": "title + ' ' + text",
              "embedder": dict(getattr(engine, "provenance", {}).get(embedder) or {}),
              **_common_header(root, models, reranker_info, cache, stretch, llm_dir)}
    llm = _llm_for(root, dataset, rows, llm_dir)
    return build_results(dataset=dataset, rows=rows, ranked=ranked, scores=scores, metrics=BEIR_METRICS, scopes={"test": list(range(len(qids)))},
                         ids=qids, header=header, reproduction=reproduction, llm=llm)


def _llm_for(root: Path, dataset: str, rows: list[dict], llm_dir: Path | None) -> dict:
    if llm_dir is not None and Path(llm_dir).resolve() != (root / LLM_DIR).resolve():
        raise ValueError("scoring reads the merged caches in bench/llm_query only")
    tagged = [r for r in rows if r.get("model") in spec.MODELS]
    if not tagged:
        return {}
    return llm_blocks_from_cache(root, {"dataset": dataset, "rows": tagged})


def _cache_sources(root: Path, dataset: str, models: Sequence[str], passage, llm_dir: Path | None) -> dict[str, CacheSource]:
    sources = {}
    for tag in models:
        path = root / cache_file(dataset, tag)
        if not path.exists():
            raise FileNotFoundError(f"{cache_file(dataset, tag)} is missing: a dataset is scored only when its caches are fully merged")
        sources[tag] = CacheSource(tag, load_cache(path), passage)
    return sources


class MissingKeys(RuntimeError):
    pass


def _stop_on_missing(sources: dict[str, Any], queries: Sequence[tuple[str, str]], expected: dict[str, dict]) -> None:
    """Every key every LLM arm needs must be in its cache (grades: the worklist's P0 and P1 top G); otherwise the run
    stops before ranking and lists them."""
    missing = []
    for tag, source in sources.items():
        if not isinstance(source, CacheSource):
            continue
        for qid, text in queries:
            want = expected.get(qid)
            units = spec.grader_candidates(want["p0_top"], want["p1_top"]) if want else []
            missing += [f"{tag} {qid} {k}" for k in source.missing(text, units)]
    if missing:
        raise MissingKeys(f"{len(missing)} cache keys missing, e.g. {missing[:10]}")


def _check_embedder_revision(embedder: str) -> None:
    """bge-small's cached snapshot must be the pinned revision (another embedder is a test stand-in)."""
    from app.eval.provenance import hf_revision

    name = embedder.split(":", 1)[1] if embedder.startswith("st:") else embedder
    if name == spec.RETRIEVAL_MODELS["embedder"]["model"]:
        rev = hf_revision(name)
        if rev != spec.RETRIEVAL_MODELS["embedder"]["revision"]:
            raise RuntimeError(f"{name}: cached revision {rev}, pinned {spec.RETRIEVAL_MODELS['embedder']['revision']}")


def _load_bge_reranker(device: str = "auto"):
    """bge-reranker-v2-m3 at the served shape (fp32, max_length 512), its revision checked against the pinned one."""
    from app.eval.models_2026 import check_reranker
    from app.eval.provenance import hf_revision, model_device
    from app.retrieval.rerank import cross_encoder_provenance, cross_encoder_scores, get_cross_encoder

    rev = hf_revision(spec.RERANK["model"])
    if rev != spec.RETRIEVAL_MODELS["reranker"]["revision"]:
        raise RuntimeError(f"bge-reranker-v2-m3: cached revision {rev}, pinned {spec.RETRIEVAL_MODELS['reranker']['revision']}")
    model = get_cross_encoder(spec.RERANK["model"], precision=spec.RERANK["precision"], max_length=spec.RERANK["max_length"],
                              device=None if device == "auto" else device, raise_errors=True)
    info = {"model": spec.RERANK["model"], "revision": rev, "device": model_device(model), **cross_encoder_provenance(model),
            "load_and_score": check_reranker(model, "bge-rerank"), "shape": spec.RERANK}

    def score(query: str, texts: list[str]) -> list[float]:
        return cross_encoder_scores(model, [(query, t) for t in texts])

    return score, rev, info


def _common_header(root: Path, models: Sequence[str], reranker_info: dict, cache: RerankCache | None, stretch: bool, llm_dir) -> dict:
    files = {}
    manifest_path = root / LLM_DIR / ("manifest_122b.json" if stretch else "manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for tag in models:
        files[tag] = {"model": spec.MODELS[tag]["name"], "digest": spec.MODELS[tag]["digest"]}
    return {"llm_query": {"spec": {"bar": spec.BAR, "arms": [list(a) for a in spec.ARMS], "not_measured": {f"{t} {p}": why for (t, p), why in spec.NOT_MEASURED.items()},
                                   "request": spec.REQUEST, "num_predict": spec.NUM_PREDICT, "truncation": spec.TRUNCATION, "g": spec.G,
                                   "prompts_sha256": {k: spec.template_sha256(k) for k in spec.KINDS}, "rerank": spec.RERANK, "k": spec.K},
                          "declaration": declaration(root), "models": files, "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.exists() else None,
                          "cache_files": (manifest.get("files") or {}), "stretch": stretch},
            "reranker": reranker_info,
            "rerank_cache": ({"entries_reused": cache.hits, "computed": cache.computed, "gpu": cache.gpu,
                              "note": "reranker scores cached by (sha256 of the query text, unit, reranker revision, GPU name); an entry is reused only on the same GPU type"}
                             if cache else None)}


def _fixture_reproduction(root: Path, gold, backend, ranked, scores, n_files: int) -> dict[str, Any]:
    """P0 = neural.json hybrid@bge-small; P1 = rerank_depth.json's row; both also against small_rerankers.json's per-query
    sidecar; hybrid+hyde[qwen2.5-7b]@bge-small = frontier_neural.json hybrid+hyde-llm@bge-small."""
    from app.eval.metrics import aggregate_retrieval
    from app.eval.models_2026 import reproduces

    def retrieval(name):
        rows = [{"id": q.id, "category": q.category, "relevant": q.relevant_ids(), "retrieved": backend.paths(u)} for q, u in zip(gold, ranked["rows"][name])]
        return aggregate_retrieval(rows, n_files=n_files).as_dict()

    ids = [q.id for q in gold]
    checks = []
    qwen = spec.arm_name("hyde", "none", spec.QWEN25_TAG)
    for rel, mine, theirs in (("bench/results/neural.json", spec.P0, spec.P0), ("bench/results/rerank_depth.json", spec.P1, spec.P1),
                              ("bench/results/frontier_neural.json", qwen, "hybrid+hyde-llm@bge-small")):
        if mine not in ranked["rows"]:
            continue
        rep = reproduces(root, rel, [{"config": theirs, "retrieval": retrieval(mine)}], [theirs])
        row = (rep.get("rows") or {}).get(theirs) or {}
        checks.append({"row": mine, "file": rel, "compared_as": theirs, "means_equal": bool(row.get("equal")), **({"differs": row["differs"]} if row.get("differs") else {})})
        side = _sidecar_scores(root, rel)
        if side is not None and theirs in side.get("scores", {}):
            checks.append(check_scores(mine, rel, scores[mine], ids, side, theirs))
    side = _sidecar_scores(root, "bench/results/small_rerankers.json")
    for name in (spec.P0, spec.P1):
        checks.append(check_scores(name, "bench/results/small_rerankers.per_query.json", scores[name], ids, side))
    return reproduction_status(checks)


def _beir_reproduction(root: Path, dataset: str, qids, ranked, scores) -> dict[str, Any]:
    """P0, P1 and dense@bge-small against beir_<d>.json (rounded means; FiQA also its TREC runs' sha256 and per-query
    rankings) and P0, P1 against beir_<d>_small_rerankers.json (ranking_sha256 and its per-query sidecar)."""
    checks = []
    first = f"bench/results/beir_{dataset}.json"
    small = f"bench/results/beir_{dataset}_small_rerankers.json"
    committed = _committed_rows(root, first)
    small_rows = _committed_rows(root, small)
    side = _sidecar_scores(root, small)
    for rel, found in ((first, committed), (small, small_rows)):
        if not found:
            checks.append({"file": rel, "found": False, "means_equal": False})  # nothing to compare is not a reproduction
    for name in (spec.P0, spec.P1, spec.DENSE):
        lists = ranked["rows"][name]
        sha = trec_sha256(name, qids, lists)
        means = {m: round(sum(scores[name][m]) / len(qids), 4) for m in ("ndcg@10", "recall@10", "recall@100", "mrr@10")}
        if name in committed:
            want = {m: committed[name]["retrieval"].get(m) for m in means}
            entry = {"row": name, "file": first, "means_equal": want == means, **({"differs": {m: [want[m], means[m]] for m in means if want[m] != means[m]}} if want != means else {})}
            run_sha = (committed[name].get("trec") or {}).get("run_sha256")
            if run_sha:
                entry["ranking_equal"] = run_sha == sha
                if not entry["ranking_equal"]:
                    entry["queries_ordered_differently"] = _trec_differences(root, dataset, name, qids, lists)
            checks.append(entry)
        if name in small_rows:
            entry = {"row": name, "file": small, "ranking_equal": small_rows[name].get("ranking_sha256") == sha}
            entry.update({k: v for k, v in check_scores(name, small, scores[name], qids, side).items() if k not in ("row", "file")})
            if not entry["ranking_equal"]:
                entry["queries_ordered_differently"] = _trec_differences(root, dataset, name, qids, lists)
            checks.append(entry)
    return reproduction_status(checks)


def _trec_differences(root: Path, dataset: str, name: str, qids, lists) -> int | None:
    """Queries whose ranking differs from a committed TREC run of the row (FiQA keeps them); None where none is committed."""
    from app.eval.beir import _slug

    path = root / "bench" / "results" / f"beir_{dataset}_trec" / f"{_slug(name)}.run"
    if not path.exists():
        return None
    committed: dict[str, list[str]] = {}
    for line in path.read_text().splitlines():
        qid, _, did, *_ = line.split()
        committed.setdefault(qid, []).append(did)
    return sum(1 for q, ids in zip(qids, lists) if committed.get(q, []) != list(ids))


def main(argv: list[str] | None = None) -> int:
    from app.eval import per_question

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=spec.DATASETS, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=ROOT / "bench" / "external")
    p.add_argument("--models", default=",".join(spec.BAR_MODELS), help="model tags whose merged caches are scored (MODEL=qwen3.5-122b: the stretch)")
    p.add_argument("--rerank-cache", type=Path, default=None, help="the reranker-score cache file (resumable)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--reproduce-only", action="store_true", help="P0, P1 (and dense@bge-small on BEIR) against the committed rows, "
                   "per-query scores and ranking hashes; writes --out as that record only, never a results file")
    args = p.parse_args(argv)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    stretch = spec.STRETCH_MODEL in models
    if stretch and spec.MAIN_MODEL not in models:
        models = [spec.MAIN_MODEL, *models]  # the stretch rows are compared with qwen3.5:27b's in the same run
    gpu = _gpu_name()
    if args.dataset == "fixture":
        blob, sidecar = score_fixture(ROOT, models=models, rerank_cache=args.rerank_cache, gpu=gpu, stretch=stretch, reproduce_only=args.reproduce_only)
    else:
        blob, sidecar = score_beir(ROOT, args.data_dir, args.dataset, models=models, rerank_cache=args.rerank_cache, gpu=gpu,
                                   stretch=stretch, device=args.device, reproduce_only=args.reproduce_only)
    out = args.out if args.out.is_absolute() else ROOT / args.out
    if args.reproduce_only:
        if out.resolve().parent == (ROOT / "bench" / "results").resolve():
            raise SystemExit("--reproduce-only writes a record, never into bench/results")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(blob, indent=1) + "\n")
        print(json.dumps(blob["reproduction"], indent=1)[:4000])
        print("reproduction:", blob["reproduction"]["status"])
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2) + "\n")
    per_question.write(per_question.sidecar_path(out), sidecar)
    print("reproduction:", blob["reproduction"]["status"])
    scope = blob["scopes"][bar.BAR_SCOPE[args.dataset]]
    for c in scope["comparisons"]:
        if c["kind"] in ("arm", "reference", "stretch"):
            print(f"{c['kind']:9s} {c['row']} - {c['other']}: {bar.fmt_block(c)} {c['verdict']} (changed {c['n_changed']})")
    print(f"wrote {out}")
    return 0


def _gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


if __name__ == "__main__":
    raise SystemExit(main())
