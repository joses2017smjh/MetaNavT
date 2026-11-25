"""M46, part (e): the plumbing around the arms. Query ids, the qrels-free query loader, the worklist, the Ollama request
builder, the resumable generation, the merge into bench/llm_query, the probe report, the printed bar.

    python -m app.eval.llm_query_gen ids --data-dir <beir>                         (make llm-query-ids; a named qrels reader: id column only)
    python -m app.eval.llm_query_gen worklist --dataset D [--queries dev] ...      (make llm-query-worklist; GPU for P1; no qrels)
    python -m app.eval.llm_query_gen generate --dataset D --model M --shard i/n    (make llm-query-generate; Ollama only)
    python -m app.eval.llm_query_gen merge ...                                     (make llm-query-merge; CPU)
    python -m app.eval.llm_query_gen probe-report ...                              (the probe at D0: format and cost on dev queries)

Only the query text (and, for the grader, one unit's text) reaches a prompt; prompts are rebuilt from the template and
the text (llm_query_spec.build_prompt) and stored only as sha256. The worklist and generation never open qrels: they
read ids from bench/llm_query/query_ids.json and the text through load_queries (queries.jsonl, `text` only).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.eval import llm_query_spec as spec

ROOT = Path(__file__).resolve().parents[2]
QUERY_IDS = "bench/llm_query/query_ids.json"
LLM_DIR = "bench/llm_query"
M43_SIDECARS = {"scifact": "bench/results/beir_scifact_small_rerankers.per_query.json", "fiqa": "bench/results/beir_fiqa_small_rerankers.per_query.json"}
DEV_QRELS = {"scifact": "train", "fiqa": "dev"}
SYNTHETIC = "bench/distill/queries.jsonl.gz"
RETRIES = 2
DETERMINISM_N = 20


# ---------------------------------------------------------------- ids (make llm-query-ids; reads the qrels' id column only)


def qrels_query_ids(path: Path) -> list[str]:
    """The query ids of a qrels TSV in first-appearance order: the first column only (the doc-id and score columns are
    never parsed)."""
    out: dict[str, None] = {}
    with Path(path).open(encoding="utf-8") as fh:
        header = fh.readline()
        if not header.lower().startswith("query-id"):
            raise ValueError(f"{path}: not a BEIR qrels file")
        for line in fh:
            if line.strip():
                out.setdefault(line.split("\t", 1)[0].strip(), None)
    return list(out)


def seeded_sample(ids: Sequence[str], n: int = spec.DEV_QUERIES["n"], seed: int = spec.DEV_QUERIES["seed"]) -> list[str]:
    """The pool's ids sorted, permuted with RandomState(seed), the first n."""
    ordered = sorted(set(ids))
    return [ordered[i] for i in np.random.RandomState(seed).permutation(len(ordered))[:n]]


def fixture_dev_pool(root: Path) -> tuple[list[str], dict[str, Any]]:
    """M43's synthetic fixture queries minus every one whose nearest gold question has token-set Jaccard >= 0.5."""
    from app.eval.distill_queries import jaccard, normalise
    from app.eval.gold import load_gold

    gold_sets = [set(normalise(q.question).split()) for q in load_gold(root / "bench" / "gold" / "questions.jsonl")]
    with gzip.open(root / SYNTHETIC, "rt", encoding="utf-8") as fh:
        pool = [json.loads(line) for line in fh if line.strip()]
    pool = [q for q in pool if q["corpus"] == "fixture"]
    kept, dropped = [], 0
    for q in pool:
        tokens = set(normalise(q["query"]).split())
        if max(jaccard(tokens, g) for g in gold_sets) >= spec.DEV_QUERIES["fixture"]["jaccard_drop_at"]:
            dropped += 1
        else:
            kept.append(q["id"])
    return kept, {"pool": len(pool), "dropped_jaccard_at_least_0.5": dropped, "kept": len(kept)}


def make_ids(root: Path, data_dir: Path) -> dict[str, Any]:
    """bench/llm_query/query_ids.json: the fixture's gold ids and 20 dev ids; per BEIR set the test ids (in the order
    of the committed M43 sidecar, which is queries.jsonl's) and 20 seeded dev ids (SciFact train, FiQA dev)."""
    from app.eval.gold import load_gold

    out: dict[str, Any] = {"about": "query ids only: the test ids every M46 row is scored on and the dev ids of the probe (never scored "
                                    "for the record); made by `make llm-query-ids`, which reads the qrels' query-id column only",
                           "dev_rule": spec.DEV_QUERIES, "checks": {}}
    gold = load_gold(root / "bench" / "gold" / "questions.jsonl")
    pool, counts = fixture_dev_pool(root)
    out["fixture"] = {"test": [q.id for q in gold], "dev": seeded_sample(pool), "dev_pool": counts}
    for dataset in ("scifact", "fiqa"):
        droot = Path(data_dir) / dataset
        test = qrels_query_ids(droot / "qrels" / "test.tsv")
        side = json.loads((root / M43_SIDECARS[dataset]).read_text())["ids"]
        if set(test) != set(side) or len(test) != len(side):
            raise RuntimeError(f"{dataset}: the qrels test ids are not the M43 sidecar's")
        out["checks"][dataset] = {"m43_sidecar": M43_SIDECARS[dataset], "same_ids": True, "order": "the M43 sidecar's (queries.jsonl order)"}
        if dataset == "fiqa":
            trec = json.loads((root / "bench" / "results" / "beir_fiqa_trec" / "query_ids.json").read_text())
            if trec != side:
                raise RuntimeError("fiqa: beir_fiqa_trec/query_ids.json differs from the M43 sidecar's ids")
            out["checks"][dataset]["beir_fiqa_trec_query_ids"] = "equal, same order"
        dev_pool = qrels_query_ids(droot / "qrels" / f"{DEV_QRELS[dataset]}.tsv")
        overlap = set(dev_pool) & set(test)
        dev_pool = [q for q in dev_pool if q not in set(test)]
        out[dataset] = {"test": list(side), "dev": seeded_sample(dev_pool), "dev_pool": {"qrels": f"{DEV_QRELS[dataset]}.tsv", "ids": len(dev_pool),
                                                                                         "dropped_because_in_test": len(overlap)}}
    return out


# ---------------------------------------------------------------- queries (qrels-free)


def load_queries(dataset_root: Path, ids: Sequence[str]) -> dict[str, str]:
    """{id: text} for `ids`, in that order, from a BEIR queries.jsonl: the `text` field only (SciFact's `metadata`
    holds evidence doc ids and is never read). Opens nothing else."""
    want = set(ids)
    found: dict[str, str] = {}
    with (Path(dataset_root) / "queries.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if str(row["_id"]) in want:
                    found[str(row["_id"])] = row["text"]
    missing = [i for i in ids if i not in found]
    if missing:
        raise KeyError(f"{len(missing)} query ids not in queries.jsonl: {missing[:5]}")
    return {i: found[i] for i in ids}


def fixture_queries(root: Path, ids: Sequence[str]) -> dict[str, str]:
    """{id: text}: a gold question's text (bench/gold) or a synthetic dev query's (bench/distill); the text only."""
    from app.eval.gold import load_gold

    texts = {q.id: q.question for q in load_gold(root / "bench" / "gold" / "questions.jsonl")}
    if any(i not in texts for i in ids):
        with gzip.open(root / SYNTHETIC, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    texts.setdefault(row["id"], row["query"])
    return {i: texts[i] for i in ids}


def dataset_queries(root: Path, data_dir: Path, dataset: str, which: str) -> dict[str, str]:
    ids = json.loads((root / QUERY_IDS).read_text())[dataset][which]
    return fixture_queries(root, ids) if dataset == "fixture" else load_queries(Path(data_dir) / dataset, ids)


# ---------------------------------------------------------------- backends for the worklist and the probe (torch)


def build_backend(root: Path, data_dir: Path, dataset: str, *, rerank_cache: Path | None, gpu: str, device: str = "auto"):
    from app.eval import llm_query_arms as arms

    if dataset == "fixture":
        from app.eval.harness import BGE_SMALL
        from app.eval.index_loader import build_index

        index = build_index(root / "bench" / "corpus" / "files", embedder_name=BGE_SMALL, retrieve_k=spec.K["fixture"], enable_router=False,
                            enable_rerank=False, mtimes="manifest")
        score, rev, info = arms._load_bge_reranker(device)
        cache = arms.RerankCache(rerank_cache, rev, gpu) if rerank_cache else None
        return arms.FixtureBackend(index, lambda q, units: score(q, [index.by_id[u].text for u in units]), cache), info
    from app.eval import beir

    corpus = beir.load_corpus(Path(data_dir) / dataset)
    engine = beir.Engine(beir.chunks_from_corpus(corpus), cache_dir=Path(data_dir) / "cache", device=device)
    score, rev, info = arms._load_bge_reranker(device)
    cache = arms.RerankCache(rerank_cache, rev, gpu) if rerank_cache else None
    return arms.BeirBackend(engine, corpus, lambda q, units: score(q, [corpus[u] for u in units]), cache), info


# ---------------------------------------------------------------- the worklist (make llm-query-worklist)


def worklist_lines(queries: dict[str, str], parent_lists: dict[str, dict], passage) -> list[dict[str, Any]]:
    """Per query, in dataset order: the hyde, multiquery, decompose and rewrite prompts, then the grader prompts for
    the union of P0's and P1's top G units. One line per prompt, the same for every model."""
    out = []
    for qid, text in queries.items():
        for kind in spec.QUERY_KINDS:
            prompt = spec.build_prompt(kind, text)
            out.append({"prompt_sha256": spec.prompt_sha256(prompt), "kind": kind, "qid": qid, "unit": None, "unit_sha256": None,
                        "template_sha256": spec.template_sha256(kind)})
        p = parent_lists[qid]
        for unit in spec.grader_candidates(p["p0_top"], p["p1_top"]):
            body = passage(unit)
            prompt = spec.build_prompt("grade", text, body)
            out.append({"prompt_sha256": spec.prompt_sha256(prompt), "kind": "grade", "qid": qid, "unit": unit,
                        "unit_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(), "template_sha256": spec.template_sha256("grade")})
    return out


def run_parents(backend, queries: dict[str, str]) -> dict[str, dict[str, Any]]:
    from app.eval import llm_query_arms as arms

    out = {}
    for qid, text in queries.items():
        p0, p1 = arms.parents(backend, text)
        f0, f1 = backend.finish(qid, p0), backend.finish(qid, p1)
        out[qid] = {"id": qid, "text": text, "p0_sha256": arms.list_sha256(f0), "p1_sha256": arms.list_sha256(f1), "p0_top": p0[: spec.G],
                    "p1_top": p1[: spec.G], "_p0": f0, "_p1": f1}
    return out


def committed_hash_check(root: Path, dataset: str, parent_lists: dict[str, dict]) -> dict[str, Any]:
    """BEIR test: P0's and P1's whole-run ranking sha256 against beir_<d>_small_rerankers.json's ranking_sha256 and
    (FiQA) beir_fiqa.json's trec.run_sha256. No qrels."""
    from app.eval.llm_query_arms import trec_sha256

    qids = list(parent_lists)
    mine = {spec.P0: trec_sha256(spec.P0, qids, [parent_lists[q]["_p0"] for q in qids]),
            spec.P1: trec_sha256(spec.P1, qids, [parent_lists[q]["_p1"] for q in qids])}
    checks = []
    for rel, field in ((f"bench/results/beir_{dataset}_small_rerankers.json", "ranking_sha256"), (f"bench/results/beir_{dataset}.json", "trec")):
        path = root / rel
        if not path.exists():
            continue
        rows = {r["config"]: r for r in json.loads(path.read_text())["results"]}
        for name, sha in mine.items():
            want = rows.get(name, {}).get("ranking_sha256") if field == "ranking_sha256" else (rows.get(name, {}).get("trec") or {}).get("run_sha256")
            if want:
                checks.append({"row": name, "file": rel, "equal": want == sha})
    return {"ranking_sha256": mine, "checks": checks, "all_equal": all(c["equal"] for c in checks),
            **({} if checks else {"note": "no committed ranking hash found to compare with"})}


def write_worklist(root: Path, data_dir: Path, dataset: str, which: str, out_dir: Path, *, rerank_cache: Path | None, gpu: str,
                   device: str = "auto", backend=None) -> dict[str, Any]:
    """Run P0 and P1, check the BEIR test parents' hashes against the committed ones (before writing anything), then
    write the worklist, the parents and a meta file. Returns the meta block."""
    queries = dataset_queries(root, data_dir, dataset, which)
    info = None
    if backend is None:
        backend, info = build_backend(root, data_dir, dataset, rerank_cache=rerank_cache, gpu=gpu, device=device)
    parent_lists = run_parents(backend, queries)
    check = committed_hash_check(root, dataset, parent_lists) if (dataset != "fixture" and which == "test") else None
    name = f"{dataset}{'' if which == 'test' else '.' + which}"
    if check is not None and not check["all_equal"]:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{name}.hash-mismatch.json").write_text(json.dumps(check, indent=1) + "\n")
        raise SystemExit(f"{dataset}: P0/P1 ranking hashes differ from the committed ones; nothing written (run the scoring command's "
                         "reproduce-only mode before any generation)")
    lines = worklist_lines(queries, parent_lists, backend.passage)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(line, sort_keys=True) + "\n" for line in lines)
    (out_dir / f"{name}.worklist.jsonl").write_text(text)
    parents = {"dataset": dataset, "queries": [{k: v for k, v in p.items() if not k.startswith("_")} for p in parent_lists.values()]}
    (out_dir / f"{name}.parents.json").write_text(json.dumps(parents, indent=None, sort_keys=True) + "\n")
    meta = {"dataset": dataset, "queries": which, "n_queries": len(queries), "n_prompts": len(lines),
            "by_kind": {k: sum(1 for x in lines if x["kind"] == k) for k in spec.KINDS},
            "worklist_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "templates_sha256": {k: spec.template_sha256(k) for k in spec.KINDS},
            "committed_hash_check": check, "gpu": gpu, "reranker": info, "code_sha": _git_head(root),
            "timestamp": datetime.now(timezone.utc).isoformat()}
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    return meta


def read_worklist(out_dir: Path, dataset: str, which: str = "test") -> tuple[list[dict], dict, dict]:
    name = f"{dataset}{'' if which == 'test' else '.' + which}"
    lines = [json.loads(x) for x in (out_dir / f"{name}.worklist.jsonl").read_text().splitlines() if x.strip()]
    parents = json.loads((out_dir / f"{name}.parents.json").read_text())
    meta = json.loads((out_dir / f"{name}.meta.json").read_text())
    return lines, parents, meta


def passages_for(root: Path, data_dir: Path, dataset: str):
    """unit -> the grader passage, without torch: the fixture's chunks as the benchmark indexes them, or the BEIR corpus."""
    if dataset == "fixture":
        from app.eval.index_loader import load_chunks_with_report

        chunks, _ = load_chunks_with_report(root / "bench" / "corpus" / "files", strategy="auto", mtimes="manifest")
        by_id = {c.chunk_id: c for c in chunks}
        return lambda u: spec.fixture_passage(by_id[u].path, by_id[u].text)
    from app.eval.beir import load_corpus

    corpus = load_corpus(Path(data_dir) / dataset)
    return lambda u: spec.beir_passage(corpus[u])


def rebuild_prompts(lines: Sequence[dict], parents: dict, passage) -> list[tuple[dict, str]]:
    """Every worklist prompt rebuilt from its template and the text; a prompt whose sha256 differs stops."""
    texts = {q["id"]: q["text"] for q in parents["queries"]}
    out = []
    for line in lines:
        prompt = spec.build_prompt(line["kind"], texts[line["qid"]], passage(line["unit"]) if line["kind"] == "grade" else None)
        if spec.prompt_sha256(prompt) != line["prompt_sha256"]:
            raise RuntimeError(f"worklist prompt {line['kind']} {line['qid']} {line['unit']}: rebuilt prompt differs")
        out.append((line, prompt))
    return out


# ---------------------------------------------------------------- the request (one at a time; (payload, error))


def request_payload(model: str, kind: str, prompt: str) -> tuple[dict | None, str | None]:
    """The /api/generate body of one call, or (None, why) for a call that must not be sent."""
    if kind not in spec.NUM_PREDICT:
        return None, f"unknown kind {kind!r}"
    if not prompt:
        return None, "empty prompt"
    return {"model": model, "prompt": prompt, "stream": spec.REQUEST["stream"], "think": spec.REQUEST["think"],
            "keep_alive": spec.REQUEST["keep_alive"], "options": spec.options(kind)}, None


def post_generate(base_url: str, payload: dict, timeout: float = 1200.0) -> tuple[dict | None, str | None]:
    """(response, None) or (None, error): one POST, nothing swallowed."""
    req = urllib.request.Request(base_url.rstrip("/") + "/api/generate", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:300]
        except Exception:  # noqa: BLE001
            pass
        return None, f"HTTP {e.code}: {body}"
    except Exception as e:  # noqa: BLE001 - transport errors are retried and recorded
        return None, f"{type(e).__name__}: {e}"
    if not isinstance(out, dict):
        return None, "not a JSON object"
    if out.get("error"):
        return None, f"server error: {str(out['error'])[:300]}"
    return out, None


def call_once(base_url: str, model: str, kind: str, prompt: str, post=post_generate) -> dict[str, Any]:
    """One cached call's record: the reply and its counts, or the error after RETRIES transport retries."""
    payload, error = request_payload(model, kind, prompt)
    if payload is None:
        return {"error": error, "retries": 0}
    tries, resp = 0, None
    t0 = time.perf_counter()
    while True:
        t_call = time.perf_counter()
        resp, error = post(base_url, payload)
        wall = (time.perf_counter() - t_call) * 1000.0
        if resp is not None or tries >= RETRIES:
            break
        tries += 1
        time.sleep(2.0 * tries)
    if resp is None:
        return {"error": error, "retries": tries, "wall_ms": round((time.perf_counter() - t0) * 1000.0, 1)}
    rec = {"reply": resp.get("response") or "", "thinking_chars": len(resp.get("thinking") or ""), "done_reason": resp.get("done_reason"),
           "retries": tries, "wall_ms": round(wall, 1), "error": None}
    for k, v in resp.items():
        if (k.endswith("_count") or k.endswith("_duration")) and isinstance(v, (int, float)):
            rec[k] = v
    rec["overflowed"] = spec.overflowed(rec.get("prompt_eval_count"), kind)
    return rec


def server_check(base_url: str, model_tag: str) -> dict[str, Any]:
    """The server's version and the model's /api/tags digest; a digest that is not the pinned one stops the run."""
    from app.eval.provenance import ollama_tag_entry, ollama_tags, ollama_version

    name = spec.MODELS[model_tag]["name"]
    entry = ollama_tag_entry(name, ollama_tags(base_url))
    info = {"version": ollama_version(base_url), "digest": entry.get("digest"), "details": entry.get("details")}
    if info["digest"] != spec.MODELS[model_tag]["digest"]:
        raise SystemExit(f"{name}: /api/tags digest {info['digest']} is not the pinned {spec.MODELS[model_tag]['digest']}")
    return info


# ---------------------------------------------------------------- generation (make llm-query-generate; resumable)


def shard_queries(qids: Sequence[str], shard: str) -> list[str]:
    """Shard 'i/n' of the dataset's queries: contiguous blocks, so a query's calls run on one GPU."""
    i, n = (int(x) for x in shard.split("/"))
    if not 1 <= i <= n:
        raise ValueError(f"bad shard {shard}")
    bounds = np.linspace(0, len(qids), n + 1).round().astype(int)
    return list(qids[bounds[i - 1]: bounds[i]])


def read_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue  # a line cut by a killed job: the call is re-issued
    return out


def cache_state(entries: Sequence[dict]) -> tuple[dict[str, dict], dict[str, set[str]]]:
    """(the first completed output per key, the jobs in which each key failed)."""
    done: dict[str, dict] = {}
    failed: dict[str, set[str]] = {}
    for e in entries:
        if e.get("error") is None:
            done.setdefault(e["key"], e)
        else:
            failed.setdefault(e["key"], set()).add(str(e.get("job")))
    return done, failed


def generate(root: Path, data_dir: Path, dataset: str, model_tag: str, shard: str, *, worklist_dir: Path, cache_dir: Path, which: str = "test",
             base_url: str, deadline: float | None = None, job: str | None = None, gpu: str | None = None, post=post_generate,
             determinism_n: int = DETERMINISM_N) -> dict[str, Any]:
    """Issue every call of the shard that has no completed output yet, one at a time, appending one JSON line per call
    (flushed); stop cleanly 10 minutes before `deadline`. A key that failed in two jobs is final. At the end of a
    complete shard, re-issue `determinism_n` seeded completed calls into a separate log."""
    lines, parents, meta = read_worklist(worklist_dir, dataset, which)
    qids = [q["id"] for q in parents["queries"]]
    mine = set(shard_queries(qids, shard))
    passage = passages_for(root, data_dir, dataset)
    work = [(line, prompt) for line, prompt in rebuild_prompts([x for x in lines if x["qid"] in mine], parents, passage)]
    name = spec.MODELS[model_tag]["name"]
    digest = spec.MODELS[model_tag]["digest"]
    d = Path(cache_dir) / model_tag / (dataset if which == "test" else f"{dataset}.{which}")
    d.mkdir(parents=True, exist_ok=True)
    tag = shard.replace("/", "of")
    path = d / f"shard-{tag}.jsonl"
    done, failed = cache_state(read_jsonl(path))
    job = job or os.environ.get("SLURM_JOB_ID") or "local"
    gpu = gpu or _gpu_name_smi()
    code_sha = _git_head(root)
    issued = skipped_final = 0
    stopped_early = False
    with path.open("a", encoding="utf-8") as fh:
        for line, prompt in work:
            key = spec.cache_key(digest, line["kind"], prompt)
            if key in done:
                continue
            if len(failed.get(key, set())) >= 2:  # failed in two jobs: final
                skipped_final += 1
                continue
            if job in failed.get(key, set()):
                continue  # failed in this job already: the next resume re-issues it once
            if deadline is not None and time.time() > deadline - 600:
                stopped_early = True
                break
            rec = call_once(base_url, name, line["kind"], prompt, post=post)
            entry = {"key": key, "prompt_sha256": line["prompt_sha256"], "kind": line["kind"], "qid": line["qid"], "unit": line["unit"],
                     "model": model_tag, **rec, "gpu": gpu, "job": job, "code_sha": code_sha, "ts": datetime.now(timezone.utc).isoformat()}
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fh.flush()
            issued += 1
            if rec.get("error") is None:
                done[key] = entry
            else:
                failed.setdefault(key, set()).add(job)
    keys = [spec.cache_key(digest, line["kind"], prompt) for line, prompt in work]
    complete = all(k in done or len(failed.get(k, set())) >= 2 for k in keys)
    summary = {"dataset": dataset, "queries": which, "model": model_tag, "shard": shard, "job": job, "gpu": gpu, "code_sha": code_sha,
               "calls_in_shard": len(work), "issued_this_job": issued, "completed": sum(1 for k in keys if k in done),
               "final_failures": sum(1 for k in keys if k not in done and len(failed.get(k, set())) >= 2), "complete": complete,
               "stopped_before_deadline": stopped_early}
    if complete and determinism_n and not (d / f"determinism-{tag}.jsonl").exists():
        summary["determinism"] = determinism(base_url, name, [done[k] for k in keys if k in done], {k: prompt for k, (_, prompt) in zip(keys, work)},
                                             d / f"determinism-{tag}.jsonl", n=determinism_n, post=post)
    return summary


def determinism(base_url: str, model_name: str, completed: Sequence[dict], prompts: dict[str, str], log: Path, *, n: int, post=post_generate) -> dict[str, Any]:
    """Re-issue n seeded completed calls on the same node; count byte-identical replies. Never replaces a cached output."""
    if not completed:
        return {"n": 0}
    pick = sorted(np.random.RandomState(0).choice(len(completed), size=min(n, len(completed)), replace=False).tolist())
    same = 0
    with log.open("w", encoding="utf-8") as fh:
        for i in pick:
            e = completed[i]
            rec = call_once(base_url, model_name, e["kind"], prompts[e["key"]], post=post)
            identical = rec.get("error") is None and rec.get("reply") == e.get("reply")
            same += int(identical)
            fh.write(json.dumps({"key": e["key"], "kind": e["kind"], "identical": identical, "error": rec.get("error")}) + "\n")
    return {"n": len(pick), "byte_identical": same, "log": log.name}


# ---------------------------------------------------------------- merge (make llm-query-merge; CPU)

KEEP_FIELDS = ("key", "prompt_sha256", "kind", "qid", "unit", "reply", "thinking_chars", "done_reason", "eval_count", "prompt_eval_count",
               "prompt_eval_cached_count", "total_duration", "load_duration", "prompt_eval_duration", "eval_duration", "wall_ms", "retries",
               "overflowed", "error", "gpu")


def write_gz(path: Path, rows: Sequence[dict]) -> str:
    """Deterministic gzip JSONL (mtime 0): re-merging the same calls gives the same bytes. Returns the sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode("utf-8")
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as fh:
            fh.write(data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_one(root: Path, data_dir: Path, dataset: str, model_tag: str, *, worklist_dir: Path, cache_dir: Path, server_logs: dict | None = None) -> dict[str, Any]:
    """One (dataset, model): every worklist key covered exactly once by its first completed output (or its final
    failure), every prompt rebuilt from template.format(query=...) and matched to its key, then written in worklist
    order. Returns the manifest block; raises when the cache is not complete."""
    lines, parents, meta = read_worklist(worklist_dir, dataset)
    passage = passages_for(root, data_dir, dataset)
    digest = spec.MODELS[model_tag]["digest"]
    work = rebuild_prompts(lines, parents, passage)
    keys = [spec.cache_key(digest, line["kind"], prompt) for line, prompt in work]
    if len(set(keys)) != len(keys):
        raise RuntimeError(f"{dataset}: duplicate prompts in the worklist")
    d = Path(cache_dir) / model_tag / dataset
    shard_files = sorted(d.glob("shard-*.jsonl"))
    entries = [e for f in shard_files for e in read_jsonl(f)]
    done, failed = cache_state(entries)
    extra = sorted({e["key"] for e in entries} - set(keys))
    if extra:
        raise RuntimeError(f"{dataset} {model_tag}: {len(extra)} cached keys are not in the worklist (another prompt set?)")
    missing = [k for k in keys if k not in done and len(failed.get(k, set())) < 2]
    if missing:
        raise IncompleteCache(f"{dataset} {model_tag}: {len(missing)} of {len(keys)} calls have no completed output yet")
    rows = []
    for k, (line, _) in zip(keys, work):
        if k in done:
            e = done[k]
        else:
            last = [x for x in entries if x["key"] == k][-1]
            e = {**last, "error": last.get("error") or "failed", "final_failure": True}
        if (e["kind"], e["qid"], e.get("unit")) != (line["kind"], line["qid"], line["unit"]):
            raise RuntimeError(f"{k}: the cached call is not the worklist's")
        rows.append({f: e.get(f) for f in KEEP_FIELDS if f in e or f in ("error",)} | ({"final_failure": True} if e.get("final_failure") else {}))
    out = root / LLM_DIR / f"{dataset}.{model_tag}.jsonl.gz"
    sha = write_gz(out, rows)
    from app.eval.llm_query_arms import read_call

    texts = {q["id"]: q["text"] for q in parents["queries"]}
    counts: dict[str, dict[str, int]] = {}
    for r in rows:
        c = counts.setdefault(r["kind"], {s: 0 for s in ("calls", "ok", "failed", "empty", "unparseable", "unchanged", "overflowed", "truncated", "thinking")})
        _, status, trunc = read_call(r, r["kind"], texts[r["qid"]])
        c["calls"] += 1
        c[status] = c.get(status, 0) + 1
        c["truncated"] += int(trunc)
        c["thinking"] += int((r.get("thinking_chars") or 0) > 0)
    shards = []
    for f in shard_files:
        es = read_jsonl(f)
        tag = f.stem.replace("shard-", "")
        metas = [json.loads(m.read_text()) for m in sorted(d.glob(f"shard-{tag}.meta-*.json"))]  # one per job: the load check, the server log
        det = d / f"determinism-{tag}.jsonl"
        det_rows = read_jsonl(det)
        shards.append({"shard": tag.replace("of", "/"), "calls_logged": len(es), "jobs": sorted({str(e.get("job")) for e in es}),
                       "gpu": sorted({e.get("gpu") for e in es if e.get("gpu")}), "code_sha": sorted({e.get("code_sha") for e in es if e.get("code_sha")}),
                       "earliest": min((e["ts"] for e in es if e.get("ts")), default=None), "latest": max((e["ts"] for e in es if e.get("ts")), default=None),
                       "server": metas,
                       "determinism": {"n": len(det_rows), "byte_identical": sum(1 for x in det_rows if x.get("identical"))} if det_rows else None})
    return {"file": f"{LLM_DIR}/{dataset}.{model_tag}.jsonl.gz", "sha256": sha, "n_calls": len(rows), "counts": counts,
            "worklist_sha256": meta["worklist_sha256"], "shards": shards, "model": spec.MODELS[model_tag]}


class IncompleteCache(RuntimeError):
    pass


def merge(root: Path, data_dir: Path, *, worklist_dir: Path, cache_dir: Path, models: Sequence[str], datasets: Sequence[str] = spec.DATASETS,
          manifest_name: str = "manifest.json") -> dict[str, Any]:
    """Merge every complete (dataset, model) into bench/llm_query, write parents.json.gz (every dataset whose worklist
    exists) and the manifest. An incomplete pair is reported and left out: a partly generated cache is never scored."""
    manifest_path = root / LLM_DIR / manifest_name
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    files = dict(old.get("files") or {})
    incomplete = {}
    for dataset in datasets:
        if not (worklist_dir / f"{dataset}.worklist.jsonl").exists():
            continue
        for tag in models:
            if not (Path(cache_dir) / tag / dataset).exists():
                incomplete[f"{dataset}.{tag}"] = "not generated"
                continue
            try:
                files[f"{dataset}.{tag}"] = merge_one(root, data_dir, dataset, tag, worklist_dir=worklist_dir, cache_dir=cache_dir)
            except IncompleteCache as e:
                incomplete[f"{dataset}.{tag}"] = str(e)
    parents = {}
    worklists = {}
    for dataset in spec.DATASETS:
        if (worklist_dir / f"{dataset}.worklist.jsonl").exists():
            _, p, meta = read_worklist(worklist_dir, dataset)
            parents[dataset] = p
            worklists[dataset] = {k: meta[k] for k in ("n_queries", "n_prompts", "by_kind", "worklist_sha256", "templates_sha256", "committed_hash_check", "gpu", "code_sha", "timestamp")}
    parents_sha = None
    if parents and manifest_name == "manifest.json":
        path = root / LLM_DIR / "parents.json.gz"
        data = json.dumps(parents, sort_keys=True).encode("utf-8")
        with open(path, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as fh:
                fh.write(data)
        parents_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    from app.eval.provenance import scrub

    earliest = min((s["earliest"] for f in files.values() for s in f["shards"] if s.get("earliest")), default=None)
    manifest = scrub({"about": "the merged LLM caches of M46 (make llm-query-merge): one entry per worklist prompt, the first completed "
                               "output per cache key; scoring reads only these files",
                      "ollama": spec.SERVER, "request": spec.REQUEST, "num_predict": spec.NUM_PREDICT, "options_base": spec.OPTIONS_BASE,
                      "prompts": spec.PROMPTS, "prompts_sha256": {k: spec.template_sha256(k) for k in spec.KINDS}, "models": {t: spec.MODELS[t] for t in models},
                      "worklists": worklists, "parents_file": {"file": f"{LLM_DIR}/parents.json.gz", "sha256": parents_sha} if parents_sha else old.get("parents_file"),
                      "files": files, "incomplete": incomplete, "earliest_cache_timestamp": earliest,
                      "kept_partial_caches": old.get("kept_partial_caches", [])})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


# ---------------------------------------------------------------- the probe at D0 (dev queries only; no qrels, no metric)


def digits_lost(reply: str | None) -> int:
    """Lines whose numbering-and-bullet strip (^[\\d.)\\-*\\s]+) would remove digits that are not a list number: the
    stripped prefix holds a digit and neither '.' nor ')' ("401k rollover" -> "k rollover"; "2023 tax" -> "tax")."""
    import re

    n = 0
    for line in (reply or "").split("\n"):
        m = re.match(r"^[\d.)\-*\s]+", line.strip())
        if m and any(c.isdigit() for c in m.group(0)) and not any(c in ".)" for c in m.group(0)):
            n += 1
    return n


def probe_report(root: Path, data_dir: Path, *, worklist_dir: Path, cache_dir: Path, models: Sequence[str], backend_factory=None,
                 out_path: Path | None = None) -> dict[str, Any]:
    """Per dataset and model, on the dev worklists: think honoured (empty `thinking`), parse statuses, truncation and
    overflow per kind, latency per kind, tokens; and, given a backend factory (torch), the fired counts (lists changed
    vs the parent) of every arm on the dev queries."""
    from app.eval import llm_query_arms as arms

    out: dict[str, Any] = {"datasets": {}}
    for dataset in spec.DATASETS:
        try:
            lines, parents, meta = read_worklist(worklist_dir, dataset, "dev")
        except FileNotFoundError:
            continue
        texts = {q["id"]: q["text"] for q in parents["queries"]}
        block: dict[str, Any] = {"n_queries": len(texts), "n_prompts": len(lines), "models": {}}
        for tag in models:
            entries = [e for f in sorted((Path(cache_dir) / tag / f"{dataset}.dev").glob("shard-*.jsonl")) for e in read_jsonl(f)]
            done, failed = cache_state(entries)
            by_kind: dict[str, Any] = {}
            for e in done.values():
                k = by_kind.setdefault(e["kind"], {"calls": 0, "statuses": {}, "truncated": 0, "thinking": 0, "wall_ms": [], "eval_count": [], "prompt_eval_count": []})
                _, status, trunc = arms.read_call(e, e["kind"], texts[e["qid"]])
                k["calls"] += 1
                k["statuses"][status] = k["statuses"].get(status, 0) + 1
                k["truncated"] += int(trunc)
                k["thinking"] += int((e.get("thinking_chars") or 0) > 0)
                k["wall_ms"].append(e.get("wall_ms") or 0.0)
                k["eval_count"].append(e.get("eval_count") or 0)
                k["prompt_eval_count"].append(e.get("prompt_eval_count") or 0)
                if e["kind"] in ("multiquery", "decompose"):
                    k["lines_losing_digits"] = k.get("lines_losing_digits", 0) + digits_lost(e.get("reply"))
            for k in by_kind.values():
                for f in ("wall_ms", "eval_count", "prompt_eval_count"):
                    v = k.pop(f)
                    k[f] = {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "max": float(max(v))} if v else None
                k["truncated_share"] = round(k["truncated"] / k["calls"], 4) if k["calls"] else None
                bad = sum(n for s, n in k["statuses"].items() if s in ("unparseable",))
                k["unparseable_share"] = round(bad / k["calls"], 4) if k["calls"] else None
            per_query_ms = {}
            for e in done.values():
                per_query_ms[e["qid"]] = per_query_ms.get(e["qid"], 0.0) + float(e.get("wall_ms") or 0.0)
            block["models"][tag] = {"completed": len(done), "failed_keys": len([k for k in failed if k not in done]), "by_kind": by_kind,
                                    "llm_s_per_query": {"p50": round(float(np.percentile(list(per_query_ms.values()), 50)) / 1000, 2),
                                                        "max": round(max(per_query_ms.values()) / 1000, 2)} if per_query_ms else None,
                                    "gpu": sorted({e.get("gpu") for e in done.values() if e.get("gpu")})}
        out["datasets"][dataset] = block
        if out_path is not None:  # the call statistics are kept even if the fired step below fails
            out_path.write_text(json.dumps(out, indent=1) + "\n")
        if backend_factory is not None:
            try:
                backend = backend_factory(dataset)
                sources = {tag: arms.CacheSource(tag, [e for f in sorted((Path(cache_dir) / tag / f"{dataset}.dev").glob("shard-*.jsonl")) for e in read_jsonl(f)],
                                                 backend.passage) for tag in models}
                queries = [(q["id"], q["text"]) for q in parents["queries"]]
                ranked = arms.rank_dataset(backend, queries, sources, beir=dataset != "fixture",
                                           expected={q["id"]: q for q in parents["queries"]})
                fired = {}
                for tag in models:
                    for t, p, par, _ in spec.ARMS:
                        name = spec.arm_name(t, p, tag)
                        fired[name] = arms.fired_summary(name, spec.P0 if par == "P0" else spec.P1, ranked)
                block["fired"] = fired
            except Exception as e:  # noqa: BLE001 - recorded per dataset; the statistics above stand
                block["fired_error"] = f"{type(e).__name__}: {e}"
            if out_path is not None:
                out_path.write_text(json.dumps(out, indent=1) + "\n")
    return out


# ---------------------------------------------------------------- the printed bar (make llm-query-arms-bar)


def print_bar(root: Path) -> int:
    """The declared bar row by row, the replication labels, the decision sentences with each named arm's cost, the
    fixture's ceiling (P1's held-out nDCG@10 and Recall@50 from small_rerankers.json), an indicative minimum detectable
    delta per dataset from the committed M43 sidecars, and the milestone's count of uncorrected intervals."""
    from app.eval import llm_query_bar as bar
    from app.eval.rerank_arms import split_questions

    blobs = bar.load_blobs(root)
    small = json.loads((root / "bench/results/small_rerankers.json").read_text())
    p1 = next(r for r in small["held_out"]["results"] if r["config"] == spec.P1)
    print(f"fixture ceiling (small_rerankers.json, held-out, {spec.P1}): nDCG@10 {p1['ci']['ndcg@10']['mean']:.4f}, Recall@50 {p1['ci']['recall@50']['mean']:.4f}")
    for dataset, rel in (("fixture", "bench/results/small_rerankers.per_query.json"), ("scifact", M43_SIDECARS["scifact"]), ("fiqa", M43_SIDECARS["fiqa"])):
        side = json.loads((root / rel).read_text())
        a, b = side["scores"][spec.P1]["ndcg@10"], side["scores"][spec.P0]["ndcg@10"]
        idx = list(range(len(a)))
        if dataset == "fixture":
            from app.eval.gold import load_gold

            idx = split_questions(load_gold(root / "bench" / "gold" / "questions.jsonl"))[1]
        diffs = [a[i] - b[i] for i in idx]
        print(f"indicative minimum detectable nDCG@10 delta, {dataset}: {bar.minimum_detectable(diffs):.4f} (2.8 x SD of P1 - P0 / sqrt({len(diffs)}), {rel})")
    for tag in spec.BAR_MODELS:
        print(f"\n{spec.MODELS[tag]['name']}")
        for row in bar.bar_rows(blobs, tag):
            cells = "; ".join(f"{d} {bar.fmt_block(row['cells'][d]['block'])} {row['words'][d] or 'missing'}" for d in spec.DATASETS)
            print(f"  {row['passes']:14s} all-three {row['all_three']:8s} replication {row.get('replication', '-'):15s} {row['config']}: {cells}")
    print()
    for dec in bar.decisions(blobs):
        print(f"rule {dec['rule']}: {dec['sentence']}")
    total = bar.n_intervals(blobs)
    print(f"\nuncorrected paired nDCG@10 intervals in the milestone's files: {total.get('total', 0)} ({total})")
    return 0


# ---------------------------------------------------------------- D0, D1 and the check (make llm-query-arms-check)

PARTS = {"a": "app/eval/llm_query_spec.py", "b": "app/eval/llm_query_parse.py", "c": "app/eval/llm_query_bar.py", "d": "app/eval/llm_query_arms.py"}
DECLARATION = "bench/llm_query/declaration.json"


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-c", "pack.threads=1", *args], cwd=root, text=True, stderr=subprocess.DEVNULL)


def _show(root: Path, sha: str, rel: str) -> str | None:
    try:
        return _git(root, "show", f"{sha}:{rel}")
    except subprocess.CalledProcessError:
        return None


def _is_ancestor(root: Path, older: str, newer: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", older, newer], cwd=root, capture_output=True).returncode == 0


def _without_num_predict(source: str) -> str:
    """Part (a)'s syntax tree with the value of the NUM_PREDICT assignment blanked (the one allowed D0..D1 change)."""
    import ast

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "NUM_PREDICT" for t in node.targets):
            node.value = ast.Constant(value=None)
    return ast.dump(tree, include_attributes=False)


def changed_functions(old: str | None, new: str | None) -> list[str]:
    """Top-level functions and classes whose source differs between two versions of a module."""
    import ast

    def defs(src):
        if src is None:
            return {}
        tree = ast.parse(src)
        return {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

    a, b = defs(old), defs(new)
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def declare(root: Path, d0: str, d1: str) -> dict[str, Any]:
    """bench/llm_query/declaration.json: D0, D1, their commit times, and what changed between them (files; the changed
    functions of parts (b) and (d)). Results headers copy it."""
    d0, d1 = _git(root, "rev-parse", d0).strip(), _git(root, "rev-parse", d1).strip()
    files = [x for x in _git(root, "diff", "--name-only", d0, d1).splitlines() if x]
    return {"D0": d0, "D1": d1, "D0_time": _git(root, "show", "-s", "--format=%cI", d0).strip(), "D1_time": _git(root, "show", "-s", "--format=%cI", d1).strip(),
            "d0_to_d1": {"files": files, "changed_functions": {part: changed_functions(_show(root, d0, rel), _show(root, d1, rel))
                                                                for part, rel in PARTS.items() if part in ("b", "d")},
                         "num_predict": {"D0": _num_predict(_show(root, d0, PARTS["a"])), "D1": _num_predict(_show(root, d1, PARTS["a"]))}},
            "rule": "between D0 and D1: (a) only NUM_PREDICT (syntax tree), (b) parser fixes the probe showed, (c) byte-identical, (d) only a "
                    "wiring bug the probe exposed with a failing test first, (e) plumbing; after D1 parts (a)-(d) are frozen"}


def _num_predict(source: str | None) -> Any:
    import ast

    if source is None:
        return None
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "NUM_PREDICT" for t in node.targets):
            return ast.literal_eval(node.value)
    return None


def arms_check(root: Path) -> list[str]:
    """The three assertions of llm-query-arms-check: every test generation shard's and scoring run's code SHA descends
    from D1 and holds parts (a)-(d) byte-identical to D1; D1 descends from D0; between D0 and D1 part (a) is equal but for
    NUM_PREDICT (syntax tree) and part (c) is byte-identical. Returns the problems (empty: OK)."""
    path = root / DECLARATION
    if not path.exists():
        return [f"{DECLARATION} is missing"]
    decl = json.loads(path.read_text())
    d0, d1 = decl["D0"], decl["D1"]
    problems = []
    if not _is_ancestor(root, d0, d1):
        problems.append(f"D1 {d1[:12]} does not descend from D0 {d0[:12]}")
    a0, a1 = _show(root, d0, PARTS["a"]), _show(root, d1, PARTS["a"])
    if a0 is None or a1 is None or _without_num_predict(a0) != _without_num_predict(a1):
        problems.append("part (a) differs between D0 and D1 beyond the NUM_PREDICT assignment")
    if _show(root, d0, PARTS["c"]) != _show(root, d1, PARTS["c"]):
        problems.append("part (c) is not byte-identical between D0 and D1")
    shas: dict[str, str] = {}
    for name in ("manifest.json", "manifest_122b.json"):
        m = root / LLM_DIR / name
        if m.exists():
            for key, f in (json.loads(m.read_text()).get("files") or {}).items():
                for shard in f.get("shards") or []:
                    for sha in shard.get("code_sha") or []:
                        shas[sha] = f"generation {key} shard {shard['shard']}"
    from app.eval.llm_query_bar import RESULTS, STRETCH_RESULTS

    for rel in (*RESULTS.values(), *STRETCH_RESULTS.values()):
        f = root / rel
        if f.exists():
            head = json.loads(f.read_text()).get("git_head")
            if head:
                shas[head] = f"scoring run {rel}"
            else:
                problems.append(f"{rel}: no git_head")
    for sha, what in sorted(shas.items()):
        if not _is_ancestor(root, d1, sha):
            problems.append(f"{what}: code {sha[:12]} does not descend from D1 {d1[:12]}")
            continue
        for part, rel in PARTS.items():
            if _show(root, d1, rel) != _show(root, sha, rel):
                problems.append(f"{what}: part ({part}) {rel} differs from D1 at {sha[:12]}")
    return problems


# ---------------------------------------------------------------- helpers


def _git_head(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def _gpu_name_smi() -> str:
    name = os.environ.get("M46_GPU_NAME")
    if name:
        return name
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL, timeout=30)
        return out.strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ids")
    s.add_argument("--data-dir", type=Path, required=True)
    s = sub.add_parser("declare")
    s.add_argument("--d0", required=True)
    s.add_argument("--d1", required=True)
    sub.add_parser("check")
    for name in ("worklist", "generate", "merge", "probe-report"):
        s = sub.add_parser(name)
        s.add_argument("--data-dir", type=Path, required=True)
        s.add_argument("--worklist-dir", type=Path, required=True)
        if name != "worklist":
            s.add_argument("--cache-dir", type=Path, required=True)
        if name in ("worklist", "generate"):
            s.add_argument("--dataset", choices=spec.DATASETS, required=True)
            s.add_argument("--queries", choices=("test", "dev"), default="test")
        if name in ("worklist", "probe-report"):
            s.add_argument("--rerank-cache-dir", type=Path, default=None)
            s.add_argument("--device", default="auto")
        if name == "generate":
            s.add_argument("--model", choices=tuple(spec.MODELS), required=True)
            s.add_argument("--shard", default="1/1")
            s.add_argument("--deadline", type=float, default=None, help="epoch seconds; the run stops cleanly 10 minutes before")
        if name in ("merge", "probe-report"):
            s.add_argument("--models", default=",".join(spec.BAR_MODELS))
        if name == "probe-report":
            s.add_argument("--out", type=Path, required=True)
            s.add_argument("--fired", action="store_true", help="also rank the dev queries (torch) for the fired counts")
    args = p.parse_args(argv)
    if args.cmd == "ids":
        out = make_ids(ROOT, args.data_dir)
        path = ROOT / QUERY_IDS
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1) + "\n")
        print({d: {k: len(v) for k, v in out[d].items() if isinstance(v, list)} for d in spec.DATASETS}, out["fixture"]["dev_pool"])
        return 0
    if args.cmd == "declare":
        out = declare(ROOT, args.d0, args.d1)
        (ROOT / DECLARATION).write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out, indent=1))
        return 0
    if args.cmd == "check":
        problems = arms_check(ROOT)
        for line in problems:
            print(f"  - {line}")
        print(f"llm-query-arms-check: {'OK' if not problems else f'{len(problems)} problems'}")
        return 1 if problems else 0
    if args.cmd == "worklist":
        gpu = _gpu_name_smi()
        # its own score cache, so the scoring run recomputes P1 rather than reading the worklist's scores
        cache = (args.rerank_cache_dir / f"worklist-{args.dataset}{'' if args.queries == 'test' else '.' + args.queries}.jsonl") if args.rerank_cache_dir else None
        meta = write_worklist(ROOT, args.data_dir, args.dataset, args.queries, args.worklist_dir, rerank_cache=cache, gpu=gpu, device=args.device)
        print(json.dumps({k: meta[k] for k in ("dataset", "queries", "n_queries", "n_prompts", "by_kind", "worklist_sha256", "committed_hash_check")}, default=str))
        return 0
    if args.cmd == "generate":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        info = server_check(base_url, args.model)
        print(json.dumps({"server": info}), flush=True)
        summary = generate(ROOT, args.data_dir, args.dataset, args.model, args.shard, worklist_dir=args.worklist_dir, cache_dir=args.cache_dir,
                           which=args.queries, base_url=base_url, deadline=args.deadline)
        print(json.dumps(summary), flush=True)
        return 0 if summary["complete"] or summary["stopped_before_deadline"] else 1
    if args.cmd == "merge":
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        name = "manifest_122b.json" if spec.STRETCH_MODEL in models else "manifest.json"
        manifest = merge(ROOT, args.data_dir, worklist_dir=args.worklist_dir, cache_dir=args.cache_dir, models=models, manifest_name=name)
        print(json.dumps({"files": {k: {"n_calls": v["n_calls"], "sha256": v["sha256"][:16]} for k, v in manifest["files"].items()},
                          "incomplete": manifest["incomplete"]}, indent=1))
        return 0
    if args.cmd == "probe-report":
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        factory = None
        if args.fired:
            def factory(dataset):
                cache = (args.rerank_cache_dir / f"{dataset}.dev.jsonl") if args.rerank_cache_dir else None
                return build_backend(ROOT, args.data_dir, dataset, rerank_cache=cache, gpu=_gpu_name_smi(), device=args.device)[0]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        report = probe_report(ROOT, args.data_dir, worklist_dir=args.worklist_dir, cache_dir=args.cache_dir, models=models, backend_factory=factory,
                              out_path=args.out)
        args.out.write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps(report, indent=1)[:6000])
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
