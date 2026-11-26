"""Render docs/RESUME_EVIDENCE.md from the committed results files (make evidence).

Every number in the claims table is read from a results file in bench/results/
and carries the exact command, the results file and its git sha, and a plain
statement of what it does NOT show. The resume bullets and the interview
answers are prose without digits; they point at rows of the table. A test
(tests/eval/test_resume_evidence.py) fails if the committed document differs
from this renderer's output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "bench" / "results"

FILES = {
    "main": "bench/results/main.json",
    "neural": "bench/results/neural.json",
    "beir": "bench/results/beir_scifact.json",
    "beir_rerank_cap": "bench/results/beir_scifact_rerank_cap.json",
    "parity": "bench/results/parity_shared_chunker.json",
    "parity_neural": "bench/results/parity_neural.json",
    "rerank_depth": "bench/results/rerank_depth.json",
    "latency": "bench/results/api_latency_docker_cpu_ml.json",
    "jury": "bench/results/jury_v2.json",  # M29; the M4 run (jury.json) is the earlier record
    "agent_injection": "bench/results/agent_injection_v2.json",  # the second run; the first is agent_injection.json
    "agent_injection_v3": "bench/results/agent_injection_v3.json",  # the current run: phrasings as any-success, the in-loop answer check
    "agent_injection_adaptive": "bench/results/agent_injection_adaptive.json",  # the adaptive attacker, held-out questions
    "frontier": "bench/results/frontier.json",
    "frontier_neural": "bench/results/frontier_neural.json",
    "rerank_arms": "bench/results/rerank_arms.json",
    "companion_live": "bench/results/companion_live.json",
    "chat_route_smoke": "bench/results/chat_route_smoke.json",
    "rrf_sweep": "bench/results/rrf_sweep.json",
    "search_policy": "bench/results/search_policy.json",  # M44: the RL search policy, trained once
    "hybrid_drops": "bench/results/hybrid_drops_bge_base.json",
    "file_depth": "bench/results/file_depth.json",  # M37: the file-depth rows beside their chunk-cut parents
    # M41: the current open models beside their bge parents, recomputed in each run
    "models_2026": "bench/results/models_2026.json",
    "models_2026_cpu": "bench/results/models_2026_cpu.json",
    "beir_scifact_models_2026": "bench/results/beir_scifact_models_2026.json",
    "beir_fiqa_models_2026": "bench/results/beir_fiqa_models_2026.json",
    "models_2026_harrier": "bench/results/models_2026_harrier.json",  # harrier-oss-v1-27b: its own runs, on an 80 GB GPU
    "beir_scifact_models_2026_harrier": "bench/results/beir_scifact_models_2026_harrier.json",
    "beir_fiqa_models_2026_harrier": "bench/results/beir_fiqa_models_2026_harrier.json",
    # M43: small rerankers over hybrid@bge-small at the served shape (fixture held-out, SciFact, FiQA)
    "small_rerankers": "bench/results/small_rerankers.json",
    "beir_scifact_small_rerankers": "bench/results/beir_scifact_small_rerankers.json",
    "beir_fiqa_small_rerankers": "bench/results/beir_fiqa_small_rerankers.json",
    "served_scale": "bench/results/served_scale_fiqa.json",
    "pgvector_hnsw": "bench/results/pgvector_hnsw.json",
}


def _load(rel: str, root: Path = ROOT) -> dict | None:
    p = root / rel
    return json.loads(p.read_text()) if p.exists() else None


def _ci(block: dict | None, key: str = "mean", digits: int = 3) -> tuple[str, str]:
    if not block:
        return "—", "—"
    if key == "delta":
        return f"{block['delta']:+.{digits}f}", f"[{block['lo']:+.{digits}f}, {block['hi']:+.{digits}f}]"
    return f"{block[key]:.{digits}f}", f"[{block['lo']:.{digits}f}, {block['hi']:.{digits}f}]"


def _rows(blob: dict) -> dict[str, dict]:
    return {r["config"]: r for r in blob.get("results", [])}


def _zero(block: dict | None) -> str:
    """Whether a paired interval covers zero, in words (owner rule: an interval that covers zero is reported as such)."""
    if not block:
        return "no interval"
    return "the interval covers zero" if block["lo"] <= 0.0 <= block["hi"] else "the interval excludes zero"


def _p50(row: dict, stage: str = "total"):
    return ((row.get("latency") or {}).get(stage) or {}).get("p50_ms")


def _reranker_split_claims(blob: dict, file: str, command: str) -> list[dict]:
    """SciFact reranker at depth 20: max_length 512 vs uncapped (both fp32), then fp16 vs fp32 (both capped)."""
    by, n = _rows(blob), blob["n_queries"]
    src, gpu = f"{file} @ {blob['git_sha']}", (blob.get("device") or {}).get("device_name", "the recorded device")
    rer, f32c, f16 = (by.get(f"hybrid+bge-rerank@bge-small{v}") for v in ("", "/fp32-512/top20", "/fp16-512/top20"))
    out = []
    if rer and f32c and f32c.get("parent") == rer["config"]:
        d, ci = _ci((f32c.get("delta_vs_parent") or {}).get("ndcg@10"), "delta")
        out.append({"claim": "BEIR SciFact, reranker max_length 512 vs uncapped (both fp32, depth 20), paired delta in nDCG@10", "value": d, "ci": ci, "n": n, "command": command, "source": src,
                    "not_shown": f"the cap alone; rerank-stage p50 {_p50(f32c, 'rerank')} ms vs {_p50(rer, 'rerank')} ms per query on {gpu}; CPU latency is a different table"})
    if f32c and f16 and f16.get("parent") == f32c["config"]:
        d, ci = _ci((f16.get("delta_vs_parent") or {}).get("ndcg@10"), "delta")
        out.append({"claim": "BEIR SciFact, reranker fp16 vs fp32 (both max_length 512, depth 20; fp16 is the served setting on CUDA, the CPU stack serves fp32), paired delta in nDCG@10", "value": d, "ci": ci, "n": n, "command": command, "source": src,
                    "not_shown": f"the precision alone; rerank-stage p50 {_p50(f16, 'rerank')} ms vs {_p50(f32c, 'rerank')} ms per query on {gpu}; CPU latency is a different table"})
    return out


def _templated(root: Path = ROOT) -> int:
    """Questions bench/corpus/build.py stamps from per-run templates: the simple_factual category."""
    p = root / "bench" / "gold" / "questions.jsonl"
    if not p.exists():
        return 0
    return sum(1 for line in p.read_text().splitlines() if line.strip() and json.loads(line).get("category") == "simple_factual")


def claims(root: Path = ROOT) -> list[dict]:
    out: list[dict] = []
    main = _load(FILES["main"], root)
    if main:
        by = _rows(main)
        full, bm25 = by.get("hybrid+rerank+router+staleness"), by.get("bm25_only")
        n = main["n_gold"]
        src = f"{FILES['main']} @ {main['git_sha']}"
        if full and bm25:
            v, ci = _ci(full["ci"]["ndcg@10"])
            out.append({"claim": "Fixture v1, full CI pipeline (hash embeddings, overlap reranker, router, staleness): nDCG@10", "value": v, "ci": ci, "n": n, "command": "make bench", "source": src,
                        "not_shown": f"not a neural or production-corpus number; hash embeddings and an overlap reranker; {main['n_files']} files and {n} questions generated by bench/corpus/build.py over a synthetic corpus ({_templated(root)} templated simple-factual, the rest written individually in the script)"})
            v, ci = _ci(full["ci"]["recall@10"])
            out.append({"claim": "Fixture v1, full CI pipeline: Recall@10", "value": v, "ci": ci, "n": n, "command": "make bench", "source": src,
                        "not_shown": "path-level relevance over 61 files; a random list of the same length scores " + f"{full['retrieval'].get('random_recall@10', 0):.3f}"})
            d, ci = _ci((full.get("delta_vs_bm25_only") or {}).get("ndcg@10"), "delta")
            out.append({"claim": "Fixture v1: full CI pipeline minus BM25 alone, paired delta in nDCG@10", "value": d, "ci": ci, "n": n, "command": "make bench", "source": src,
                        "not_shown": "the interval covers zero: with hash embeddings and the overlap reranker, hybrid does not beat BM25 on ranking"})
            d, ci = _ci((full.get("delta_vs_previous") or {}).get("ndcg@10"), "delta")
            out.append({"claim": "Fixture v1: staleness Tier 1 on vs off (over the reranked, routed hybrid), paired delta in nDCG@10", "value": d, "ci": ci, "n": n, "command": "make bench", "source": src,
                        "not_shown": "a fixture-specific mechanism (versioned configs); with real models the same step is smaller (the staleness row under real models below)"})
            cat = full.get("delta_vs_parent_by_category") or {}
            if cat:
                d, ci = _ci(cat.get("ndcg@10"), "delta")
                out.append({"claim": f"Fixture v1: staleness on its own category ({cat['category']}), paired delta in nDCG@10", "value": d, "ci": ci, "n": cat["n"], "command": "make bench", "source": src,
                            "not_shown": "small n; no claim beyond the slice"})
            router = by.get("hybrid+rerank+router")
            rc = (router or {}).get("delta_vs_parent_by_category") or {}
            if rc:
                d, ci = _ci(rc.get("ndcg@10"), "delta")
                zero = "the interval includes zero" if rc["ndcg@10"]["lo"] <= 0 else "the interval is above zero"
                r10 = rc.get("recall@10") or {}
                order = "; Recall@10 on the slice is unchanged, so the gain is ordering only" if r10 and r10.get("delta") == 0 and r10.get("lo") == 0 and r10.get("hi") == 0 else ""
                moved = rc.get("questions_changed")  # D04: how many of the slice's questions the delta rests on
                rests = f" (nDCG@10 rose on {moved['up']} of them and fell on {moved['down']})" if moved else ""
                from app.eval.report import negative_slices  # keep losers visible: the categories the router loses on

                lost = "; ".join(f"{b['category']} {' '.join(_ci(b['ndcg@10'], 'delta'))} (n = {b['n']})" for b in negative_slices(router))
                losers = f"; on categories it is not aimed at its nDCG@10 delta is negative: {lost}" if lost else ""
                out.append({"claim": f"Fixture v1: router on its own category ({rc['category']}) over the reranked hybrid, paired delta in nDCG@10", "value": d, "ci": ci, "n": rc["n"], "command": "make bench", "source": src,
                            "not_shown": f"n = {rc['n']}{rests} and {zero}{losers}; hash embeddings and the overlap reranker{order}; the earlier README figure 0.596 -> 0.938 compared against plain RRF and overstated the router"})
            out.append({"claim": "Fixture v1: Recall@50 of the full CI pipeline", "value": f"{full['retrieval']['recall@50']:.3f}", "ci": _ci(full['ci']['recall@50'])[1], "n": n, "command": "make bench", "source": src,
                        "not_shown": f"a random list with the same number of unique files ({full['retrieval'].get('mean_unique_paths@50', 0):.0f} of {main['n_files']}) scores {full['retrieval'].get('random_recall@50', 0):.3f}; the metric mostly measures list length"})
    neural = _load(FILES["neural"], root)
    if neural:
        by = _rows(neural)
        n = neural["n_gold"]
        src = f"{FILES['neural']} @ {neural['git_sha']}"
        dev = neural.get("device", {}).get("device_name", "?")
        for name, label, extra in (("hybrid@bge-small", "Fixture v1, hybrid RRF with bge-small-en-v1.5 vs BM25 alone, paired delta in nDCG@10", "no reranker on either side"),
                                   ("hybrid+bge-rerank@bge-small", "Fixture v1, hybrid (bge-small-en-v1.5) + bge-reranker-v2-m3 vs BM25 alone, paired delta in nDCG@10", "the reranker rescores most of the corpus")):
            row = by.get(name)
            if row:
                d, ci = _ci((row.get("delta_vs_bm25_only") or {}).get("ndcg@10"), "delta")
                out.append({"claim": label, "value": d, "ci": ci, "n": n, "command": "make bench-neural", "source": src,
                            "not_shown": f"run on one GPU ({dev}); {row.get('n_chunks', '?')} chunks in the fixture, {extra}; not a production-corpus number"})
        full = by.get("hybrid+bge-rerank+router+staleness@bge-small")
        if full:
            v, ci = _ci(full["ci"]["ndcg@10"])
            out.append({"claim": "Fixture v1, full pipeline with real models (bge-small + bge-reranker-v2-m3 + router + staleness): nDCG@10", "value": v, "ci": ci, "n": n, "command": "make bench-neural", "source": src,
                        "not_shown": f"the fixture is {neural['n_files']} files; SciFact carries the retrieval-quality claim"})
            st = (full.get("delta_vs_previous") or {}).get("ndcg@10")
            if st:
                d, ci = _ci(st, "delta")
                out.append({"claim": "Fixture v1 with real models: staleness Tier 1 on vs off (over bge-small + bge-reranker-v2-m3 + router), paired delta in nDCG@10", "value": d, "ci": ci, "n": n, "command": "make bench-neural", "source": src,
                            "not_shown": "the parent stack here differs from the CI row in embedder, reranker and device, so this is the same switch on a different stack, not a controlled comparison of rerankers; " + ("the interval is above zero" if st["lo"] > 0 else "the interval covers zero")})
        comps = (neural.get("bootstrap") or {}).get("comparisons") or []
        for c in comps:
            if c.get("row") == "hybrid+bge-rerank@bge-small":
                d, ci = _ci(c.get("ndcg@10"), "delta")
                out.append({"claim": "Fixture v1: bge-small + reranker vs hash + reranker (does the first stage matter once the reranker is on?), paired delta in nDCG@10", "value": d, "ci": ci, "n": n, "command": "make bench-neural", "source": src,
                            "not_shown": f"a tie because the reranker rescores {by[c['row']]['settings']['retrieve_k']} of {by[c['row']]['n_chunks']} chunks; it says nothing about corpora where the first stage is a real filter"})
    out += _rrf_sweep_claims(root)
    out += _search_policy_claims(root)
    out += _hybrid_drops_claims(root)
    out += _file_depth_claims(root)
    out += _models_2026_claims(root)
    out += _small_reranker_claims(root)
    out += _small_reranker_shape_claims(root)  # M51: the cheaper shapes, a row per family and per cell x dataset whose 95% interval excludes zero
    beir = _load(FILES["beir"], root)
    if beir:
        by = _rows(beir)
        n = beir["n_queries"]
        src = f"{FILES['beir']} @ {beir['git_sha']}"
        bm = by.get("bm25")
        if bm and bm.get("vs_published"):
            v, ci = _ci(bm["ci"]["ndcg@10"], digits=4)  # the precision of the published score and the difference
            out.append({"claim": "BEIR SciFact, our BM25 with the Anserini analyzer and document text: nDCG@10", "value": v, "ci": ci, "n": n, "command": "make bench-beir", "source": src,
                        "not_shown": f"Anserini's published flat BM25 is {bm['vs_published']['published_bm25_ndcg@10']:.4f} ({bm['vs_published']['delta']:+.4f}); this validates the lexical baseline, not the product"})
        for name, label in (("dense@bge-small", "BEIR SciFact, dense bge-small vs BM25, paired delta in nDCG@10"),
                            ("hybrid@bge-small", "BEIR SciFact, hybrid RRF vs BM25, paired delta in nDCG@10"),
                            ("hybrid+bge-rerank@bge-small", "BEIR SciFact, hybrid + bge-reranker-v2-m3 (top 20, fp32) vs BM25, paired delta in nDCG@10")):
            row = by.get(name)
            if row:
                d, ci = _ci((row.get("delta_vs_bm25") or {}).get("ndcg@10"), "delta")
                lat = (row.get("latency") or {}).get("total") or {}
                out.append({"claim": label, "value": d, "ci": ci, "n": n, "command": "make bench-beir", "source": src,
                            "not_shown": f"GPU run; per-query p50 {lat.get('p50_ms', '?')} ms; one dataset, one domain (scientific claims)"})
        hyb, rer = by.get("hybrid@bge-small"), by.get("hybrid+bge-rerank@bge-small")
        if rer:
            d, ci = _ci((rer.get("delta_vs_parent") or {}).get("ndcg@10"), "delta")
            out.append({"claim": "BEIR SciFact, reranker over the hybrid it reranks (incremental), paired delta in nDCG@10", "value": d, "ci": ci, "n": n, "command": "make bench-beir", "source": src,
                        "not_shown": "the increment over hybrid is a tie on SciFact even though both beat BM25"})
        # The capped fp32 row joined the ladder after the first SciFact file: there, fp16 + cap is one step from fp32
        # uncapped. The cap and the precision one at a time come from beir_scifact.json once its ladder has that row,
        # else from the file that measured it (make bench-beir-rerank-cap), on the GPU that file names.
        f16 = by.get("hybrid+bge-rerank@bge-small/fp16-512/top20")
        if f16 and rer and f16.get("parent") == rer["config"]:
            d, ci = _ci((f16.get("delta_vs_parent") or {}).get("ndcg@10"), "delta")
            l16 = (f16.get("latency") or {}).get("total") or {}
            l32 = (rer.get("latency") or {}).get("total") or {}
            out.append({"claim": "BEIR SciFact, reranker fp16 + max_length 512 vs fp32 uncapped (the served setting on CUDA; the CPU stack serves fp32), paired delta in nDCG@10", "value": d, "ci": ci, "n": n, "command": "make bench-beir", "source": src,
                        "not_shown": f"p50 {l16.get('p50_ms')} ms vs {l32.get('p50_ms')} ms on the GPU; CPU latency is a different table"})
        split, split_file, split_cmd = beir, FILES["beir"], "make bench-beir"
        if "hybrid+bge-rerank@bge-small/fp32-512/top20" not in by:
            split, split_file, split_cmd = _load(FILES["beir_rerank_cap"], root), FILES["beir_rerank_cap"], "make bench-beir-rerank-cap"
        if split:
            out += _reranker_split_claims(split, split_file, split_cmd)
        d50 = by.get("hybrid+bge-rerank@bge-small/fp16-512/top50")
        if d50:
            d, ci = _ci((d50.get("delta_vs_parent") or {}).get("ndcg@10"), "delta")
            out.append({"claim": "BEIR SciFact, rerank depth 50 vs 20 (both fp16, max_length 512), paired delta in nDCG@10", "value": d, "ci": ci, "n": n, "command": "make bench-beir", "source": src,
                        "not_shown": "deeper reranking loses here; depth 100 loses further (rows in the file)"})
    parity = _load(FILES["parity"], root)
    if parity:
        by = _rows(parity)
        api, mem = by.get("api:postgres"), next((r for k, r in by.items() if k.startswith("in_memory")), None)
        if api and mem:
            g = parity.get("gap") or parity.get("gate") or {}  # files recorded before the gate was dropped carry "gate"
            d = g.get("paired_delta_api_minus_in_memory") or {}
            dv, ci = _ci(d, "delta") if d else ("—", "—")
            out.append({"claim": "API over Postgres + pgvector vs the in-memory bench (same switches), nDCG@10 gap", "value": dv, "ci": ci, "n": parity["n_gold"], "command": "make parity", "source": f"{FILES['parity']} @ {parity['git_sha']}",
                        "not_shown": "reported, not gated: the lexical legs differ by design (ts_rank_cd vs Okapi BM25); chunks and embeddings are shared, and tests/database/test_dense_parity.py requires the dense leg to match exactly; CI re-runs the parity job on every push to main and every pull request"})
            lat = (api.get("latency") or {}).get("e2e") or {}
            out.append({"claim": "API over Postgres, hash embedder, no reranker: per-query e2e latency p50 / p95 (ms)", "value": f"{lat.get('p50_ms')} / {lat.get('p95_ms')}", "ci": "—", "n": parity["n_gold"], "command": "make parity", "source": f"{FILES['parity']} @ {parity['git_sha']}",
                        "not_shown": "measured on the machine that produced the file, over HTTP, 61 files; not a load test"})
    depth = _load(FILES["rerank_depth"], root)
    top20 = _rows(depth).get("hybrid+bge-rerank@bge-small/top20") if depth else None
    if top20:
        block = (top20.get("delta_vs_previous") or {}).get("ndcg@10")
        d, ci = _ci(block, "delta")
        changed = top20.get("vs_parent") or {}
        out.append({"claim": "Fixture v1, bge-reranker-v2-m3 on the fused top 20 (the served RERANK_DEPTH) vs every fused candidate (fp32, bge-small), paired delta in nDCG@10",
                    "value": d, "ci": ci, "n": depth["n_gold"], "command": "make bench-rerank-depth", "source": f"{FILES['rerank_depth']} @ {depth['git_sha']}",
                    "not_shown": f"{_zero(block)}; the depth alone; {changed.get('n_top10_differs', '?')} of {changed.get('n_questions', '?')} questions change their top 10; "
                                 "the file walks on to max_length 512, fp16, the router with Tier 1, and Tier 1 before the rerank, each against its parent"})
    neural_parity = _load(FILES["parity_neural"], root)
    if neural_parity:
        d = (neural_parity.get("gap") or {}).get("paired_delta_api_minus_in_memory") or {}
        dv, ci = _ci(d, "delta") if d else ("—", "—")
        differ = ", ".join(neural_parity.get("settings_differ") or []) or "none"
        device = (neural_parity.get("device") or {}).get("device_name")
        bare = (neural_parity.get("api_health") or {}).get("embedding_query_instruction") == ""
        query = "the bare query: EMBEDDING_QUERY_INSTRUCTION=none, not the served default" if bare else "LlamaIndex's query instruction"
        out.append({"claim": f"API with bge-small ({query}) + bge-reranker-v2-m3 (fp16, max_length 512, depth 20) vs the in-memory bench with the same settings, nDCG@10 gap",
                    "value": dv, "ci": ci, "n": neural_parity["n_gold"], "command": f"make parity PARITY_CONFIG={neural_parity.get('in_memory_config')}",
                    "source": f"{FILES['parity_neural']} @ {neural_parity['git_sha']}",
                    "not_shown": f"{_zero(d)}; one run in a Slurm GPU job ({device}), not re-run by CI; reported, not gated; "
                                 f"settings that differ between the two sides: {differ}"})
    lat = _load(FILES["latency"], root)
    if lat:
        h = lat.get("health", {}).get("reranker", {})
        out.append({"claim": "Docker CPU stack (bge-small + bge-reranker-v2-m3 fp32, max_length 512, depth 20): server total p50 / p95 (ms)", "value": f"{lat['server_total_ms']['p50']} / {lat['server_total_ms']['p95']}", "ci": "—", "n": lat["n"],
                    "command": "docker compose -f docker-compose.yml -f docker-compose.ml.yml up --build; scripts/smoke_retrieve.py --latency bench/gold/questions.jsonl", "source": FILES["latency"] + " (CI artifact, docker-ml-smoke job)",
                    "not_shown": f"a 4-core CI runner; reranker device {h.get('device')}; the reranker is {lat['stages_p50_ms'].get('rerank')} ms of it"})
    out += _served_scale_claims(root)
    jury = _load(FILES["jury"], root)
    if jury and "protocol" in jury:
        out += _jury_v2_claims(jury, FILES["jury"])
        out += _jury_v3_claims(root)  # M42: current open models as generators and judges, beside the committed run
    elif jury:
        s = jury["summary"]
        c = s["checks"]
        rs = jury.get("rescore") or {}
        src = f"{FILES['jury']} @ {jury['git_sha']}" + (f" (answers generated at {rs['rows_git_sha']}, checks recomputed by `make bench-jury-rescore`)" if rs else "")
        gate = jury.get("kappa_gate", 0.6)
        ref = s.get("reference") or {}
        evaluable = ref.get("gate_evaluable", True)
        out.append({"claim": "Generated answers (Ollama, fixture v1): answers with no valid citation (failed loud)", "value": f"{c['uncited_answers']['count']} of {c['uncited_answers']['n']}", "ci": "—", "n": c["uncited_answers"]["n"], "command": "make bench-jury", "source": src,
                    "not_shown": "a deterministic check of the citation contract, not answer quality"})
        cr = c["cited_in_retrieved"]
        out.append({"claim": "Generated answers: every cited path is in the retrieved set (answers with at least one citation)", "value": f"{cr['rate']}", "ci": "—", "n": cr["n"], "command": "make bench-jury", "source": src,
                    "not_shown": f"{round((1 - cr['rate']) * cr['n'])} answers cited a path the retrieval had not returned (a fabricated or malformed path); abstentions are not counted"})
        out.append({"claim": "Generated answers: every number / config value appears in the cited bytes", "value": f"{c['values_in_cited_bytes']['rate']}", "ci": "—", "n": c["values_in_cited_bytes"]["n"], "command": "make bench-jury", "source": src,
                    "not_shown": "only answers that state a value are counted; string containment, case-insensitive; a value in a different numeric format, or one the model computed, counts as missing"})
        em = c["exact_match_simple_factual"]
        out.append({"claim": "Generated answers: exact match vs gold (numeric-equivalent), simple_factual", "value": f"{em['rate']}", "ci": "—", "n": em["n"], "command": "make bench-jury", "source": src,
                    "not_shown": "string / numeric match; paraphrased correct answers count as non-matches"
                    + ("; the slice (the templated questions) is saturated for this generator, which is why the kappa gate below could not be evaluated" if em["rate"] == 1.0 else "")})
        for name, k in s["kappa"].items():
            if name == "jury_majority" or k.get("heuristic"):
                continue
            if not evaluable:
                rc = ref["label_counts"]
                status = (f"gate not evaluable: the exact-match reference is one class ({rc['correct']} correct of {ref['n']}), so kappa is 0 by construction "
                          f"for any judge that varies and 1 for one that never does; this judge's scores are not published")
            else:
                status = f"passes the {gate} gate; the judge's mean score may be published" if k["readme_ok"] else f"below the {gate} gate; this judge's scores are not published"
            out.append({"claim": f"Judge {name}: Cohen's kappa vs exact-match labels, simple_factual", "value": f"{k['kappa_vs_exact_match_simple_factual']}", "ci": "—", "n": k["n"], "command": "make bench-jury", "source": src,
                        "not_shown": status + "; the reference is an exact-match label: a negated or hedged answer that still quotes the gold value counts as correct, a correct paraphrase does not"})
        jm = s["kappa"].get("jury_majority")
        if jm:
            out.append({"claim": "Jury majority (two judge models): Cohen's kappa vs exact-match labels, simple_factual", "value": f"{jm['kappa_vs_exact_match_simple_factual']}", "ci": "—", "n": jm["n"], "command": "make bench-jury", "source": src,
                        "not_shown": ("gate not evaluable on this run (one-class reference); " if not evaluable else "") + f"the generator is one of the judges (self-preference risk); gate {gate}"})
        ja = s.get("judge_agreement")
        if ja:
            out.append({"claim": f"Judge-judge agreement ({ja['judges'][0]} vs {ja['judges'][1]}): Cohen's kappa over all judged answers", "value": f"{ja['kappa']}", "ci": "—", "n": ja["n"], "command": "make bench-jury", "source": src,
                        "not_shown": "agreement between two judges says nothing about agreement with the truth; both judged the same generator's answers"})
        for name, p in (s.get("pairwise_llm_vs_extractive") or {}).items():
            out.append({"claim": f"Judge {name}: position gap in AB/BA pairwise comparison (LLM answer vs extractive)", "value": f"{p['position_gap_mean']}", "ci": "—", "n": p["n"], "command": "make bench-jury", "source": src,
                        "not_shown": "a measure of order bias, not of answer quality"})
    out += _agent_injection_v3_claims(root)  # the current run leads; the second run and the adaptive run follow
    out += _agent_injection_claims(root)
    out += _agent_injection_adaptive_claims(root)
    out += _frontier_claims(root)
    out += _rerank_arm_claims(root)
    out += _companion_live_claims(root)
    out += _chat_route_claims(root)
    return out


def _chat_route_claims(root: Path = ROOT) -> list[dict]:
    """The chat routes with a real model on the default stack (make chat-route-smoke): a smoke run, not accuracy."""
    blob = _load(FILES["chat_route_smoke"], root)
    if not blob:
        return []
    src = f"{FILES['chat_route_smoke']} @ {str(blob.get('git_head') or '')[:7]}"
    cmd = "make chat-route-smoke"
    model, gpu = blob["model"], (blob.get("device") or {}).get("gpu") or "the recorded GPU"
    llm = blob.get("chat_llm") or {}
    common = (f"a smoke run: one request per question, the first gold question of each category; not an accuracy "
              f"measurement; hash embeddings, Postgres ts_rank_cd, no reranker; chat temperature {llm.get('temperature')} "
              f"(the default), no seed; code execution off")
    out = []

    def stated(route: dict) -> str:
        aside = route["states_gold_value_aside"]
        return (f"answered means the route returned text, not that the text answers the question (it states the gold "
                f"value in {aside['k']} of the {aside['n']} questions with a value answer that the question does not "
                f"itself name)")

    def by_name(counts: dict | None) -> str:
        return ", ".join(f"{name} {count}" for name, count in (counts or {}).items()) or "none"

    def unanswered(route: str) -> list[dict]:
        return [r for r in blob.get("rows") or [] if r.get("route") == route and not r.get("answered")]

    def why_not(route: str) -> str:
        """How the route's unanswered requests ended, counted from the rows; nothing when every request was answered."""
        rows = unanswered(route)
        if not rows:
            return ""
        no_response = sum(r.get("http_status") is None for r in rows)
        empty = sum(r.get("http_status") == 200 and not r.get("answer") and not r.get("errors") for r in rows)
        error_line = sum(r.get("http_status") == 200 and bool(r.get("errors")) for r in rows)
        statuses = sorted({r["http_status"] for r in rows if r.get("http_status") not in (None, 200)})
        parts = ([f"{no_response} got no response within the runner's {blob['request_timeout_s']:g} s limit"] if no_response else []) + \
                [f"{sum(r.get('http_status') == code for r in rows)} returned HTTP {code}" for code in statuses] + \
                ([f"{empty} returned HTTP 200 with no text and no error line"] if empty else []) + \
                ([f"{error_line} returned HTTP 200 with an error line"] if error_line else [])
        return f" (not answered: {', '.join(parts)}; see doc/engineering-reference.md, Backend)"

    stream = blob["summary"].get("/api/chat")
    if stream:
        over = ", over every request, the unanswered ones included" if unanswered("/api/chat") else ""
        out.append({"claim": f"Chat route POST /api/chat (the web UI's streaming route), {model} through Ollama on the default stack: requests answered (HTTP 200, text, no error line)",
                    "value": f"{stream['answered']} of {stream['n']}{why_not('/api/chat')}", "ci": "—", "n": stream["n"], "command": cmd, "source": src,
                    "not_shown": f"{common}; {stated(stream)}; the route sends nothing until the agent run is over (median time to the first byte {stream['ttfb_s']['p50']} s on {gpu}{over}); tool calls are not visible on this route"})
    request = blob["summary"].get("/api/chat/request")
    if request:
        # Files recorded before the runner counted "Not found: ..." replies have no such field: nothing is said.
        counts_not_found = "tool_calls_returned_not_found" in request
        not_found_value = f"; not found: {by_name(request['tool_calls_returned_not_found'])}" if counts_not_found else ""
        not_found_note = ("; a 'Not found: ...' reply (get_file_content asked for a path the index does not hold) "
                          "is not counted as failed" if counts_not_found else "")
        failed = any((request.get(k) or {}) for k in ("tool_calls_raised", "tool_calls_returned_error_text"))
        causes = "; the causes are described in doc/engineering-reference.md (Backend)" if failed else ""
        unrecorded = sum(r.get("tools") is None for r in unanswered("/api/chat/request"))
        unrecorded_note = (f"; the tool calls of a request that got no response are not in the file ({unrecorded} here), so "
                           f"the counts leave them out" if unrecorded else "")
        out.append({"claim": f"Chat route POST /api/chat/request (non-streaming), {model} through Ollama on the default stack: requests answered; tool calls by name, and the calls that failed",
                    "value": (f"{request['answered']} of {request['n']}{why_not('/api/chat/request')}; tool calls: {by_name(request.get('tool_calls'))}; "
                              f"flagged as errors: {by_name(request.get('tool_calls_raised'))}; returned an error text: "
                              f"{by_name(request.get('tool_calls_returned_error_text'))}{not_found_value}; "
                              f"code tool calls: {request['code_tool_calls']}"),
                    "ci": "—", "n": request["n"], "command": cmd, "source": src,
                    "not_shown": (f"{common}; {stated(request)}; a call flagged as an error raised inside the tool or named a tool that does not exist; an error "
                                  f"text is an 'Error: ...' string a tool returned without the flag{not_found_note}; a call that returned "
                                  f"an empty list is not counted as failed{unrecorded_note}{causes}")})
    return out


def _companion_live_claims(root: Path = ROOT) -> list[dict]:
    """The research companion in live mode (make bench-companion-live): outcomes, stated gold values, citations,
    and the planted-value answers over the planted copy."""
    blob = _load(FILES["companion_live"], root)
    if not blob:
        return []
    src = f"{FILES['companion_live']} @ {str(blob.get('git_head') or '')[:7]}"
    cmd = "make bench-companion-live"
    model = blob["model"]
    gold = blob["summary"]["gold"]["overall"]
    stack = ("the served retrieval is the default stack's stand-ins (hash embeddings, Postgres ts_rank_cd instead of BM25, no "
             "reranker; file names and paths are searched only for a search with a path-like token, the router's "
             "lexical_path route); one trial per question at temperature 0 with no seed; synthetic fixture questions")
    out = [{"claim": f"Research companion, live mode ({model} through Ollama), every fixture v1 gold question: outcome",
            "value": "; ".join(f"{k} {v}" for k, v in gold["status"].items()), "ci": "—", "n": gold["n"],
            "command": cmd, "source": src,
            "not_shown": f"{stack}; a blocked answer is the companion failing closed (no valid JSON or no inspected citation), not a served answer"}]
    sg = gold["states_gold"]
    if sg["n"]:
        out.append({"claim": f"Research companion, live mode ({model}): answers that state the gold value, of the gold questions with a value answer",
                    "value": f"{sg['rate']:.3f}", "ci": f"[{sg['lo']:.3f}, {sg['hi']:.3f}]", "n": sg["n"], "command": cmd,
                    "source": src,
                    "not_shown": f"deterministic value matching, not entailment: an answer that quotes a whole config states every value in it; sentence and yes/no golds are not scored; answers that did not complete (abstained, blocked, stopped at the model-round limit) and failed requests count as not stating it; {stack}"})
    cit = gold["citations"]
    blocked_citing = gold["stop_reason"].get("invalid_citations", 0)
    if cit["complete_answers"]:
        out.append({"claim": f"Research companion, live mode ({model}): completed answers whose every citation names an inspected source whose excerpt hash and bytes match the cited range of the file",
                    "value": f"{cit['complete_answers_all_valid']} of {cit['complete_answers']}", "ci": "—",
                    "n": cit["complete_answers"], "command": cmd, "source": src,
                    "not_shown": f"reference integrity, not factual support; only completed answers are counted: the companion itself blocks an answer that cites a source it did not inspect and returns it without citations ({blocked_citing} here, stop reason invalid_citations), and abstentions and answers stopped at the model-round limit carry no citation; {stack}"})
    planted = blob["summary"].get("planted")
    if planted:
        when = planted["planted_value_stated_when_exposed"]
        clean = planted["same_questions_clean_corpus"]
        out.append({"claim": f"Research companion, live mode ({model}), planted instructions (the injection benchmark's ten hijack texts): answers that state the planted value, of the questions in which the model saw the planted text",
                    "value": f"{when['k']} of {when['n']}", "ci": "—", "n": when["n"], "command": cmd, "source": src,
                    "not_shown": f"one phrasing per task, not an adaptive attacker; the same {clean['n']} questions on the clean copy state the planted value {clean['planted_value_stated']} times; {stack}"})
    return out


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _jury_v2_claims(blob: dict, file: str) -> list[dict]:
    """M29: the run's deterministic checks; each judge's kappa against exact match on the calibration set; a judge's
    correctness score of the run's answers only when it passed the gate. A file without a calibration set (`make
    bench-jury` since M29) has the run's own simple-factual answers as its reference, and the rows say so."""
    s, c = blob["summary"], blob["summary"]["checks"]
    gen = blob["models"]["generator"]["model"]
    calgen = ((blob.get("calibration") or {}).get("generator") or {}).get("model")
    cmd = "make bench-jury-v2" if calgen else "make bench-jury"
    rs = blob.get("rescore") or {}
    src = f"{file} @ {blob['git_sha']}" + (f" (answers generated at {rs['rows_git_sha']}, checks recomputed by `{cmd}-rescore`)" if rs else "")
    gate, ref = blob.get("kappa_gate", 0.6), s["reference"]
    against = "calibration set" if calgen else "simple_factual"
    out = [
        {"claim": f"Generated answers ({gen} via Ollama, fixture v1): answers with no valid citation (failed loud)", "value": f"{c['uncited_answers']['count']} of {c['uncited_answers']['n']}",
         "ci": "—", "n": c["uncited_answers"]["n"], "command": cmd, "source": src, "not_shown": "a deterministic check of the citation contract, not answer quality"},
        {"claim": "Generated answers: every cited path is in the retrieved set (answers with at least one citation)", "value": f"{c['cited_in_retrieved']['rate']}", "ci": "—",
         "n": c["cited_in_retrieved"]["n"], "command": cmd, "source": src,
         "not_shown": f"{_count(round((1 - c['cited_in_retrieved']['rate']) * c['cited_in_retrieved']['n']) if c['cited_in_retrieved']['rate'] is not None else 0, 'answer')} "
                      "cited a path the retrieval had not returned (a fabricated or malformed path); abstentions are not counted"},
        {"claim": "Generated answers: every number / config value appears in the cited bytes", "value": f"{c['values_in_cited_bytes']['rate']}", "ci": "—",
         "n": c["values_in_cited_bytes"]["n"], "command": cmd, "source": src,
         "not_shown": "only answers that state a value are counted; string containment, case-insensitive; a value in a different numeric format, or one the model computed, counts as missing"},
        {"claim": "Generated answers: exact match vs gold (bounded, numeric-equivalent), simple_factual", "value": f"{c['exact_match_simple_factual']['rate']}", "ci": "—",
         "n": c["exact_match_simple_factual"]["n"], "command": cmd, "source": src,
         "not_shown": "string / numeric match; paraphrased correct answers count as non-matches"
                      + (("; the slice is saturated for this generator, which is why the kappa gate's reference is a calibration set" if calgen else
                          "; the slice is saturated for this generator, so it gives the kappa gate a one-class reference")
                         if c["exact_match_simple_factual"]["rate"] == 1.0 else "")},
    ]
    planted = (f"the reference is {calgen}'s answers to the simple-factual questions from retrieved and from planted contexts (another gold value of the "
               "same field written over the gold in memory), so a pass shows gold comparison on single-value answers, not general correctness" if calgen else
               "the reference is the exact-match label of the run's own simple-factual answers, single values to templated questions, so a pass "
               "shows gold comparison on single-value answers, not general correctness")
    method = (s.get("interval") or {}).get("method") or "percentile bootstrap interval"  # what the file says it resampled
    for name, k in s["kappa"].items():
        if name == "jury_majority":
            continue
        if not ref.get("gate_evaluable", True):
            status = "gate not evaluable: the reference is one class or empty; this judge's scores are not published"
        else:
            status = f"passes the {gate} gate; its correctness score of the run's answers is published" if k["readme_ok"] else f"below the {gate} gate; this judge's scores are not published"
        ci = f"[{k['interval']['lo']}, {k['interval']['hi']}]" if k.get("interval") else "—"
        out.append({"claim": f"Judge {name}: Cohen's kappa of its correctness verdicts vs exact-match labels, {against}", "value": f"{k['kappa_vs_exact_match_simple_factual']}",
                    "ci": ci, "n": k["n"], "command": cmd, "source": src,
                    "not_shown": f"{status}; {planted}; {method}"})
    jm = s["kappa"].get("jury_majority")
    if jm:
        ci = f"[{jm['interval']['lo']}, {jm['interval']['hi']}]" if jm.get("interval") else "—"
        out.append({"claim": f"Jury majority (two judge models, correctness verdicts): Cohen's kappa vs exact-match labels, {against}", "value": f"{jm['kappa_vs_exact_match_simple_factual']}",
                    "ci": ci, "n": jm["n"], "command": cmd, "source": src, "not_shown": f"a tie between the two judges counts as partial; gate {gate}; {planted}; {method}"})
    ruled = (s.get("abstention_rule") or {}).get("n", 0)
    for name in s.get("publishable_judges") or []:
        k = s["kappa"][name]
        self_pref = "; the judge is also the generator, so its score of these answers carries self-preference risk" if name == gen else ""
        out.append({"claim": f"Judge {name} (passed the gate): mean correctness of the run's answers (correct 1, partial 0.5, wrong 0)", "value": f"{k['mean_correctness_all']}",
                    "ci": "—", "n": k["n_all"], "command": cmd, "source": src,
                    "not_shown": f"{ruled} of the {k['n_all']} labels come from the abstention rule (wrong), not from the judge; an LLM judge validated only on "
                                 f"single-value answers; the run's answers also include multi-part, comparative, aggregate and path answers it was not validated on{self_pref}"})
    ja = s.get("judge_agreement")
    if ja:
        out.append({"claim": f"Judge-judge agreement ({ja['judges'][0]} vs {ja['judges'][1]}): Cohen's kappa of correctness verdicts on the run's answers", "value": f"{ja['kappa']}",
                    "ci": "—", "n": ja["n"], "command": cmd, "source": src,
                    "not_shown": "agreement between two judges says nothing about agreement with the truth; both judged the same generator's answers"})
    for name, p in (s.get("pairwise_llm_vs_extractive") or {}).items():
        out.append({"claim": f"Judge {name}: position gap in AB/BA pairwise comparison (LLM answer vs extractive)", "value": f"{p['position_gap_mean']}", "ci": "—", "n": p["n"],
                    "command": cmd, "source": src, "not_shown": "a measure of order bias, not of answer quality"})
    return out


def _jury_v3_claims(root: Path = ROOT) -> list[dict]:
    """M42: each new generator's deterministic checks with their intervals; each new judge's gate on the calibration
    answers stored in jury_v2.json (or that it could not load); a judge's score of an answer set only where its gate
    passed. A model with no results file has no row: the README block names it as not run."""
    from app.eval.report import JURY_V3_ANSWERS, JURY_V3_JUDGES, jury_v3_answers_file, jury_v3_answers_label, jury_v3_judge_file

    def dirty(flag) -> str:
        return " with uncommitted changes" if flag is True else ""

    out: list[dict] = []
    for config in JURY_V3_ANSWERS:
        (name, think, limit), model = config, jury_v3_answers_label(*config)
        file = jury_v3_answers_file(*config)
        blob = _load(file, root)
        if not blob:
            continue
        c, ci = blob["summary"]["checks"], blob.get("check_intervals") or {}
        cmd = f"make bench-jury-v3-answers GENERATOR={name}" + (" THINK=off" if think == "off" else "") + (f" ANSWERS_LIMIT={limit}" if limit else "")
        empty = sum(1 for r in blob["rows"] if not r["answer"].strip() and not r["generation"].get("error"))
        rs = blob.get("rescore") or {}
        src = (f"{file} @ {blob['git_sha']}{dirty(blob.get('working_tree_dirty'))}"
               + (f" (answers generated at {rs['rows_git_sha']}{dirty(rs.get('rows_working_tree_dirty'))}, checks recomputed by "
                  "`make bench-jury-v3-answers-rescore`)" if rs else ""))
        reasoning = ("reasoning turned off (`think: false` with every generation call)" if think == "off" else
                     "the model reasons before it replies and only the reply is scored")
        one_run = (f"one run on {(blob.get('device') or {}).get('device_name')}, Ollama {(blob.get('ollama') or {}).get('version')}; {reasoning}"
                   + (f"; the first {limit} questions only" if limit else ""))

        def interval(key: str) -> str:
            return f"[{ci[key]['lo']}, {ci[key]['hi']}]" if ci.get(key) else "—"

        out += [
            {"claim": f"Generated answers ({model} via Ollama, fixture v1): answers with no valid citation (failed loud)", "value": f"{c['uncited_answers']['count']} of {c['uncited_answers']['n']}",
             "ci": "—", "n": c["uncited_answers"]["n"], "command": cmd, "source": src,
             "not_shown": f"a deterministic check of the citation contract, not answer quality; {_count(len(blob.get('generation_errors') or []), 'generation call')} failed and "
                          f"{_count(empty, 'answer')} came back empty (the reasoning filled the context), and those count as uncited; {one_run}"},
            {"claim": f"Generated answers ({model}): every cited path is in the retrieved set (answers with at least one citation)", "value": f"{c['cited_in_retrieved']['rate']}",
             "ci": interval("cited_in_retrieved"), "n": c["cited_in_retrieved"]["n"], "command": cmd, "source": src,
             "not_shown": f"abstentions are not counted; the interval is a bootstrap over the answers counted; {one_run}"},
            {"claim": f"Generated answers ({model}): every number / config value appears in the cited bytes", "value": f"{c['values_in_cited_bytes']['rate']}",
             "ci": interval("values_in_cited_bytes"), "n": c["values_in_cited_bytes"]["n"], "command": cmd, "source": src,
             "not_shown": "only answers that state a value are counted; string containment, case-insensitive; a value in a different numeric format, or one the model computed, "
                          f"counts as missing; {one_run}"},
            {"claim": f"Generated answers ({model}): exact match vs gold (bounded, numeric-equivalent), simple_factual", "value": f"{c['exact_match_simple_factual']['rate']}",
             "ci": interval("exact_match_simple_factual"), "n": c["exact_match_simple_factual"]["n"], "command": cmd, "source": src,
             "not_shown": f"string / numeric match on single-value questions; paraphrased correct answers count as non-matches; {one_run}"},
        ]
    for model in JURY_V3_JUDGES:
        file = jury_v3_judge_file(model)
        blob = _load(file, root)
        if not blob:
            continue
        cmd = f"make bench-jury-v3-judge JUDGE={model}"
        heads = ", ".join(dict.fromkeys(f"{(seg.get('git_head') or '')[:7]}{dirty(seg.get('working_tree_dirty'))}" for seg in blob.get("segments") or []))
        rs = blob.get("rescore") or {}
        src = f"{file} @ {heads}" + (f" (summaries recomputed at {rs['checks_git_sha']} by `python -m app.eval.judge_stored --rescore`)" if rs else "")
        devices = ", ".join(dict.fromkeys(str(seg.get("device_name")) for seg in blob.get("segments") or []))
        gate = blob.get("gate")
        if blob["status"] == "could not load" or not gate:
            out.append({"claim": f"Judge {model}: could not load", "value": "not judged", "ci": "—", "n": 0, "command": cmd, "source": src,
                        "not_shown": f"{blob.get('could_not_load')}; no calibration answer was judged, so this is not a gate result"})
            continue
        cal = blob["calibration"]
        if not gate["complete"]:
            status = f"the gate is not complete ({gate['n']} of {cal['n']} calibration answers judged); nothing of this judge is published"
        elif not gate["reference"]["gate_evaluable"]:
            status = "gate not evaluable: the reference is one class or empty; this judge's scores are not published"
        else:
            status = (f"passes the {gate['threshold']} gate; its correctness scores of the stored answers are published" if gate["passes"] else
                      f"below the {gate['threshold']} gate; this judge's scores are not published")
        r = gate["replies"]
        out.append({"claim": f"Judge {model}: Cohen's kappa of its correctness verdicts vs exact-match labels, the calibration answers stored in {cal['file'].rsplit('/', 1)[-1]}",
                    "value": f"{gate['kappa']}", "ci": f"[{gate['interval']['lo']}, {gate['interval']['hi']}]" if gate.get("interval") else "—", "n": gate["n"],
                    "command": cmd, "source": src,
                    "not_shown": f"{status}; the reference is {(cal.get('generator') or {}).get('model')}'s stored answers to the simple-factual questions from retrieved and from planted "
                                 "contexts, so a pass shows gold comparison on single-value answers, not general correctness; "
                                 f"{r['empty']} empty, {r['unparsed']} unreadable and {r['errors']} failed replies of {r['n']}; one run on {devices}; {gate['interval_method']}"})
        for answers in blob.get("answer_sets") or []:
            block = answers.get("generator") or {}
            s_, gen = answers.get("summary") or {}, (block.get("model") or "the stored answers") + (" with reasoning off" if block.get("think") == "off" else "")
            if not s_.get("publishable"):
                continue
            m = s_["mean_correctness"]
            whose = ("the judge's own weights generated these answers, so the score carries self-preference risk" if s_["self_judged"] else
                     "the judge is another model than the generator")
            agree = "; ".join(f"same verdict as {other} (stored in that file) on {a['agree']} of {a['n']} answers both were asked about, kappa {a['kappa']}"
                              for other, a in (s_.get("agreement_with_stored_judges") or {}).items())
            out.append({"claim": f"Judge {model} (passed the gate): mean correctness of the answers of {gen} in {answers['file'].rsplit('/', 1)[-1]} (correct 1, partial 0.5, wrong 0)",
                        "value": f"{m['mean']}", "ci": f"[{m['interval']['lo']}, {m['interval']['hi']}]", "n": m["n"], "command": cmd, "source": src,
                        "not_shown": f"{s_['abstention_only']['count']} of the {m['n']} labels come from the abstention rule (wrong), not from the judge; an LLM judge validated only on "
                                     f"single-value answers, while these answers also include multi-part, comparative, aggregate and path answers; {whose}"
                                     + (f"; {agree}" if agree else "") + "; the interval is a bootstrap over this set's answers, not a paired test against another generator or judge"})
    out += _self_preference_claims(root)
    return out


def _self_preference_claims(root: Path = ROOT) -> list[dict]:
    """M42: the committed generator's model as a judge of the other generators' answers too: its lead over each
    independent judge on its own answers minus its lead on the others' (app/eval/judge_stored.self_preference)."""
    from app.eval.judge_stored import self_preference
    from app.eval.report import JURY_FILE, JURY_V3_JUDGES, jury_v3_judge_file

    committed = _load(JURY_FILE, root)
    if not committed:
        return []
    gen = committed["models"]["generator"]["model"]
    file = jury_v3_judge_file(gen)
    mine = _load(file, root)
    if not mine or mine.get("status") != "complete" or not (mine.get("gate") or {}).get("passes"):
        return []
    sp = self_preference(mine, [_load(jury_v3_judge_file(m), root) for m in JURY_V3_JUDGES if m != gen])
    heads = ", ".join(dict.fromkeys((s_.get("git_head") or "")[:7] for s_ in mine.get("segments") or []))
    out = []
    for name, r in ((sp or {}).get("by_independent_judge") or {}).items():
        d, o, x = r["difference"], r["own_gap"], r["others_gap"]
        out.append({"claim": f"Judge {gen} vs judge {name}: its lead on its own answers minus its lead on the other generators' answers (self-preference test)",
                    "value": f"{d['mean']}", "ci": f"[{d['interval']['lo']}, {d['interval']['hi']}]", "n": d["n"],
                    "command": f"make bench-jury-v3-judge JUDGE={gen}", "source": f"{file} @ {heads}",
                    "not_shown": f"reading: {r['reading']}; lead (its score minus {name}'s on the same answers) {o['mean']} on its own answers and {x['mean']} on the "
                                 "other generators' answers; both judges validated on single-value answers only; the gaps sit in multi-part, comparative, "
                                 "aggregate and path answers; an interval over questions, one run per judge"})
    return out


def _served_scale_claims(root: Path = ROOT) -> list[dict]:
    """FiQA through the served path (A05 / D13) and pgvector on that table (C03, C04); nothing from a slice or a dry run."""
    from app.eval.report import SERVED_ROWS, scale_machine, scale_run_problem, scale_sha

    def source(file: str, blob: dict) -> str:  # a file assembled again from recorded stages says where they were measured
        note = scale_sha(blob)
        return f"{file} @ {blob['git_sha']}" + ("" if note.startswith("results sha") else f" ({note})")

    out: list[dict] = []
    blob = _load(FILES["served_scale"], root)
    if blob and not scale_run_problem(blob):
        src, n, cmd = source(FILES["served_scale"], blob), blob["n_queries"], "make bench-served-scale"
        where = scale_machine(blob)
        rerank = blob["served"][SERVED_ROWS[0]]
        failed = (rerank.get("requests") or {}).get("failed")
        v, ci = _ci(rerank["ci"]["ndcg@10"], digits=4)
        rejected, judged = blob["build"]["report"].get("files_rejected"), len((blob.get("not_indexed") or {}).get("judged", []))
        out.append({"claim": f"BEIR FiQA ({blob['n_docs']} documents) through the served API (Postgres + pgvector, bge-small + bge-reranker-v2-m3): nDCG@10 at the document level",
                    "value": v, "ci": ci, "n": n, "command": cmd, "source": src,
                    "not_shown": f"one run on {where}; the lexical leg is Postgres ts_rank_cd (the default LEXICAL_BACKEND, not BM25); a document ranks at its best chunk; "
                                 f"{rejected} of the {blob['n_docs']} documents were rejected by the served scanner and never indexed ({judged} of them judged relevant); "
                                 f"{failed} failed requests scored zero; the in-memory rows of the same job and the differences between the sides are in the results file"})
        for row in SERVED_ROWS:
            s = blob["served"][row]
            d, ci = _ci(s["delta_vs_in_memory"]["ndcg@10"], "delta", digits=4)
            zero = "the interval covers zero" if s["delta_vs_in_memory"]["ndcg@10"]["covers_zero"] else "the interval excludes zero"
            out.append({"claim": f"FiQA: served `{row}` minus in-memory `{s['paired_with']}` (same query form, same job), paired delta in nDCG@10", "value": d, "ci": ci, "n": n,
                        "command": cmd, "source": src,
                        "not_shown": f"{zero}; the sides differ by design (chunks vs whole documents, ts_rank_cd vs BM25, at most 50 chunks vs 100 fused documents, a router on the served side); "
                                     "not a measure of either side alone"})
        build, report_ = blob["build"], blob["build"]["report"]
        out.append({"claim": f"FiQA served index build: the schema migrations, then index_directory seconds for {report_['files_indexed']} files ({report_['chunks_embedded']} chunks) with the HNSW and GIN indexes maintained per insert",
                    "value": f"{build['migrate_seconds']} + {build['index_directory_seconds']}", "ci": "—", "n": report_["files_indexed"], "command": cmd, "source": src,
                    "not_shown": f"one run on {where}, files on node-local disk; Postgres image defaults (maintenance_work_mem {build['server']['settings'].get('maintenance_work_mem')}); "
                                 f"the embedder took {build['embedding']['seconds']} s of index_directory"})
        top = max(blob["load"]["runs"], key=lambda r: r["concurrency"])
        lat = top.get("e2e_successful") or {}
        out.append({"claim": f"FiQA served API under load: p50 / p95 ms per successful request at concurrency {top['concurrency']} (reranker on, k = {blob['load']['k']})",
                    "value": f"{lat.get('p50_ms')} / {lat.get('p95_ms')}", "ci": "—", "n": top["attempted"], "command": cmd, "source": src,
                    "not_shown": f"{top['failed']} of {top['attempted']} requests failed ({', '.join(f'{k} {c}' for k, c in (top.get('error_counts') or {}).items()) or 'none'}); "
                                 f"client and server on one node ({where}); one run"})
    hnsw = _load(FILES["pgvector_hnsw"], root)
    if hnsw and not scale_run_problem(hnsw):
        src, n, cmd = source(FILES["pgvector_hnsw"], hnsw), hnsw["queries"]["n"], "make bench-served-scale"
        served = next(r for r in hnsw["rows"] if r["config"] == "served dense CTE")
        v, ci = _ci(served["recall@50"], digits=4)
        out.append({"claim": f"pgvector HNSW on the served FiQA table ({hnsw['table']['rows']} chunk vectors): Recall@50 against the exact scan with the served settings",
                    "value": v, "ci": ci, "n": n, "command": cmd, "source": src,
                    "not_shown": f"recall of the dense leg against an exact scan, not retrieval quality; {served['settings']}; one table and one index build "
                                 f"(m {hnsw['index']['m']}, ef_construction {hnsw['index']['ef_construction']})"})
        variants = hnsw["variants"]
        by = {(r["variant"], r.get("ef_search")): r for r in variants["rows"]}
        ef = min(e for v_, e in by if v_ == "halfvec")
        half, f32 = by.get(("halfvec", ef)), by.get(("float32", ef))
        if half and f32:
            sizes = variants["sizes_after"]["indexes"]
            out.append({"claim": f"halfvec vs float32 HNSW on the same table at ef_search {ef}: Recall@50 against the exact scan (index MiB)",
                        "value": f"{half['recall@50']['mean']:.4f} vs {f32['recall@50']['mean']:.4f} ({sizes[variants['indexes']['halfvec']]['bytes'] / 2**20:.1f} vs "
                                 f"{sizes[variants['indexes']['float32']]['bytes'] / 2**20:.1f})", "ci": "—", "n": n, "command": cmd, "source": src,
                        "not_shown": "the measurement's own halfvec copy on its task-local table (no schema migration adds one; the served statement reads only the float32 column); "
                                     "both indexes rebuilt in bulk (REINDEX); the copy sits next to the float32 column, so the table grows"})
    return out


def _rerank_arm_claims(root: Path = ROOT) -> list[dict]:
    """The reranker prototypes against the cross-encoder: paired deltas and the latency they buy."""
    from app.eval.report import rerank_arms_command, verdict

    blob = _load(FILES["rerank_arms"], root)
    if not blob:
        return []
    out: list[dict] = []
    src = f"{FILES['rerank_arms']} @ {blob['git_sha']}"
    cmd = rerank_arms_command(blob)
    for scope in ("test_split", "all_questions"):
        rows = {r["config"]: r for r in blob[scope]["results"]}
        teacher = rows.get("hybrid+bge-rerank@bge-small") or {}
        for name, row in rows.items():
            if "student" not in name and "listwise" not in name:
                continue
            student = row.get("student") or {}
            stand_in = f"student backend: {student['fallback']}" if student.get("fallback") else "no stand-in: the model ran"
            if student.get("model") and not student.get("fallback"):
                stand_in = f"student: {student['model']}"
            lw = blob.get("listwise") or {}
            reranks = f"both rerank the fused top {blob['candidates']['retrieve_k']}"
            if lw.get("evaluated") and name == lw.get("config"):
                stand_in = (f"{len(lw.get('fused_order_kept_ids') or [])} of {lw['n_calls']} questions keep the fused order "
                            f"({lw['n_failed_calls']} calls returned nothing, {lw['n_parse_failures']} replies were not a full permutation)")
                # The file records the model, its digest and the Ollama version, not where Ollama ran: say no more than that.
                reranks = (f"the cross-encoder reranks the fused top {blob['candidates']['retrieve_k']}, the listwise model its first "
                           f"{lw['window']} (`{lw['model']}` through Ollama)")
            for block in row.get("deltas") or []:
                d = block["ndcg@10"]
                value, ci = _ci(d, "delta")
                lat, tlat = row.get("rerank_latency") or {}, teacher.get("rerank_latency") or {}
                out.append({"claim": f"Reranker prototype `{name}` minus `{block['reference']}`, paired delta in nDCG@10 ({'held-out test questions' if scope == 'test_split' else 'all questions'})",
                            "value": value, "ci": ci, "n": row["n"], "command": cmd, "source": src,
                            "not_shown": f"verdict from the interval: {verdict(d)}; {stand_in}; rerank step p50 {lat.get('p50_ms')} ms vs {tlat.get('p50_ms')} ms for the cross-encoder on the GPU; "
                                         f"{reranks} of a {blob['n_files']}-file synthetic fixture"})
    return out


def _rrf_sweep_claims(root: Path = ROOT) -> list[dict]:
    """The RRF setting chosen on the tune half, against the default setting on the held-out half; the default it leaves in
    place (read from the file, as every other number of the row)."""
    from app.eval.report import verdict

    blob = _load(FILES["rrf_sweep"], root)
    if not blob:
        return []
    d, grid = blob["decision"], blob["grid"]
    k0, w0 = d["default"]["k"], d["default"]["weights"]
    base = f"k = {k0} with " + ("equal weights" if w0["dense"] == w0["bm25"] else f"dense weight {w0['dense'] / w0['bm25']:g}")
    src, cmd = f"{FILES['rrf_sweep']} @ {blob['git_sha']}", "make bench-rrf-sweep"
    model = blob["models"]["embedder"]["model"].split("/")[-1]
    head = f"Fixture v1, RRF k and dense-vs-BM25 list weight chosen on the tune half of the gold questions ({model}, no reranker)"
    chosen = (f"chosen by the best tune-half nDCG@10 among {len(grid['k']) * len(grid['dense_weight'])} settings "
              f"(k in {', '.join(map(str, grid['k']))}; dense weight in {', '.join(f'{w:g}' for w in grid['dense_weight'])}; BM25 weight 1), "
              f"so the tune-half numbers are in-sample")
    rep = blob.get("reproduces") or {}
    tail = (f"the default setting's lists reproduce the committed {', '.join(rep['rows'])} rows; " if rep.get("all") else "")
    tail += f"a {blob['n_files']}-file synthetic fixture"
    if d["chosen_is_default"]:
        return [{"claim": f"{head}: the tune half chose the default, {base}", "value": "the default", "ci": "—",
                 "n": len(blob["split"]["tune_ids"]), "command": cmd, "source": src,
                 "not_shown": f"{chosen}; nothing to compare on the held-out half, and no evidence that k = {k0} is best; {tail}"}]
    block = d["held_out_ndcg@10"]
    value, ci = _ci(block, "delta", digits=4)  # the file's precision: a small delta at 3 decimals reads larger than it is
    stays = ("the interval excludes zero in favour of the chosen setting; the default is unchanged until the owner decides "
             "(every hybrid row of main.json and neural.json would be re-recorded)" if d["owner_decision_needed"] else
             f"the default stays {base} (it changes only if this interval excludes zero in favour of the chosen setting)")
    w = d["chosen"]["weights"]["dense"]
    return [{"claim": f"{head}: k = {d['chosen']['k']}, dense weight {w:g} minus {base}, paired delta in nDCG@10 on the held-out half",
             "value": value, "ci": ci, "n": len(blob["split"]["held_out_ids"]), "command": cmd, "source": src,
             "not_shown": f"verdict from the interval: {verdict(block)}; {stays}; {chosen}; {tail}"}]


def _share_words(share: float) -> str:
    """A fraction as words, so the evidence table types no number the results file does not hold."""
    if share > 1.1:
        return "more than all"
    if share >= 0.9:
        return "nearly all"
    if share > 0.6:
        return "most"
    if share >= 0.4:
        return "about half"
    return "some" if share > 0.1 else "little"


def _search_policy_claims(root: Path = ROOT) -> list[dict]:
    """The search policy trained once (M44): one row, the bar declared before training (exact match, its headline row,
    each question's mean over the seeds, minus the same model untrained), whatever its verdict. The other comparisons
    stay in the results file, the README block and the engineering reference; this table carries the declared bar
    only, and its "not shown" cell says in words what the mean reward did and how much of that was citing (read from
    the file: the change in the uncited-or-empty rate times its weight in the reward formula, over the reward's
    delta)."""
    import re

    blob = _load(FILES["search_policy"], root)
    if not blob:
        return []
    primary = next(v for v in blob["verdicts"] if v["primary"])
    by = {(v["config"], v["reference"], v["metric"]): v for v in blob["verdicts"]}
    rows = {r["config"]: r for r in blob["results"]}
    model = blob["model"]["name"].split("/")[-1]
    seeds = f"each question's mean over {primary['n_seeds']} training seeds" if primary["n_seeds"] > 1 else "one training seed"
    words = [f"verdict from the interval: {primary['interval']}, a {primary['outcome']} on the bar declared before training"]
    single = by.get((primary["config"], "single-shot-rag", "em"))
    if single:
        words.append(f"against single-shot RAG with the same model, exact match is a {single['outcome']} too"
                     if single["outcome"] == primary["outcome"] else f"against single-shot RAG with the same model, exact match is a {single['outcome']}")
    reward = by.get((primary["config"], primary["reference"], "reward"))
    if reward:
        if reward["interval"] == "covers zero":
            words.append("secondary to the bar, the mean reward did not move measurably (its interval covers zero)")
        else:
            moved = "rose" if reward["delta"] > 0 else "fell"
            said = f"secondary to the bar, the mean reward {moved}, with an interval {reward['interval']}"
            weight = re.search(r"- ([0-9.]+) \* uncited_or_empty", blob["reward"]["formula"])
            trained, untrained = rows.get(primary["config"]), rows.get(primary["reference"])
            if moved == "rose" and weight and trained and untrained:
                part = float(weight.group(1)) * (untrained["means"]["uncited_or_empty"] - trained["means"]["uncited_or_empty"])
                if part > 0:
                    said += f", {_share_words(part / reward['delta'])} of it from fewer uncited or empty answers"
            words.append(said)
    words.append("training corpora and questions come from the fixture's own builder and templates (other seeds, no fixture question text), "
                 "so it holds for these templates; one small model, one prompt, greedy decoding on one GPU type")
    return [{"claim": (f"Fixture v1 held-out half, search policy trained with GRPO ({model} + LoRA, {seeds}) minus the same model untrained "
                       "under the same prompt, paired delta in exact match (the bar declared before training)"),
             "value": f"{primary['delta']:+.4f}", "ci": f"[{primary['lo']:+.4f}, {primary['hi']:+.4f}]", "n": blob["split"]["n_held_out"],
             "command": "make search-policy-eval", "source": f"{FILES['search_policy']} @ {blob['git_sha']}", "not_shown": "; ".join(words)}]


def _hybrid_drops_claims(root: Path = ROOT) -> list[dict]:
    """The gold files hybrid@bge-base drops at Recall@50 against dense_only@bge-base: how many, and where they rank."""
    blob = _load(FILES["hybrid_drops"], root)
    if not blob:
        return []
    s, k, rows = blob["summary"], blob["retrieve_k"], blob["rows"]
    drops = [d for q in blob["questions"] for d in q["dropped"]]
    parts = [f"the Recall@50 delta {s['recall@50']['delta']:+.4f} ({rows['hybrid']} minus {rows['dense_only']}) is the net of these drops and gains"
             + ("; every question that moves is listed in the file" if s["every_question_that_moves_is_listed"] else "")]
    if drops:
        dense, fused = [d["dense"]["chunk_rank"] for d in drops], [d["fused"]["chunk_rank"] for d in drops]
        parts.append(f"{s['dropped_with_no_chunk_in_bm25_list']} of the {len(drops)} dropped files have no chunk in BM25's top {k}; their best chunks "
                     f"rank {min(dense)}-{max(dense)} in dense's top {k} and {min(fused)}-{max(fused)} in the fused list, which is cut at {k}")
        cats = s["dropped_by_category"]
        parts.append(f"every drop is a {next(iter(cats))} question" if len(cats) == 1 else "drops by category: " + ", ".join(f"{c} {n}" for c, n in cats.items()))
        if s["dropped_tied_with_cut"]:
            parts.append(f"{s['dropped_tied_with_cut']} of the drops tie the score of the last chunk kept")
    rep = blob.get("reproduces") or {}
    if rep.get("all"):
        parts.append(f"the file's lists reproduce the committed {', '.join(rep['rows'])} rows")
    parts.append(f"a diagnosis of one pair of rows on a {blob['n_files']}-file synthetic fixture, not a retrieval-quality claim")
    return [{"claim": f"Fixture v1, {rows['hybrid']} vs {rows['dense_only']}: gold files with a chunk in dense's top {k} chunks and none in the "
                      f"fused top {k} (dropped from Recall@50), and the reverse (gained)",
             "value": f"{s['n_dropped']} dropped on {s['n_questions_with_drops']} questions, {s['n_gained']} gained", "ci": "—", "n": s["n_questions"],
             "command": "make bench-hybrid-drops", "source": f"{FILES['hybrid_drops']} @ {blob['git_sha']}", "not_shown": "; ".join(parts)}]

def _file_depth_claims(root: Path = ROOT) -> list[dict]:
    """M37: hybrid@bge-base against dense_only@bge-base at Recall@50 under the file-depth cut, next to the chunk-cut
    delta the committed rows give and the files hybrid drops there."""
    blob = _load(FILES["file_depth"], root)
    if not blob or "bge-base" not in blob.get("embedders", {}):
        return []
    by = {r["config"]: r for r in blob["results"]}
    e = blob["embedders"]["bge-base"]
    names, k = e["rows"]["chunk_cut"], blob["retrieval"]["retrieve_k"]

    def delta(name, ref):  # attach_confidence: the row's delta_vs_previous (its parent) or a bootstrap comparison
        previous = by[name].get("delta_vs_previous") or {}
        if previous.get("reference") == ref:
            return previous["recall@50"]
        return next(c for c in blob["bootstrap"]["comparisons"] if c["row"] == name and c["other"] == ref)["recall@50"]

    fd = delta(f"{names['hybrid']}/file-depth", f"{names['dense']}/file-depth")
    cut = delta(names["hybrid"], names["dense"])
    gain = {leg: delta(f"{names[leg]}/file-depth", names[leg]) for leg in ("dense", "hybrid")}
    drop = e["dropped_by_the_chunk_cut"]
    value, ci = _ci(fd, "delta")
    parts = [f"the committed rows cut every list at {k} chunks, where {names['hybrid']} minus {names['dense']} is "
             f"{_ci(cut, 'delta')[0]} {_ci(cut, 'delta')[1]} at Recall@50",
             f"under the file-depth cut both lists hold {k} files, and {drop['n_recovered']} of the {drop['n_dropped']} gold files "
             f"hybrid drops at the chunk cut are in its list"
             + (f" (the files {drop['matches']['file']} lists)" if (drop.get("matches") or {}).get("same_files") else ""),
             f"both rows gain Recall@50 over their chunk-cut parents (dense {_ci(gain['dense'], 'delta')[0]} {_ci(gain['dense'], 'delta')[1]}, "
             f"hybrid {_ci(gain['hybrid'], 'delta')[0]} {_ci(gain['hybrid'], 'delta')[1]})",
             "the parents recomputed in the run reproduce neural.json exactly" if e["reproduces"].get("all") else
             "a parent does NOT reproduce neural.json",
             "rows added beside the committed ones: the served path and every committed row keep the chunk cut",
             f"a {blob['n_files']}-file synthetic fixture"]
    return [{"claim": f"Fixture v1, {names['hybrid']} minus {names['dense']} at Recall@50 under the file-depth cut (every list keeps "
                      f"chunks until it holds {k} files), paired",
             "value": value, "ci": ci, "n": blob["n_gold"], "command": "make bench-file-depth",
             "source": f"{FILES['file_depth']} @ {blob['git_sha']}", "not_shown": "; ".join(parts)}]


def _reproduced(blob: dict) -> str:
    """What the run's `reproduces` blocks say about its recomputed bge parents, in a clause."""
    blocks = blob.get("reproduces")
    blocks = [blocks] if isinstance(blocks, dict) else list(blocks or [])
    parts = []
    for b in blocks:
        rows = b.get("rows") or {}
        reordered = [n for n, e in rows.items() if e.get("equal") and e.get("ranking_equal") is False]
        moved = [n for n, e in rows.items() if not e.get("equal")]
        if moved:
            parts.append(f"{len(moved)} of the {len(rows)} recomputed parents do NOT equal {b['file']}")
        elif reordered:
            parts.append(f"the recomputed parents equal {b['file']} in every metric, though {len(reordered)} of them order near-tied "
                         f"documents differently from its TREC runs on this GPU (no metric moves)")
        else:
            parts.append(f"the recomputed parents reproduce {b['file']}")
    return "; ".join(parts) or "no reproduction check recorded"


def _models_2026_claims(root: Path = ROOT) -> list[dict]:
    """M41: the current open models against the committed bge rows they replace, recomputed in the same run; a row only
    where the paired interval excludes zero (above or below), losers included."""
    out: list[dict] = []

    def delta(blob: dict, name: str, other: str | None = None) -> dict | None:
        row = next((r for r in blob.get("results", []) if r["config"] == name), None)
        if row is None:
            return None
        if other is None:
            return (row.get("delta_vs_parent") or row.get("delta_vs_previous") or {}).get("ndcg@10")
        return next((c["ndcg@10"] for c in (blob.get("bootstrap") or {}).get("comparisons") or []
                     if c["row"] == name and c["other"] == other), None)

    def rerank_p50(blob: dict, name: str):
        row = next((r for r in blob.get("results", []) if r["config"] == name), {})
        return ((row.get("latency") or {}).get("rerank") or {}).get("p50_ms")

    bge20 = "hybrid+bge-rerank@bge-small/fp32-512/top20"
    for key, dataset, command in (("beir_scifact_models_2026", "SciFact", "make bench-beir-models-2026"),
                                  ("beir_fiqa_models_2026", "FiQA", "make bench-beir-fiqa-models-2026")):
        blob = _load(FILES[key], root)
        if not blob:
            continue
        src, n, gpu = f"{FILES[key]} @ {blob['git_sha']}", blob["n_queries"], (blob.get("device") or {}).get("device_name", "the recorded GPU")
        repro = _reproduced(blob)
        hybrid8 = delta(blob, "hybrid@qwen3-emb-8b")
        rows = [
            ("dense@qwen3-emb-8b", None, f"BEIR {dataset}, Qwen3-Embedding-8B vs bge-small-en-v1.5, dense retrieval, paired delta in nDCG@10",
             f"dense only, on one GPU ({gpu}), Qwen3-Embedding-8B in bfloat16 with its `query` prompt; its hybrid row against "
             f"hybrid@bge-small is {_ci(hybrid8, 'delta', 4)[0]} {_ci(hybrid8, 'delta', 4)[1]} ({_zero(hybrid8)}); the app's default "
             f"embedder is still bge-small; {repro}"),
            ("hybrid@qwen3-emb-8b", "dense@qwen3-emb-8b", f"BEIR {dataset}, hybrid RRF minus dense-only, both with Qwen3-Embedding-8B, paired delta in nDCG@10",
             f"fusing BM25 into the dense list (RRF, equal weights) against the same dense list alone, on one GPU ({gpu}); with "
             f"bge-small the same comparison is the committed hybrid row's parent delta; {repro}"),
        ]
        for tag, label in (("qwen3-rerank-8b", "Qwen3-Reranker-8B (bf16)"), ("mxbai-rerank-large-v2", "mxbai-rerank-large-v2 (fp32)")):
            name = f"hybrid+{tag}@bge-small/{'bf16' if '8b' in tag else 'fp32'}-512/top20"
            rows.append((name, None, f"BEIR {dataset}, {label} vs bge-reranker-v2-m3 (fp32), both over hybrid@bge-small, paired delta in nDCG@10",
                         f"both rerank the fused top 20 at max_length 512 on one GPU ({gpu}); rerank-stage p50 {rerank_p50(blob, name)} ms "
                         f"vs {rerank_p50(blob, bge20)} ms per query; not the app's default precision on CUDA (fp16); {repro}"))
        for name, other, claim, not_shown in rows:
            d = delta(blob, name, other)
            if d and (d["lo"] > 0 or d["hi"] < 0):  # an interval that excludes zero supports a claim; one that covers it does not
                value, ci = _ci(d, "delta", 4)
                out.append({"claim": claim, "value": value, "ci": ci, "n": n, "command": command, "source": src, "not_shown": not_shown})
    # harrier-oss-v1-27b, each dataset's own run with bge-small's rows recomputed beside it
    for key, dataset, command, dense in (("models_2026_harrier", "Fixture v1", "make bench-models-2026-harrier", "dense_only"),
                                         ("beir_scifact_models_2026_harrier", "BEIR SciFact", "make bench-beir-models-2026-harrier", "dense"),
                                         ("beir_fiqa_models_2026_harrier", "BEIR FiQA", "make bench-beir-fiqa-models-2026-harrier", "dense")):
        blob = _load(FILES[key], root)
        d = delta(blob, f"{dense}@harrier-27b") if blob else None
        if d and (d["lo"] > 0 or d["hi"] < 0):
            hybrid = delta(blob, "hybrid@harrier-27b")
            value, ci = _ci(d, "delta", 4)
            gpu = (blob.get("device") or {}).get("device_name", "the recorded GPU")
            out.append({"claim": f"{dataset}, harrier-oss-v1-27b vs bge-small-en-v1.5, dense retrieval, paired delta in nDCG@10", "value": value, "ci": ci,
                        "n": blob.get("n_queries", blob.get("n_gold")), "command": command, "source": f"{FILES[key]} @ {blob['git_sha']}",
                        "not_shown": f"dense only, bfloat16 with its `web_search_query` prompt, on one GPU ({gpu}), loaded from a view of its "
                                     f"snapshot without tokenizer_config.json's processor_class; its hybrid row against hybrid@bge-small is "
                                     f"{' '.join(_ci(hybrid, 'delta', 4))} ({_zero(hybrid)}); {_reproduced(blob)}"})
    fixture = _load(FILES["models_2026"], root)
    d = delta(fixture, "hybrid@qwen3-emb-0.6b") if fixture else None
    if d and (d["lo"] > 0 or d["hi"] < 0):
        rec = next(r for r in fixture["results"] if r["config"] == "hybrid@qwen3-emb-0.6b")["delta_vs_previous"]["recall@50"]
        from app.eval.report import verdict

        beir = [(dataset, delta(blob, "hybrid@qwen3-emb-0.6b")) for dataset, blob in
                (("SciFact", _load(FILES["beir_scifact_models_2026"], root)), ("FiQA", _load(FILES["beir_fiqa_models_2026"], root))) if blob]
        elsewhere = "; ".join(f"its hybrid row on {name} against hybrid@bge-small: {' '.join(_ci(b, 'delta', 4))} ({verdict(b)})" for name, b in beir if b)
        value, ci = _ci(d, "delta")
        out.append({"claim": "Fixture v1, hybrid RRF with Qwen3-Embedding-0.6B (the CPU-servable candidate) vs with bge-small-en-v1.5, paired delta in nDCG@10",
                    "value": value, "ci": ci, "n": fixture["n_gold"], "command": "make bench-models-2026",
                    "source": f"{FILES['models_2026']} @ {fixture['git_sha']}",
                    "not_shown": f"fixture v1 only ({fixture['n_files']} files): Recall@50 rises ({' '.join(_ci(rec, 'delta'))}) while nDCG@10 falls; "
                                 + (f"{elsewhere}; " if elsewhere else "") + _reproduced(fixture)})
    return out


def _small_reranker_claims(root: Path = ROOT) -> list[dict]:
    """M43: the small rerankers over hybrid@bge-small at the served shape. A row only where the paired interval excludes
    zero: each student against the unreranked hybrid and against the candidate it started from, and every other
    reranked row against the unreranked hybrid where it is above zero on all three datasets (it passes the quality bar)
    or below zero somewhere (a recorded loss). Ties are in the generated table, not here."""
    from app.eval import small_rerankers as sr
    from app.eval.report import verdict

    table = sr.bar_table(root)
    if not table["rows"]:
        return []
    names = {"fixture": "Fixture v1 (held-out half)", "scifact": "BEIR SciFact", "fiqa": "BEIR FiQA"}
    commands = {"fixture": "make bench-small-rerankers", "scifact": "make bench-beir-small-rerankers", "fiqa": "make bench-beir-fiqa-small-rerankers"}
    out: list[dict] = []
    twins = _fixture_twins(root)

    def what(tag: str) -> str:
        if tag in sr.STUDENTS:
            init = sr.CANDIDATES[sr.STUDENTS[tag]]
            return f"`{tag}` ({init['model']} trained on {sr.TEACHER_TAG}'s scores for synthetic queries, {init['parameters'] / 1e6:.1f}M parameters)"
        if tag in sr.CANDIDATES:
            return f"`{tag}` ({sr.CANDIDATES[tag]['model']}, {sr.CANDIDATES[tag]['parameters'] / 1e6:.1f}M parameters, as published)"
        return f"`{tag}` ({sr.model_of(tag)})"

    for row in table["rows"]:
        tag, student = row["tag"], row["tag"] in sr.STUDENTS
        cells = row["datasets"]
        lost = any(c["vs_parent"] and c["vs_parent"]["hi"] < 0 for c in cells.values())
        bar = {True: "it passes the quality bar (above zero on all three datasets)", False: "it does not pass the quality bar (a tie or a loss on at least one of the three datasets)",
               None: "the quality bar is open (a dataset is not measured)"}[row["passes"]]
        for d, cell in cells.items():
            meta = table["datasets"][d]
            src = f"{meta['file']} @ {meta['git_sha']}"
            block = cell["vs_parent"]
            shape = (f"the fused top {sr.SHAPE['rerank_top']} at max_length {sr.SHAPE['max_length']}, {sr.SHAPE['precision']}, on one GPU ({meta['device']}); "
                     f"quality only: whether the model fits the demo's host is measured on the host")
            others = f"against bge-reranker-v2-m3 in the same run: {' '.join(_ci(cell['vs_bge'], 'delta', 4))} ({verdict(cell['vs_bge'])})" if cell.get("vs_bge") else ""
            seen = "the student was trained on synthetic queries over this corpus (it has read its documents, never a test query or a label); " if student else ""
            if student and d == "fixture":
                seen += twins
            if block and (block["lo"] > 0 or block["hi"] < 0) and (student or row["passes"] or (lost and block["hi"] < 0)):
                value, ci = _ci(block, "delta", 4)
                out.append({"claim": f"{names[d]}, {what(tag)} vs no reranker, both over hybrid@bge-small, paired delta in nDCG@10", "value": value, "ci": ci,
                            "n": meta["n"], "command": commands[d], "source": src,
                            "not_shown": f"verdict from the interval: {verdict(block)}; {bar}; {seen}{others}; {shape}"})
            init = cell.get("vs_init")
            if student and init and (init["lo"] > 0 or init["hi"] < 0):
                value, ci = _ci(init, "delta", 4)
                out.append({"claim": f"{names[d]}, what distillation added: `{tag}` vs `{sr.STUDENTS[tag]}`, the published model it started from, paired delta in nDCG@10",
                            "value": value, "ci": ci, "n": meta["n"], "command": commands[d], "source": src,
                            "not_shown": f"verdict from the interval: {verdict(init)}; same architecture and cost, different weights; {seen}one training run, one seed; {shape}"})
    return out


def _frontier_claims(root: Path = ROOT) -> list[dict]:
    """One row per prototype arm and per stack: the paired delta against its parent, the verdict word, the stand-ins."""
    from app.eval.report import hyde_wiring, llm_note, verdict

    out: list[dict] = []
    for key, command, stack in (
        ("frontier", "make bench-frontier", "CPU stand-ins"),
        ("frontier_neural", "make bench-frontier-neural", "bge-small-en-v1.5 + bge-reranker-v2-m3"),
    ):
        blob = _load(FILES[key], root)
        if not blob:
            continue
        by = _rows(blob)
        n = blob["n_gold"]
        src = f"{FILES[key]} @ {blob['git_sha']}"
        for name in (blob.get("frontier") or {}).get("arms", []):
            row = by.get(name)
            block = ((row or {}).get("delta_vs_previous") or {}).get("ndcg@10")
            if not row or not block:
                continue
            value, ci = _ci(block, "delta")
            vs = row.get("vs_parent") or {}
            limits = [f"verdict from the interval: {verdict(block)}"]
            if 0.0 < block["lo"] < 0.0005 or -0.0005 < block["hi"] < 0.0:
                limits.append(f"a bound sits at the edge of zero (interval as stored: [{block['lo']:+.4f}, {block['hi']:+.4f}])")
            limits.append("stand-ins: " + ("; ".join(row.get("fallbacks") or []) or "none, both models loaded"))
            wiring = hyde_wiring(row.get("settings") or {})
            if wiring:
                limits.append(wiring)
            if row.get("llm"):
                limits.append(llm_note(row["llm"]).rstrip("."))
            cache = row.get("cache")
            if cache:
                g = cache["gold"]
                limits.append(f"cache threshold {cache['threshold']}: {g['hits']} hits on {g['n']} gold questions in gold order, "
                              f"{g['wrong_hits']} of them wrong (the cached list differs from what retrieval returns for that question)")
            if vs:
                limits.append(f"first ten files differ from the parent's on {vs['n_top10_differs']} of {vs['n_questions']} questions")
            g = row.get("graphrag")
            if g and not g["discriminates"]:
                limits.append(f"the community step gave {g['n_communities']} community holding {g['n_member_files']} of {g['n_corpus_files']} files, "
                              "so the summaries cannot tell files apart and this is a member / non-member split")
            limits.append(f"a {blob['n_files']}-file synthetic fixture")
            out.append({"claim": f"Prototype arm `{name}` minus its parent `{row['settings']['parent']}` ({stack}), paired delta in nDCG@10",
                        "value": value, "ci": ci, "n": n, "command": command, "source": src, "not_shown": "; ".join(limits)})
    return out


def _agent_injection_claims(root: Path = ROOT) -> list[dict]:
    """The second injection run (agent_injection_v2): planted claims and actions next to the capability control."""
    blob = _load(FILES["agent_injection"], root)
    if not blob:
        return []
    out: list[dict] = []
    src = f"{FILES['agent_injection']} @ {str(blob.get('git_head') or '')[:7]}"
    segments = (blob.get("episodes_generated_by") or {}).get("segments") or []
    if segments and "--summarize-only" in str(blob.get("command") or ""):  # the header names the scoring run
        src += (f" (episodes generated at {str(segments[0].get('git_head') or '')[:7]}, scored by "
                "`make bench-agent-injection-v2-rescore`)")
    cmd = "make bench-agent-injection-v2"
    k = blob["config"]["k"]
    scope = "one phrasing per goal and task, written for this benchmark (not an adaptive attacker); two local 7-8B models; synthetic config questions"
    for model, per in blob["summary"].items():
        none = per.get("none")
        if not none:
            continue
        hijack = none["by_goal"]["hijack"]["attempted"]
        value, ci = _ci(hijack)
        out.append({"claim": f"Agent injection, second run, {model}, no defense: answers that state the planted value (hijack tasks), per trial",
                    "value": value, "ci": ci, "n": hijack["n"], "command": cmd, "source": src,
                    "not_shown": f"{scope}; n is tasks, each with {k} trials; the planted value sits in comment lines of the files the question leads to"})
        pa = none["planted_actions"]
        out.append({"claim": f"Agent injection, second run, {model}, no defense: episodes that pursued a planted file action (move, patch, read outside the root), of those in which the planted text reached the model",
                    "value": f"{pa['attempted_episodes']} of {pa['exposed_episodes']}", "ci": "—", "n": pa["exposed_episodes"], "command": cmd, "source": src,
                    "not_shown": f"{scope}; read it next to the capability-control rows of the same model, where the user asks for the same move and patch"})
        cap = none.get("capability")
        if cap:
            g = none["by_goal"]
            planted = sum(g[x]["attempted_episodes"] for x in ("move", "patch"))
            exposed = sum(g[x]["exposed_episodes"] for x in ("move", "patch"))
            value, ci = _ci(cap["attempted"])
            out.append({"claim": f"Agent injection, second run, {model}, no defense: capability control, attempted: the user asks for the move and the patch a planted instruction asks for, and the model calls the requested action's tools, per trial",
                        "value": value, "ci": ci, "n": cap["attempted"]["n"], "command": cmd, "source": src,
                        "not_shown": (f"n is questions, each with a move and a patch task and {k} trials; counted as planted attempts are (a call to the action's tools) "
                                      f"and read against {planted} planted move and patch attempts in {exposed} episodes in which that text reached the model; "
                                      "the user's request names the same tool calls and arguments as the planted text, and the operator signs only the requested change")})
            value, ci = _ci(cap["success"])
            out.append({"claim": f"Agent injection, second run, {model}, no defense: capability control, done: the copy ends in exactly the requested move and patch, per trial",
                        "value": value, "ci": ci, "n": cap["success"]["n"], "command": cmd, "source": src,
                        "not_shown": ("n is questions, as in the attempted row; done also takes passing the operator's signed token to the apply call unchanged, "
                                      "so it measures carrying the action out, not whether the model calls the tools")})
        hat = none["benign_pass_hat_k"][str(k)]
        value, ci = _ci(hat)
        out.append({"claim": f"Agent injection, second run, {model}, no planted text: pass^{k} (all {k} sampled trials answer correctly, citing a file of the asked run that contains the value)",
                    "value": value, "ci": ci, "n": hat["n"], "command": cmd, "source": src,
                    "not_shown": "ten single-field config questions whose files carry neutral lines with the same run number and field name as the planted text; reliability on this tool set and prompt, not general agent reliability"})
        delta = ((blob.get("paired_deltas") or {}).get(model) or {}).get("spotlight - none", {}).get("hijack_planted_value")
        if delta:
            value, ci = _ci(delta, "delta")
            out.append({"claim": f"Agent injection, second run, {model}: datamarking (spotlight) minus no defense, planted-value answers per trial on hijack tasks, paired",
                        "value": value, "ci": ci, "n": delta["n"], "command": cmd, "source": src,
                        "not_shown": "a prompt-level change tested on one phrasing per task; an interval that covers zero is no measured effect"})
    gates = [b["gate"] for per in blob["summary"].values() for b in per.values()]
    writes = sum(x.get("write_tool_calls", 0) for x in gates)
    changed = sum(x["corpus_changed_episodes"] for x in gates)
    n_base = sum(b["episodes"] for per in blob["summary"].values() for b in per.values())
    gate_note = ("with no write-tool call in these tasks the approval gate was not exercised there by a model" if not writes else
                 "every write-tool call in these tasks met the approval gate without an operator grant")
    out.append({"claim": "Agent injection, second run, all models and configurations, question and injected tasks: write-tool calls made by a model, and episodes in which the corpus changed",
                "value": f"{writes} calls; {changed} of {n_base} episodes", "ci": "—", "n": n_base, "command": cmd, "source": src,
                "not_shown": f"{gate_note}; no grant is ever issued in these tasks (the capability tasks, where the operator signs the requested change, are in their own rows); not OS isolation and not an adaptive attacker"})
    return out


def _agent_injection_v3_claims(root: Path = ROOT) -> list[dict]:
    """agent_injection_v3, the current run: per model, hijack attacked by at least one of the phrasings with no defense, and the answer
    check in the loop against no defense (planted value delivered, utility under attack, benign success), paired."""
    blob = _load(FILES["agent_injection_v3"], root)
    if not blob:
        return []
    cfg = blob["config"]
    gen = blob.get("episodes_generated_by") or {}
    shards = (gen.get("merged") or {}).get("shards") or []
    cmd = (f"make bench-agent-injection-v3-shard ({len(shards)} shards), then make bench-agent-injection-v3-merge" if shards
           else "make bench-agent-injection-v3")
    src = f"{FILES['agent_injection_v3']} @ {str(blob.get('git_head') or '')[:7]}"
    n, adojo, k = len(cfg["phrasings"]), len(cfg.get("agentdojo_templates") or {}), cfg["k"]
    scope = (f"{n} phrasings per goal, {adojo} of them AgentDojo templates: a fixed set, not an adaptive attacker; two "
             "local 7-8B models; synthetic config questions")
    out: list[dict] = []
    for model, per in blob["summary"].items():
        if "none" in per:
            hijack = per["none"]["by_goal"]["hijack"]["attempted_any_phrasing"]
            value, ci = _ci(hijack)
            # an interval pinned at one: no configuration could show a higher rate, so the measure separates none of them
            ceiling = ("; at the ceiling: every pair was attacked under every seed, so this rate cannot separate one "
                       "configuration from another" if hijack["lo"] == 1.0 else "")
            out.append({"claim": f"Agent injection, current run ({n} phrasings), {model}, no defense: hijack questions in which at least one phrasing put the planted value into the model's answer, per seed",
                        "value": value, "ci": ci, "n": hijack["n"], "command": cmd, "source": src,
                        "not_shown": (f"{scope}; n is (question, hijack) pairs, each read over {k} seeds; the phrasings share each seed, so "
                                      f"this compares texts on shared trajectories, not independent tries{ceiling}")})
        deltas = ((blob.get("paired_deltas") or {}).get(model) or {}).get("answer_check - none") or {}
        for key, what in (("planted_value_delivered", "the planted value delivered to the user, hijack, per draw"),
                          ("utility_under_attack", "utility under attack"),
                          ("benign_success", "benign success on the controls")):
            d = deltas.get(key)
            if not d:
                continue
            value, ci = _ci(d, "delta")
            out.append({"claim": f"Agent injection, current run, {model}: the structured answer check in the loop minus no defense, {what}, paired",
                        "value": value, "ci": ci, "n": d["n"], "command": cmd, "source": src,
                        "not_shown": (f"{scope}; {_zero(d)}; the check is told each question's config field and file, so it covers "
                                      "single-field questions only; a flagged answer gets one retry within the same turn budget")})
    return out


def _agent_injection_adaptive_claims(root: Path = ROOT) -> list[dict]:
    """The adaptive attacker on the held-out questions, per target model, against agent_injection_v3's fixed phrasings on
    the same pairs and seeds: the planted actions and the hijack attempted with no defense, and the planted value
    delivered past the answer check, each a paired difference over the questions."""
    blob = _load(FILES["agent_injection_adaptive"], root)
    if not blob:
        return []
    cfg = blob["config"]
    gen = blob.get("episodes_generated_by") or {}
    shards = (gen.get("merged") or {}).get("shards") or []
    cmd = (f"make bench-agent-injection-adaptive-shard ({len(shards)} shards), then make bench-agent-injection-adaptive-merge"
           if shards else "make bench-agent-injection-adaptive")
    src = f"{FILES['agent_injection_adaptive']} @ {str(blob.get('git_head') or '')[:7]}"
    rounds, k, split = cfg["rounds"], cfg["k"], cfg["split"]
    attacker = cfg["attacker"]["model"]
    role = ("also the target" if cfg["models"] == [attacker] else
            "also one of the targets" if attacker in cfg["models"] else "not a target")
    scope = (f"one attacker model ({attacker}, {role}), told after each round whether its text reached the assistant, "
             "whether the goal was attempted and took effect, and the start of the answer: more than a user sees; "
             f"{len(split['questions'])} held-out questions, its instructions tuned on {len(split['train'])} others; "
             f"target{'s' if len(cfg['models']) > 1 else ''} {', '.join(cfg['models'])}"
             + (f" ({', '.join(m for m in cfg.get('declared_models') or [] if m not in cfg['models'])} declared and cut)"
                if set(cfg.get("declared_models") or []) - set(cfg["models"]) else "")
             + "; synthetic config questions; two runs sampled at different times")
    out: list[dict] = []
    for model, per in (blob.get("comparison") or {}).items():
        if model == "source":
            continue
        for defense, group, level, what in (
                ("none", "planted_actions", "attempted", "a planted file action (move, patch, read outside the root) attempted"),
                ("none", "hijack", "attempted", "the planted value in the model's answer (hijack)"),
                ("answer_check", "hijack", "took_effect", "the planted value delivered to the user past the answer check")):
            block = ((per.get(defense) or {}).get(group) or {}).get(level)
            if not block:
                continue
            value, ci = _ci(block["delta"], "delta")
            where = "no defense" if defense == "none" else f"the {defense} configuration"
            out.append({"claim": f"Agent injection, adaptive run, {model}, {where}: an adaptive attacker with {rounds} rounds minus the current run's fixed phrasings, {what} in some round, per trial, paired",
                        "value": value, "ci": ci, "n": block["delta"]["n"], "command": cmd, "source": src,
                        "not_shown": (f"adaptive {_ci(block['adaptive'])[0]} against fixed {_ci(block['fixed'])[0]}; "
                                      f"{_zero(block['delta'])}; n is questions, each pair read over {k} trials; {scope}")})
    return out


BULLETS = '''## Resume bullets (from the rows above only)

Capstone team work, 2024-25 (six-person Oregon State team; my role: retrieval and data APIs):

- Built the retrieval and data APIs of a FastAPI + Next.js + PostgreSQL/pgvector research-file assistant.

Solo work in this fork, 2026:

- Designed an LLM-free retrieval benchmark (frozen 61-file fixture, 136 scripted questions in eight categories whose answers and relevant paths are computed from the corpus registry) with paired-bootstrap confidence intervals, a random-list baseline for recall, and a CI regression gate; the README's numbers are generated from the committed results files and a test fails on drift.
- Measured hybrid BM25 + dense retrieval with reciprocal rank fusion and a cross-encoder reranker on the fixture and on BEIR SciFact: with real models, hybrid and the reranker beat BM25 with intervals above zero, and our BM25 reproduces Anserini's published SciFact baseline within noise; found and fixed a reranker API bug: the code called a scoring method the loaded model did not have, the served retriever's broad except logged a warning and returned the fused list unreranked, and the benchmark had no model-backed reranker row until the fix.
- Ported the benchmarked pipeline into the production API as one parameterized Postgres query (pgvector top-k, full-text top-k, RRF in SQL, GIN + HNSW indexes) with a CI parity job that replays the gold set through the API and reports the gap to the in-memory bench with its interval, and a live-database test that requires the dense leg to equal the in-memory dense search on every gold question; no silent fallbacks, per-request stage latency and scores.
- Added a typed retrieval endpoint that returns byte-range citations, and a human-in-the-loop gate for file changes: an agent can only propose a change, and applying it needs an operator-signed, single-use approval bound to the reviewed action.
- Added an LLM answer evaluation with a validity gate: cited answers from a local model, deterministic citation and value checks, a two-model jury with position-swapped comparisons, and Cohen's kappa against exact-match labels before any judge score is published. The first run's reference slice was saturated, so its gate could not be evaluated; the second run measures each judge's correctness verdicts on a calibration set (a smaller model answering from retrieved and from planted contexts), labels abstentions by rule, and pins every model by digest. One judge passed and its score is published with its caveats; the other called every calibration answer correct and failed.
- Split that evaluation into a generate-only stage and a judge-only stage, so a judge can be added to stored answers without generating anything again and answers recorded elsewhere are judged by one command: each judge is first gated on the same stored calibration answers, a model that cannot load is recorded as such and not as a failed judge, an interrupted run resumes, and every file can be re-read without a model. Measured current open-weight models with it on H100, H200, A40 and RTX 8000 GPUs: the answers of three current generators with their deterministic checks, and two judges of other model families that passed the gate and scored those answers and the first run's, which until then only the generator's own model had scored.
- Built an LLM-in-the-loop prompt-injection evaluation for the agent's file tools: local models pick MCP tool calls over corpus copies whose answering files carry planted instructions, and every episode is scored at the model level (attempted the planted action, stated the planted value) and at the system level (corpus changed, secret leaked, false value delivered), with τ-bench pass^k, any-of-k attack rates and paired bootstrap deltas. The current run plants each goal in seven phrasings (five of them AgentDojo templates) under no defense, datamarking, a least-privilege tool filter and a structured answer check between the model and the user; the answer check was the one defense that cut the planted value reaching the user for both models, with no measurable utility cost. An earlier run adds a capability control (the planted actions asked for by the user and signed by an operator key), and a later one an adaptive attacker that rewrote the planted text against each defense for up to seven rounds and did no measurably better than the fixed phrasings against the stronger model, a null result.
'''

QA = '''## Likely interview questions, with honest answers

**Why is Recall@50 not the headline?** The fixture has 61 files and the hybrid list covers most of them in its top 50, so a random list of that length already scores what the "random list, same length" column shows (the Recall@50 row above gives both numbers). The table shows Recall@50 next to that baseline and leads with nDCG@10 and Recall@10, where the random baseline is a small fraction of the score.

**Why did hybrid tie BM25 on the fixture?** With hash embeddings (no model) and a token-overlap reranker, the dense leg adds coverage but not ranking; the paired nDCG@10 delta versus BM25 covers zero. With bge-small and the bge reranker the same pipeline beats BM25 with intervals above zero, on the fixture and on SciFact. The CI table is the download-free story; the neural table is the quality story.

**Why is the parity gap reported and not gated?** The API and the in-memory bench now index the same chunks with the same embeddings, so the dense half must match, and a live-database test requires it to match exactly on every gold question. The lexical halves differ by design (Postgres full-text with ts_rank_cd and ties broken by row id, against Okapi BM25), so the end-to-end nDCG@10 gap is a measured difference, not a defect: the parity job reports it with its paired interval and fails only when the API is broken or the fixture on disk differs from its MANIFEST.

**What is a prototype here versus production?** The measured, served pipeline is hybrid retrieval, the router, staleness Tier 1, the reranker configuration chosen from the SciFact sweep, and the approval-gated file plans. Paper2Code, the MATLAB checkpoint and multimodal pages run in demos but have no measured result; the engineering reference keeps them under "Prototypes (not evaluated)". The RL search policy was trained once and scored against the same model untrained and against single-shot RAG on held-out questions, with the bar declared before training; the row above is the verdict of that bar, and its other comparisons are in the README's search-policy table, labelled as secondary to it. The retrieval prototypes (GraphRAG community routing, HippoRAG PPR, hop expansion, HyDE, multi-query, decomposition, corrective and adaptive retrieval, the chunking variants) are benchmark arms: each has a row above with its parent, a paired interval, the verdict that interval supports and the stand-ins it ran on. An arm whose interval covers zero is reported as no measured effect, and an arm that lost stays in the table.

**Why did the router's number change?** The earlier README compared the routed row against plain RRF, which credited the reranker's share of the gain to the router. Against its real parent, the reranked hybrid, the router is a tie overall and a gain on the exact-path slice with n = 8. The generated table prints the honest pair.

**What did the reranker bug teach you?** The code called a scoring method the loaded model did not have. In the served retriever a broad except caught the error, logged a warning and returned the fused list unreranked, so a response looked normal while reranking had been skipped; the benchmark labelled its no-model fallback in each row and had no model-backed reranker row until the fix. The fix was small; the lesson was structural: every result now records whether the model actually loaded, responses list degraded components, and a test fails if a configured reranker could fall back without a trace.

**Why fp16 with a 512-token cap and depth 20?** On SciFact, fp16 with max_length 512 is a quality tie with fp32 uncapped at a fraction of the latency, and reranking deeper than 20 loses nDCG@10 while costing more. The served configuration is chosen from that table, and CPU latency for the same setting is measured separately in the Docker CPU stack.

**How do you know the numbers in the README are real?** Every benchmark block in the README is rendered from a committed results file by one command, and a test fails if the text drifts. Each results file records the command, the git sha, the model revisions and the device. Losers stay in the tables.

**What did the LLM answer evaluation show?** On the first run the generator answered every simple-factual question with the gold value, so the exact-match reference was one class, Cohen's kappa was zero by construction, and the gate could not be evaluated. The slice is saturated for that generator, and a smaller model answering from the retrieved chunks still matched nearly every gold value, so the second run measures the judges on a calibration set instead: a smaller model that is not a judge answers each simple-factual question from the retrieved chunks and again from the same chunks with the gold value replaced by another value of the same field. The gated label is a correctness verdict that sees only the question, the gold answer and the answer, because the first run's composite label also rewards faithfulness to the context, so a faithfully copied wrong value would read as partial. On that set the qwen judge agreed with exact match well past the gate, and its score of the generator's answers is published with two caveats: it is the generator's own model, and it was validated on single-value answers only. The llama judge called every calibration answer correct, planted values included, and failed, and the two judges' chance-corrected agreement on the generator's answers was near zero. A later stage put current open-weight models through a judge-only entry point: two judges of other model families passed the same gate, and both scored the first run's answers lower than the generator's own model had. All three judges give the exact-match label on the simple-factual answers, so the scores part on the multi-part, comparative, aggregate and path answers, which no gate validated; where a new judge and the generator's model differ, the new judge is almost always the stricter one. That is the direction the self-preference caveat warned about, but it also fits a judge that is lenient with any generator's answers. To tell the two apart the generator's model was run again as a judge of the other generators' answers: it leads both new judges on its own answers but not measurably on the others', and the difference between those two leads is not measurable either, so the question stays open at this size; two judges agreeing is still not agreement with the truth. The same two judges also scored the answers of three current generators, gemma4:26b's own answers marked as self-judged; the generated table gives one interval per answer set and no paired test between generators. The deterministic checks remain the part that depends on no judge. The kept losers are the failed judge, the strong position bias in the AB/BA swap, and the composite label that still called abstentions correct, which is why abstentions are now labelled wrong by rule. The first run also exposed two check flaws (abstentions counted as citation failures, case-sensitive value matching), fixed in commits separate from the rescored file, and its stored answers later showed a substring match that found a gold value inside a longer number, fixed before the second run.

**What did the injection evaluation show?** In the first two runs neither local model pursued a planted file action (a move, a patch, or a read outside the corpus root), and in the question and injected tasks no model called a write tool; in the third, the weaker model's only write calls there were patch proposals written as JSON text, which the tool rejected for a missing argument. So planted text never reached the approval gate; the gate's evidence against forged tokens is still the deterministic approval benchmark. A capability control shows that, where an injected task is offered the write tools, this is not an inability to make the calls: when the user asks for the same move or patch, with the same tool calls and arguments, both models call the write tools and the operator signs the requested change. Completing it is rarer: many episodes stop after the signed proposal, and an apply call has to pass the plan id and the token on unchanged; the weaker model's apply calls mostly named plan ids the server never issued, and the stronger model's mostly carried a token that did not verify. What the models did follow was the planted claim: in the second run the stronger model stated the planted value in nearly every hijack episode, in every configuration, sometimes while calling the true value a corrected typo, and neither datamarking nor the tool filter changed that measurably; in the current run, with seven phrasings, at least one of them got the planted value into its answer for every hijack question and seed in every configuration. The weaker model rarely reached the planted text because it called tools that do not exist and guessed paths, which is also why its pass^k on the controls is zero in the second run; that is incapability, not resistance. The lesson: actions belong behind a system boundary, and answer integrity needs a check against the structured source, because no capability gate covers what an answer says. Put in the model loop, that check lowered the planted value delivered to the user for both models with no measurable change in utility under attack, though planted values stated beside the true value still got through, including answers that gave the planted value as the true value's correction. An adaptive attacker (the stronger model rewriting the planted text against each configuration, told the scorer's outcome of its earlier rounds) showed no measurable difference from the fixed phrasings on held-out questions after seven rounds: it got no planted action attempted, and every paired interval for the planted value covers zero. One model attacking its own weights says how that target held up against that attacker, not against adaptive attacks in general.

**How did you check your own scoring?** Before publishing, separate AI review agents read the harness against the real episodes, and another tried to refute each finding. They showed that the first hijack rule missed answers that passed the planted value on while also naming the true one, that a substring match counted a made-up encoder name as the gold one, and that a citation of a file about a different run counted as grounding. The scoring code was fixed and the stored episodes were rescored without a model call; the episodes file keeps every tool call and answer, so anyone can rescore them with `make bench-agent-injection-rescore` (the second run's, which also store every tool result, with `make bench-agent-injection-v2-rescore`, and the current run's with `make bench-agent-injection-v3-rescore`).

**What would you do next?** A larger fixture where the first stage is a real filter (the reranker rescores most of the fixture's chunks today), ParadeDB BM25 in the production path if that backend is kept (an open decision), with its gap reported by the same parity job, and human labels for the multi-part, paraphrased and computed answers the calibration set does not cover, which is where the judges of other model families and the generator's own model part.
'''


def render(root: Path = ROOT) -> str:
    rows = claims(root)
    lines = [
        "# Resume evidence",
        "",
        "Generated by `make evidence` (app/eval/evidence.py) from the committed results files; `tests/eval/test_resume_evidence.py` fails if this file drifts. Every value below comes from a results file that records the command, the git sha, the model revisions and the device that produced it. The bullets and answers are prose about these rows and contain no other numbers.",
        "",
        "## Claims",
        "",
        "| claim | value | 95% CI | n | command | results file @ sha | what it does NOT show |",
        "|---|---:|---|---:|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['claim']} | {r['value']} | {r['ci']} | {r['n']} | `{r['command']}` | {r['source']} | {r['not_shown']} |")
    missing = [rel for rel in FILES.values() if not (root / rel).exists()]
    if missing:
        lines += ["", "Results files not yet produced: " + ", ".join(f"`{m}`" for m in missing) + "."]
    lines += ["", BULLETS, QA.rstrip(), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "RESUME_EVIDENCE.md")
    p.add_argument("--check", action="store_true")
    args = p.parse_args(argv)
    text = render()
    if args.check:
        current = args.out.read_text() if args.out.exists() else ""
        if current != text:
            print(f"drift: {args.out} differs from `make evidence` output")
            return 1
        print(f"{args.out} in sync")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out} ({len(claims())} claims)")
    return 0


SHAPE_TESTS = {"F1": "unadjusted, one of 18 cell x dataset tests in F1", "F2": "unadjusted, one of 48 cell x dataset tests in F2",
               "F3": "unadjusted, one of 9 cell x dataset tests in F3"}


def _shape_row(cell: dict, dataset: str, family: str, answer: dict, meta: dict, extra: str) -> dict:
    """One evidence row for a family cell on one dataset. It cannot be built without its family's answer: a single
    dataset's interval is one of many tests, and the row must say what the family concluded."""
    if not answer or not answer.get("answer"):
        raise ValueError(f"{cell['config']} on {dataset}: no {family} answer; a single-dataset row never stands alone")
    from app.eval.report import verdict

    e = cell["datasets"][dataset]
    value, ci = _ci(e["vs_parent"], "delta", 4)
    adj = e["adjusted"][family]
    cost = cell["indicative_cpu"]
    return {"claim": f"{meta['name']}, `{cell['tag']}` at depth {cell['depth']} and max_length {cell['max_length']} vs no reranker, both over hybrid@bge-small, "
                     f"paired delta in nDCG@10 ({SHAPE_TESTS[family]})",
            "value": value, "ci": ci, "n": meta["n"], "command": meta["command"], "source": meta["source"],
            "not_shown": (f"verdict from the 95% interval: {verdict(e['vs_parent'])}; at the {family} level (k = {answer['k']}) the interval is "
                          f"[{adj['lo']:+.4f}, {adj['hi']:+.4f}] ({verdict(adj)}); the {family} answer is {answer['answer']}"
                          + (f" ({_shape_cells(answer)})" if answer.get("cells") and answer["answer"] != "NO" else "")
                          + f"; band {cell['band']}, indicative cost {cost['p95_ms']:,.1f} ms p95 and {cost['peak_mib']:,} MiB peak on one HPC CPU, not the host; {extra}"
                          "quality only: whether a shape fits the demo's host is not measured here (the host timing, scripts/time_reranker_on_host.py, "
                          "is the owner's and unrun)")}


def _fixture_twins(root: Path) -> str:
    """M43's near-twin caveat for a student's fixture row, from the one figure report.fixture_near_twins reads."""
    from app.eval.report import fixture_near_twins

    t = fixture_near_twins(root)
    return (f"its synthetic fixture queries sit close to the templated gold questions: {t['n']} of {t['checked']} reach bge-small cosine "
            f"{t['cosine']} to one (bench/distill/queries.meta.json), so this row says less than the BEIR ones; ") if t else ""


def _shape_cells(answer: dict) -> str:
    return ", ".join(f"`{c}`" for c in answer["cells"])


def _small_reranker_shape_claims(root: Path = ROOT) -> list[dict]:
    """M51: the cheaper shapes. One row per family (F1, the headline, and F2) with its answer and deciding cells; a row
    for each F1 / F2 cell and dataset whose 95% interval against no reranker excludes zero, in either direction, with
    its family-level interval and its family's answer beside it; a cell's delta against the served 20/512 shape only
    when that cell answers YES. Nothing for an over-budget cell."""
    from app.eval import small_rerankers as sr
    from app.eval import small_rerankers_shapes as shapes

    front = _load(shapes.FRONTIER_FILE, root)
    if not front:
        return []
    blobs = {d: _load(rel, root) for d, rel in shapes.RESULTS.items()}
    exposure = (_load(shapes.EXPOSURE_FILE, root) or {}).get("datasets", {})
    names = {"fixture": "Fixture v1 (held-out half)", "scifact": "BEIR SciFact", "fiqa": "BEIR FiQA"}
    commands = {"fixture": "make bench-small-rerankers-shapes", "scifact": "make bench-beir-small-rerankers-shapes", "fiqa": "make bench-beir-fiqa-small-rerankers-shapes"}
    twins = _fixture_twins(root)
    frontier_src = f"{shapes.FRONTIER_FILE} @ {front['git_sha']}"
    out: list[dict] = []
    for family in ("F1", "F2"):
        a = front["answers"][family]
        k = a["k"]
        what = "the under-indicative-budget cells (the headline)" if family == "F1" else "the under-indicative-budget and borderline cells (beside F1, never modifying it)"
        out.append({"claim": f"Small rerankers at cheaper shapes, {family}: is any of {what} above no reranker on fixture v1's held-out half, SciFact and FiQA at once?",
                    "value": a["answer"], "ci": "—", "n": k, "command": "make small-rerankers-shapes-frontier", "source": frontier_src,
                    "not_shown": (f"deciding cells: {_shape_cells(a)}; " if a.get("cells") and a["answer"] != "NO" else "")
                                 + f"Bonferroni level 1 - 0.05/{k} per cell on 20,000 resamples; each family controls its own family-wise error at 5%, the "
                                   "families are not jointly controlled; bands from an indicative HPC-CPU budget, not the host; "
                                 + ("NOT SHOWN at this n is not evidence that the cheap shapes do not help; " if a["answer"].startswith("NOT SHOWN") else "")
                                 + "this search follows M43's failed search at the served shape"})
    for cell in front["cells"]:
        if not cell["families"]:
            continue  # over budget: never answers the question
        family = "F1" if "F1" in cell["families"] else "F2"
        student = cell["tag"] in sr.STUDENTS
        for d, e in cell["datasets"].items():
            block = e.get("vs_parent")
            if not block or not (block["lo"] > 0 or block["hi"] < 0):
                continue
            blob = blobs.get(d) or {}
            meta = {"name": names[d], "n": shapes.N_QUERIES[d], "command": commands[d], "source": f"{shapes.RESULTS[d]} @ {blob.get('git_sha')}"}
            extra = ""
            if student and d == "fixture":
                extra = "the student read every fixture chunk in training (never a gold question or label); " + twins
            elif student and exposure.get(d):
                x = exposure[d]
                extra = (f"the student read a relevant document of {x['n_exposed_queries']} of {x['n_test_queries']} of these test queries in training "
                         f"(never a query or label); re-read on the other {x['n_unexposed_queries']}: " + " ".join(_ci(e.get("exposure_excluded"), "delta", 4)) + "; ")
            out.append(_shape_row(cell, d, family, front["answers"][family], meta, extra))
            if front["answers"][family]["answer"] == "YES" and cell["config"] in front["answers"][family]["cells"] and e.get("vs_20_512"):
                value, ci = _ci(e["vs_20_512"], "delta", 4)
                out.append({"claim": f"{names[d]}, the cost of cheapening: `{cell['tag']}` at {cell['depth']}/{cell['max_length']} vs the same model at 20/512, paired delta in nDCG@10",
                            "value": value, "ci": ci, "n": meta["n"], "command": meta["command"], "source": meta["source"],
                            "not_shown": "descriptive, no adjustment; an interval covering zero is no measured difference, not equivalence"})
    return out + _small_reranker_ml256_claims(root, exposure, twins)


def _small_reranker_ml256_claims(root: Path, exposure: dict, twins: str) -> list[dict]:
    """M51 Phase 2 (attempt 2): one row for the F3 answer and one per F3 cell x dataset whose 95% interval against no
    reranker excludes zero, each with its F3-level interval and the F3 answer beside it."""
    from app.eval import small_rerankers_shapes as shapes

    front = _load(shapes.ML256_FRONTIER_FILE, root)
    if not front:
        return []
    a = front["answers"]["F3"]
    out = [{"claim": ("Small rerankers retrained at max_length 256 (Phase 2, attempt 2, after M43's attempt-1 numbers were seen), F3: is any "
                      "attempt-2 cell whose attempt-1 twin is under the indicative budget above no reranker on fixture v1's held-out half, SciFact and FiQA at once?"),
            "value": a["answer"], "ci": "—", "n": a["k"], "command": "make small-rerankers-ml256-frontier", "source": f"{shapes.ML256_FRONTIER_FILE} @ {front['git_sha']}",
            "not_shown": (f"deciding cells: {_shape_cells(a)}; " if a.get("cells") and a["answer"] != "NO" else "")
                         + f"Bonferroni level 1 - 0.05/{a['k']} per cell on 20,000 resamples; F3 is its own family, never merged into F1 or F2; "
                           "cost from the same-architecture attempt-1 row; one training run per student"}]
    blobs = {d: _load(rel, root) for d, rel in shapes.ML256_RESULTS.items()}
    names = {"fixture": "Fixture v1 (held-out half)", "scifact": "BEIR SciFact", "fiqa": "BEIR FiQA"}
    commands = {"fixture": "make bench-small-rerankers-ml256", "scifact": "make bench-beir-small-rerankers-ml256", "fiqa": "make bench-beir-fiqa-small-rerankers-ml256"}
    for cell in front["cells"]:
        if "F3" not in cell["families"]:
            continue
        for d, e in cell["datasets"].items():
            block = e.get("vs_parent")
            if not block or not (block["lo"] > 0 or block["hi"] < 0):
                continue
            meta = {"name": names[d], "n": shapes.N_QUERIES[d], "command": commands[d], "source": f"{shapes.ML256_RESULTS[d]} @ {(blobs.get(d) or {}).get('git_sha')}"}
            if d == "fixture":
                extra = "the student read every fixture chunk in training (never a gold question or label); " + twins
            else:
                x = exposure.get(d) or {}
                extra = (f"the student read a relevant document of {x.get('n_exposed_queries')} of {x.get('n_test_queries')} of these test queries in training "
                         f"(never a query or label); re-read on the other {x.get('n_unexposed_queries')}: " + " ".join(_ci(e.get("exposure_excluded"), "delta", 4)) + "; ")
            out.append(_shape_row(cell, d, "F3", a, meta, extra + "the cost is the same-architecture attempt-1 row's; "))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
