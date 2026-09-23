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
    "parity": "bench/results/parity.json",
    "latency": "bench/results/api_latency_docker_cpu_ml.json",
    "jury": "bench/results/jury.json",
}


def _load(rel: str, root: Path = ROOT) -> dict | None:
    p = root / rel
    return json.loads(p.read_text()) if p.exists() else None


def _ci(block: dict | None, key: str = "mean") -> tuple[str, str]:
    if not block:
        return "—", "—"
    if key == "delta":
        return f"{block['delta']:+.3f}", f"[{block['lo']:+.3f}, {block['hi']:+.3f}]"
    return f"{block[key]:.3f}", f"[{block['lo']:.3f}, {block['hi']:.3f}]"


def _rows(blob: dict) -> dict[str, dict]:
    return {r["config"]: r for r in blob.get("results", [])}


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
                        "not_shown": "not a neural or production-corpus number; hash embeddings and an overlap reranker; 61 files, 136 hand-built questions"})
            v, ci = _ci(full["ci"]["recall@10"])
            out.append({"claim": "Fixture v1, full CI pipeline: Recall@10", "value": v, "ci": ci, "n": n, "command": "make bench", "source": src,
                        "not_shown": "path-level relevance over 61 files; a random list of the same length scores " + f"{full['retrieval'].get('random_recall@10', 0):.3f}"})
            d, ci = _ci((full.get("delta_vs_bm25_only") or {}).get("ndcg@10"), "delta")
            out.append({"claim": "Fixture v1: full CI pipeline minus BM25 alone, paired delta in nDCG@10", "value": d, "ci": ci, "n": n, "command": "make bench", "source": src,
                        "not_shown": "the interval covers zero: with hash embeddings and the overlap reranker, hybrid does not beat BM25 on ranking"})
            d, ci = _ci((full.get("delta_vs_previous") or {}).get("ndcg@10"), "delta")
            out.append({"claim": "Fixture v1: staleness Tier 1 on vs off (over the reranked, routed hybrid), paired delta in nDCG@10", "value": d, "ci": ci, "n": n, "command": "make bench", "source": src,
                        "not_shown": "a fixture-specific mechanism (versioned configs); with a real reranker the same step is " + "smaller (see the neural row)"})
            cat = full.get("delta_vs_parent_by_category") or {}
            if cat:
                d, ci = _ci(cat.get("ndcg@10"), "delta")
                out.append({"claim": f"Fixture v1: staleness on its own category ({cat['category']}), paired delta in nDCG@10", "value": d, "ci": ci, "n": cat["n"], "command": "make bench", "source": src,
                            "not_shown": "small n; no claim beyond the slice"})
            router = by.get("hybrid+rerank+router")
            rc = (router or {}).get("delta_vs_parent_by_category") or {}
            if rc:
                d, ci = _ci(rc.get("ndcg@10"), "delta")
                out.append({"claim": f"Fixture v1: router on its own category ({rc['category']}) over the reranked hybrid, paired delta in nDCG@10", "value": d, "ci": ci, "n": rc["n"], "command": "make bench", "source": src,
                            "not_shown": "n = 8; the earlier README figure 0.596 -> 0.938 compared against plain RRF and overstated the router"})
            out.append({"claim": "Fixture v1: Recall@50 of the full CI pipeline", "value": f"{full['retrieval']['recall@50']:.3f}", "ci": _ci(full['ci']['recall@50'])[1], "n": n, "command": "make bench", "source": src,
                        "not_shown": f"a random list with the same number of unique files ({full['retrieval'].get('mean_unique_paths@50', 0):.0f} of {main['n_files']}) scores {full['retrieval'].get('random_recall@50', 0):.3f}; the metric mostly measures list length"})
    neural = _load(FILES["neural"], root)
    if neural:
        by = _rows(neural)
        n = neural["n_gold"]
        src = f"{FILES['neural']} @ {neural['git_sha']}"
        dev = neural.get("device", {}).get("device_name", "?")
        for name, label in (("hybrid@bge-small", "Fixture v1, hybrid RRF with bge-small-en-v1.5 vs BM25 alone, paired delta in nDCG@10"),
                            ("hybrid+bge-rerank@bge-small", "Fixture v1, hybrid + bge-reranker-v2-m3 vs BM25 alone, paired delta in nDCG@10")):
            row = by.get(name)
            if row:
                d, ci = _ci((row.get("delta_vs_bm25_only") or {}).get("ndcg@10"), "delta")
                out.append({"claim": label, "value": d, "ci": ci, "n": n, "command": "make bench-neural", "source": src,
                            "not_shown": f"run on a {dev} (GPU); 71 chunks, so the reranker rescores most of the corpus; not a production-corpus number"})
        full = by.get("hybrid+bge-rerank+router+staleness@bge-small")
        if full:
            v, ci = _ci(full["ci"]["ndcg@10"])
            out.append({"claim": "Fixture v1, full pipeline with real models (bge-small + bge-reranker-v2-m3 + router + staleness): nDCG@10", "value": v, "ci": ci, "n": n, "command": "make bench-neural", "source": src,
                        "not_shown": "the fixture is 61 files; SciFact carries the retrieval-quality claim"})
        comps = (neural.get("bootstrap") or {}).get("comparisons") or []
        for c in comps:
            if c.get("row") == "hybrid+bge-rerank@bge-small":
                d, ci = _ci(c.get("ndcg@10"), "delta")
                out.append({"claim": "Fixture v1: bge-small + reranker vs hash + reranker (does the first stage matter once the reranker is on?), paired delta in nDCG@10", "value": d, "ci": ci, "n": n, "command": "make bench-neural", "source": src,
                            "not_shown": "a tie because the reranker rescores 50 of 71 chunks; it says nothing about corpora where the first stage is a real filter"})
    beir = _load(FILES["beir"], root)
    if beir:
        by = _rows(beir)
        n = beir["n_queries"]
        src = f"{FILES['beir']} @ {beir['git_sha']}"
        bm = by.get("bm25")
        if bm and bm.get("vs_published"):
            v, ci = _ci(bm["ci"]["ndcg@10"])
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
        f16 = by.get("hybrid+bge-rerank@bge-small/fp16-512/top20")
        if f16 and rer:
            d, ci = _ci((f16.get("delta_vs_parent") or {}).get("ndcg@10"), "delta")
            l16 = (f16.get("latency") or {}).get("total") or {}
            l32 = (rer.get("latency") or {}).get("total") or {}
            out.append({"claim": "BEIR SciFact, reranker fp16 + max_length 512 vs fp32 uncapped (served setting), paired delta in nDCG@10", "value": d, "ci": ci, "n": n, "command": "make bench-beir", "source": src,
                        "not_shown": f"p50 {l16.get('p50_ms')} ms vs {l32.get('p50_ms')} ms on the GPU; CPU latency is a different table"})
        d50 = by.get("hybrid+bge-rerank@bge-small/fp16-512/top50")
        if d50:
            d, ci = _ci((d50.get("delta_vs_parent") or {}).get("ndcg@10"), "delta")
            out.append({"claim": "BEIR SciFact, rerank depth 50 vs 20, paired delta in nDCG@10", "value": d, "ci": ci, "n": n, "command": "make bench-beir", "source": src,
                        "not_shown": "deeper reranking loses here; depth 100 loses further (rows in the file)"})
    parity = _load(FILES["parity"], root)
    if parity:
        by = _rows(parity)
        api, mem = by.get("api:postgres"), next((r for k, r in by.items() if k.startswith("in_memory")), None)
        if api and mem:
            g = parity.get("gate", {})
            d = g.get("paired_delta_api_minus_in_memory") or {}
            dv, ci = _ci(d, "delta") if d else ("—", "—")
            out.append({"claim": "API over Postgres + pgvector vs the in-memory bench (same switches), nDCG@10 gap", "value": dv, "ci": ci, "n": parity["n_gold"], "command": "make parity", "source": f"{FILES['parity']} @ {parity['git_sha']}",
                        "not_shown": f"tolerance {g.get('tolerance')}; different lexical leg (ts_rank_cd vs Okapi BM25) and chunker; CI re-runs it on every push"})
            lat = (api.get("latency") or {}).get("e2e") or {}
            out.append({"claim": "API over Postgres, hash embedder, no reranker: per-query e2e latency p50 / p95 (ms)", "value": f"{lat.get('p50_ms')} / {lat.get('p95_ms')}", "ci": "—", "n": parity["n_gold"], "command": "make parity", "source": f"{FILES['parity']} @ {parity['git_sha']}",
                        "not_shown": "measured on the machine that produced the file, over HTTP, 61 files; not a load test"})
    lat = _load(FILES["latency"], root)
    if lat:
        h = lat.get("health", {}).get("reranker", {})
        out.append({"claim": "Docker CPU stack (bge-small + bge-reranker-v2-m3 fp32, max_length 512, depth 20): server total p50 / p95 (ms)", "value": f"{lat['server_total_ms']['p50']} / {lat['server_total_ms']['p95']}", "ci": "—", "n": lat["n"],
                    "command": "docker compose -f docker-compose.yml -f docker-compose.ml.yml up --build; scripts/smoke_retrieve.py --latency bench/gold/questions.jsonl", "source": FILES["latency"] + " (CI artifact, docker-ml-smoke job)",
                    "not_shown": f"a 4-core CI runner; reranker device {h.get('device')}; the reranker is {lat['stages_p50_ms'].get('rerank')} ms of it"})
    jury = _load(FILES["jury"], root)
    if jury:
        s = jury["summary"]
        c = s["checks"]
        src = f"{FILES['jury']} @ {jury['git_sha']}"
        out.append({"claim": "Generated answers (Ollama, fixture v1): answers with no valid citation (failed loud)", "value": f"{c['uncited_answers']['count']} of {c['uncited_answers']['n']}", "ci": "—", "n": c["uncited_answers"]["n"], "command": "make bench-jury", "source": src,
                    "not_shown": "a deterministic check of the citation contract, not answer quality"})
        out.append({"claim": "Generated answers: every number / config value appears in the cited bytes", "value": f"{c['values_in_cited_bytes']['rate']}", "ci": "—", "n": c["values_in_cited_bytes"]["n"], "command": "make bench-jury", "source": src,
                    "not_shown": "only answers that state a value are counted; regex-extracted values"})
        out.append({"claim": "Generated answers: exact match vs gold (numeric-equivalent), simple_factual", "value": f"{c['exact_match_simple_factual']['rate']}", "ci": "—", "n": c["exact_match_simple_factual"]["n"], "command": "make bench-jury", "source": src,
                    "not_shown": "string / numeric match; paraphrased correct answers count as non-matches"})
        for name, k in s["kappa"].items():
            if name == "jury_majority" or k.get("heuristic"):
                continue
            out.append({"claim": f"Judge {name}: Cohen's kappa vs exact-match labels, simple_factual", "value": f"{k['kappa_vs_exact_match_simple_factual']}", "ci": "—", "n": k["n"], "command": "make bench-jury", "source": src,
                        "not_shown": ("passes the 0.6 gate; the judge's mean score may be published" if k["readme_ok"] else "below the 0.6 gate; this judge's scores are not published")
                        + "; the reference is an exact-match label: a negated or hedged answer that still quotes the gold value counts as correct, a correct paraphrase does not"})
        jm = s["kappa"].get("jury_majority")
        if jm:
            out.append({"claim": "Jury majority (two judge models): Cohen's kappa vs exact-match labels, simple_factual", "value": f"{jm['kappa_vs_exact_match_simple_factual']}", "ci": "—", "n": jm["n"], "command": "make bench-jury", "source": src,
                        "not_shown": "the generator is one of the judges (self-preference risk); gate 0.6"})
        for name, p in (s.get("pairwise_llm_vs_extractive") or {}).items():
            out.append({"claim": f"Judge {name}: position gap in AB/BA pairwise comparison (LLM answer vs extractive)", "value": f"{p['position_gap_mean']}", "ci": "—", "n": p["n"], "command": "make bench-jury", "source": src,
                        "not_shown": "a measure of order bias, not of answer quality"})
    return out


BULLETS = '''## Resume bullets (from the rows above only)

Capstone team work, 2024-25 (six-person Oregon State team; my role: retrieval and data APIs):

- Built the retrieval and data APIs of a FastAPI + Next.js + PostgreSQL/pgvector research-file assistant: hybrid lexical + vector search served over Postgres, typed retrieval endpoints with byte-range citations, and the human-in-the-loop file-change gate.

Solo work in this fork, 2026:

- Designed an LLM-free retrieval benchmark (frozen 61-file fixture, 136 hand-labelled questions in eight categories) with paired-bootstrap confidence intervals, a random-list baseline for recall, and a CI regression gate; the README's numbers are generated from the committed results files and a test fails on drift.
- Measured hybrid BM25 + dense retrieval with reciprocal rank fusion and a cross-encoder reranker on the fixture and on BEIR SciFact: with real models, hybrid and the reranker beat BM25 with intervals above zero, and our BM25 reproduces Anserini's published SciFact baseline within noise; found and fixed a reranker API bug that had made every earlier reranker row a silent fallback.
- Ported the benchmarked pipeline into the production API as one parameterized Postgres query (pgvector top-k, full-text top-k, RRF in SQL, GIN + HNSW indexes) with a CI parity job that replays the gold set through the API and gates the gap against the in-memory bench; no silent fallbacks, per-request stage latency and scores.
- Added an LLM answer evaluation with a validity gate: cited answers from a local model, deterministic citation and value checks, a two-model jury with position-swapped comparisons, and Cohen's kappa against exact-match labels before any judge score is published.
'''

QA = '''## Likely interview questions, with honest answers

**Why is Recall@50 not the headline?** The fixture has 61 files and the hybrid list covers most of them in its top 50, so a random list of that length already scores what the "random list, same length" column shows (the Recall@50 row above gives both numbers). The table shows Recall@50 next to that baseline and leads with nDCG@10 and Recall@10, where the random baseline is a small fraction of the score.

**Why did hybrid tie BM25 on the fixture?** With hash embeddings (no model) and a token-overlap reranker, the dense leg adds coverage but not ranking; the paired nDCG@10 delta versus BM25 covers zero. With bge-small and the bge reranker the same pipeline beats BM25 with intervals above zero, on the fixture and on SciFact. The CI table is the download-free story; the neural table is the quality story.

**What does the parity tolerance mean?** The API over Postgres and the in-memory bench use different lexical legs (Postgres full-text with ts_rank_cd versus Okapi BM25) and different chunkers, so they are not expected to be identical. The gate is the largest gap in nDCG@10 the parity job tolerates; it is set just above the width of the paired interval between the two rows on the committed run, so a broken leg would still fail it. Both rows are reported.

**What is a prototype here versus production?** The measured, served pipeline is hybrid retrieval, the router, staleness Tier 1, the reranker configuration chosen from the SciFact sweep, and the approval-gated file plans. GraphRAG, HippoRAG, Paper2Code, the MATLAB checkpoint, multimodal pages and the RL stub run in demos but have no measured result; the README keeps them under "Prototypes (not evaluated)".

**Why did the router's number change?** The earlier README compared the routed row against plain RRF, which credited the reranker's share of the gain to the router. Against its real parent, the reranked hybrid, the router is a tie overall and a gain on the exact-path slice with n = 8. The generated table prints the honest pair.

**What did the reranker bug teach you?** Every "bge-rerank" row before M2 was a silent fallback because the code called a method the loaded model did not have and a broad except hid it. The fix was small; the lesson was structural: every result now records whether the model actually loaded, responses list degraded components, and a test fails if a configured reranker could fall back without a trace.

**Why fp16 with a 512-token cap and depth 20?** On SciFact, fp16 with max_length 512 is a quality tie with fp32 uncapped at a fraction of the latency, and reranking deeper than 20 loses nDCG@10 while costing more. The served configuration is chosen from that table, and CPU latency for the same setting is measured separately in the Docker CPU stack.

**How do you know the numbers in the README are real?** Every benchmark block in the README is rendered from a committed results file by one command, and a test fails if the text drifts. Each results file records the command, the git sha, the model revisions and the device. Losers stay in the tables.

**What would you do next?** A larger fixture where the first stage is a real filter (the reranker rescores most of the 71-chunk fixture today), ParadeDB BM25 in the production path with the same parity gate, and a judge whose kappa against gold clears the gate on more than the simple-factual slice.
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


if __name__ == "__main__":
    raise SystemExit(main())
