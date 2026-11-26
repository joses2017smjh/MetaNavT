"""What a ColBERT model would cost the public demo's API on a CPU (M49): latency per call for each shape, and memory.

    python scripts/measure_late_interaction_cpu.py [--models answerai-colbert-small,colbertv2] [--threads 2]
                                                   [--out bench/results/late_interaction_cpu.json]

INDICATIVE ONLY, never the serving bar: one HPC CPU, not the demo's host. Run on the share node of
bench/results/small_rerankers_cpu.json (cn-e02, Xeon Silver 4114), so the MaxSim rerank rows sit beside MiniLM-L12's
and gte-modernbert-base's from the same node; another node is named and not compared across.

For each model (a tag of app/eval/late_interaction.py, at its saved document length) the measuring program,
scripts/late_interaction_timing_payload.py, runs in a fresh process (so the memory peak is that model's own): import
main with PUBLIC_DEMO=1, build bge-small as start-up does, load the model, then time the shapes (query encode; rerank on
the fly at depths 5, 10 and 20; rerank from stored token matrices; the fixture first stage; the parent's first stage),
reading VmRSS and VmHWM after each step. CUDA is hidden. Linux only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/measure_late_interaction_cpu.py"
PAYLOAD = "scripts/late_interaction_timing_payload.py"
INDICATIVE = ("indicative only: one HPC CPU, not the demo's host; the serving bar (added p95 and peak memory on the host, "
              "api_memory_mib) is not decided here")


def run_payload(env: dict) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / PAYLOAD)], capture_output=True, text=True, env=env, cwd=ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "bench" / "results" / "late_interaction_cpu.json")
    parser.add_argument("--models", default="answerai-colbert-small,colbertv2", help="comma-separated tags of app/eval/late_interaction.py")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for torch: the host has 2 vCPUs")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--min-repeats", type=int, default=5)
    parser.add_argument("--budget-seconds", type=float, default=120.0, help="per shape: a slow one gets fewer timed calls, never under --min-repeats")
    args = parser.parse_args(argv)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from app.eval import late_interaction as li
    from app.eval.provenance import command_line, device_info, git_sha, git_state, hf_revision, run_provenance, scrub

    started = git_state(ROOT)
    rows = []
    for tag in [t.strip() for t in args.models.split(",") if t.strip()]:
        li.check_revision(tag, hf_revision(li.MODELS[tag]["model"]))
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "BGE_ALLOW_DOWNLOAD": "1", "PYTHONPATH": str(ROOT), "TIMING_TAG": tag, "TIMING_ROOT": str(ROOT),
               "TIMING_THREADS": str(args.threads), "TIMING_REPEATS": str(args.repeats), "TIMING_MIN_REPEATS": str(args.min_repeats),
               "TIMING_BUDGET_SECONDS": str(args.budget_seconds)}
        done = run_payload(env)
        lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
        if done.returncode != 0 or not lines:
            raise SystemExit(f"{tag} failed:\n{scrub(done.stderr[-3000:])}")
        result = json.loads(lines[-1][len("RESULT "):])
        rows.append(result)
        print(tag, [(s["shape"], s.get("depth"), s["p50_ms"], s["p95_ms"], s["n"]) for s in result["shapes"]],
              "peak MiB", max(s["memory_mib"]["VmHWM"] for s in result["shapes"]), flush=True)
    record = {"git_sha": git_sha(ROOT),
              **run_provenance(root=ROOT, started=started, sources=[SCRIPT, PAYLOAD, "app/eval/late_interaction.py", "app/multimodal/late_interaction.py",
                                                                   "app/settings.py", "main.py"]),
              "command": command_line(), "device": {**device_info("cpu"), "threads": args.threads}, "label": INDICATIVE,
              "method": {"as": "scripts/measure_rerank_cpu.py's (M43) with the ColBERT model through the repository's adapter: import main, bge-small, then the model, a fresh process per model",
                         "program": PAYLOAD, "depths": list((5, 10, 20)), "precision": "fp32", "document_length": "each model's saved setting",
                         "passages": "600 seeded words each, cut at the document length: the most a shape can cost",
                         "queries": "seeded word lists, never a benchmark question", "calls": "one warm-up, then up to `repeats` calls, each another query",
                         "repeats": args.repeats, "min_repeats": args.min_repeats, "budget_seconds_per_shape": args.budget_seconds,
                         "memory": "VmRSS, VmHWM, RssAnon, RssFile from /proc/self/status, MiB; the peak never falls"},
              "compare_with": "bench/results/small_rerankers_cpu.json (MiniLM-L12 and gte-modernbert-base, the same method), only when both ran on the same node",
              "results": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scrub(record), indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
