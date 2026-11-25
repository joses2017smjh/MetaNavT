"""What a small reranker would cost the public demo's API on a CPU: latency per rerank at several shapes, and memory.

    python scripts/measure_rerank_cpu.py [--models minilm-l6,student-minilm-l6] [--threads 2] [--checkpoint-dir <dir>]
                                         [--out bench/results/small_rerankers_cpu.json]

INDICATIVE ONLY. The serving bar of M43 is measured on the host itself (scripts/time_reranker_on_host.py): the same
reranking job took 98 s and 155 s on two HPC CPUs, so a timing from this machine says which shapes are worth trying
there, not whether a model fits.

The method is scripts/measure_api_memory.py's, with the reranker swapped, and the measuring program is
scripts/rerank_timing_payload.py, the one the host's own measurement runs. For each model (a tag of
app/eval/small_rerankers.py) and each max_length of MAX_LENGTHS it runs in a fresh process, so the peak is that
model's own:
  1. import main with PUBLIC_DEMO=1, build the embedder as start-up does and embed 8 queries and 32 passages
     (steps 1 and 2 of measure_api_memory.py);
  2. load the reranker as the retriever does (load_reranker, fp32, that max_length);
  3. for each depth of DEPTHS, ascending: one warm-up call, then repeated calls, each a different query with `depth`
     different passages, every passage long enough to be cut at max_length (so a pair costs what the shape can cost at
     most), scored in one batch as the retriever scores them; p50 and p95 of the call times.
After each step the process's resident set and its peak so far are read from /proc/self/status (VmRSS, VmHWM), in MiB.
The peak never falls, so with depths ascending the peak after the last depth is the peak of the largest. The number of
timed calls is `repeats`, fewer for a slow model (never under `min_repeats`): what fits `budget_seconds` per depth.

CUDA is hidden. With a CUDA build of torch the memory is an upper bound for the image's CPU wheel. Linux only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/measure_rerank_cpu.py"
PAYLOAD = "scripts/rerank_timing_payload.py"  # the measuring program: the one the host's own measurement runs
MAX_LENGTHS = (256, 512)
DEPTHS = (5, 10, 20)
INDICATIVE = ("indicative only: one HPC CPU, not the demo's host; the serving bar is measured on the host itself "
              "(scripts/time_reranker_on_host.py)")


def run_payload(env: dict) -> subprocess.CompletedProcess:
    """The measuring program in a fresh process, with its settings (TIMING_*) in the environment."""
    return subprocess.run([sys.executable, str(ROOT / PAYLOAD)], capture_output=True, text=True, env=env, cwd=ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "bench" / "results" / "small_rerankers_cpu.json")
    parser.add_argument("--models", default="minilm-l6,minilm-l12", help="comma-separated tags of app/eval/small_rerankers.py")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for torch (OMP_NUM_THREADS): the host has 2 vCPUs")
    parser.add_argument("--repeats", type=int, default=30, help="timed calls per depth at most")
    parser.add_argument("--min-repeats", type=int, default=5)
    parser.add_argument("--budget-seconds", type=float, default=240.0, help="per depth: a slow model gets fewer timed calls, never under --min-repeats")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="where the students' checkpoints are (default: $SMALL_RERANKER_CHECKPOINTS)")
    args = parser.parse_args(argv)
    if str(ROOT) not in sys.path:  # run as `python scripts/measure_rerank_cpu.py`
        sys.path.insert(0, str(ROOT))
    from app.eval import small_rerankers as sr
    from app.eval.provenance import command_line, device_info, git_sha, git_state, hf_revision, run_provenance, scrub

    started = git_state(ROOT)  # before anything runs, as every results writer does
    rows = []
    for tag in [t.strip() for t in args.models.split(",") if t.strip()]:
        source, record = sr.reranker_source(tag, args.checkpoint_dir)
        if tag not in sr.STUDENTS:
            record = {**record, "revision": hf_revision(source)}
            sr.check_revision(tag, record["revision"])
        for max_length in MAX_LENGTHS:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "BGE_ALLOW_DOWNLOAD": "1", "PYTHONPATH": str(ROOT), "TIMING_MODEL": source,
                   "TIMING_MAX_LENGTH": str(max_length), "TIMING_DEPTHS": ",".join(str(d) for d in DEPTHS), "TIMING_THREADS": str(args.threads),
                   "TIMING_REPEATS": str(args.repeats), "TIMING_MIN_REPEATS": str(args.min_repeats), "TIMING_BUDGET_SECONDS": str(args.budget_seconds)}
            done = run_payload(env)
            lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
            if done.returncode != 0 or not lines:
                raise SystemExit(f"{tag} at max_length {max_length} failed:\n{scrub(done.stderr[-2000:])}")
            result = json.loads(lines[-1][len("RESULT "):])
            result["reranker"].pop("revision", None)  # the row's own record names the model (a student by its sha256, never its path)
            rows.append({"tag": tag, "config": sr.row_name(tag), **record, **result})
            print(tag, max_length, [(s["depth"], s["p50_ms"], s["p95_ms"], s["n"]) for s in result["shapes"]],
                  "peak MiB", result["shapes"][-1]["memory_mib"]["VmHWM"], flush=True)
    record = {"git_sha": git_sha(ROOT),
              **run_provenance(root=ROOT, started=started, sources=[SCRIPT, PAYLOAD, "app/eval/small_rerankers.py", "app/engine/retriever.py",
                                                                   "app/retrieval/rerank.py", "app/settings.py", "main.py"]),
              "command": command_line(), "device": {**device_info("cpu"), "threads": args.threads}, "label": INDICATIVE,
              "method": {"as": "scripts/measure_api_memory.py with the reranker swapped: import main, the embedder, then the reranker, in a fresh process per model and max_length",
                         "max_lengths": list(MAX_LENGTHS), "depths": list(DEPTHS), "precision": "fp32",
                         "program": PAYLOAD, "passages": "600 seeded words each, so every pair is cut at max_length: the most a shape can cost",
                         "calls": "one warm-up, then up to `repeats` calls, each another query with other passages, one batch per call",
                         "repeats": args.repeats, "min_repeats": args.min_repeats, "budget_seconds_per_depth": args.budget_seconds,
                         "memory": "VmRSS, VmHWM, RssAnon, RssFile from /proc/self/status, MiB; the peak never falls, and depths ascend"},
              "results": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scrub(record), indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
