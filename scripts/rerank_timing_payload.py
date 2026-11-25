"""Time one reranker inside the public demo's API process: p50 and p95 per rerank call at several depths, and memory.

Not run by hand. Two callers hand it its settings in the environment and read the one line it prints:
  scripts/measure_rerank_cpu.py       a child process per model and max_length, on whatever CPU runs it (indicative)
  scripts/time_reranker_on_host.py    piped into `python -` in a one-off container of the deployed image on the
                                      demo's host (the serving bar's measurement)
so both numbers come from the same program. It uses only the standard library and the app's own loaders, because the
image on the host is the deployed one, whatever commit that is.

In order, in one process, as the API meets them (the method of scripts/measure_api_memory.py):
  1. import main with PUBLIC_DEMO=1: the public app, as the container imports it;
  2. build the embedder as start-up does, embed 8 queries and a batch of 32 passages;
  3. load the reranker as the retriever does (load_reranker: TIMING_MODEL, fp32, TIMING_MAX_LENGTH);
  4. for each depth of TIMING_DEPTHS, ascending: one warm-up call, then repeated calls, each another query with `depth`
     other passages, every passage longer than the token cap (so a pair costs the most the shape can cost), scored in
     one batch as the retriever scores them.
After each step: the process's resident set and its peak so far (/proc/self/status, MiB), and the cgroup's peak where
the kernel reports one (inside a container that is the container's own). The peak never falls; depths ascend.
"""

import json
import os
import random
import sys
import time

# The single host's retrieval settings (deploy/single-host/compose.yml), as scripts/measure_api_memory.py sets them; the
# reranker's own are set from TIMING_* below. CUDA is hidden.
SETTINGS = {
    "PUBLIC_DEMO": "1", "ENVIRONMENT": "prod", "MODEL_PROVIDER": "ollama", "EMBEDDING_PROVIDER": "huggingface",
    "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5", "EMBEDDING_DIM": "384", "RERANKER_PRECISION": "fp32", "RERANK_DEPTH": "20",
    "RERANKER_REQUIRED": "true", "CUDA_VISIBLE_DEVICES": "",
}
QUERY = "What learning rate did run 47 use?"
PASSAGE = "learning rate warmup schedule "
WORDS = ("learning", "rate", "warmup", "schedule", "encoder", "fusion", "checkpoint", "validation", "loss", "batch", "epoch", "seed",
         "config", "slurm", "trellis", "stereo", "bark", "render", "baseline", "ablation", "metric", "dataset", "split", "optimizer")
PASSAGE_WORDS = 600  # well past 512 tokens: every pair is cut at the token cap


def memory():
    """VmRSS, VmHWM, RssAnon and RssFile of this process, MiB (the kernel's kB are KiB)."""
    with open("/proc/self/status") as fh:
        fields = dict(line.split(":", 1) for line in fh if ":" in line)
    return {name: round(int(fields[name].split()[0]) / 1024) for name in ("VmRSS", "VmHWM", "RssAnon", "RssFile")}


def cgroup_peak_mib():
    """The memory peak of this process's cgroup, MiB: cgroup v2's memory.peak, else v1's max_usage_in_bytes; None when
    neither can be read (an old kernel, or no cgroup of its own)."""
    for path in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            with open(path) as fh:
                return round(int(fh.read().strip()) / 2 ** 20)
        except (OSError, ValueError):
            continue
    return None


def texts(seed, depth):
    """One query and `depth` passages, all different, from a seeded shuffle of WORDS: the same texts on every machine."""
    rng = random.Random(seed)
    query = "what " + " ".join(rng.sample(WORDS, 7)) + " did run %d use" % (seed % 97)
    return [(query, " ".join(rng.choice(WORDS) for _ in range(PASSAGE_WORDS))) for _ in range(depth)]


def percentile(samples, q):
    """The q-th percentile with linear interpolation between order statistics (numpy.percentile's default)."""
    ordered = sorted(samples)
    at = (len(ordered) - 1) * q / 100.0
    low = int(at)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (at - low)


def summary_ms(samples):
    return {"p50_ms": round(percentile(samples, 50), 1), "p95_ms": round(percentile(samples, 95), 1), "min_ms": round(min(samples), 1),
            "max_ms": round(max(samples), 1), "n": len(samples)}


def measure(env):
    model = env["TIMING_MODEL"]
    max_length = int(env.get("TIMING_MAX_LENGTH", "512"))
    depths = sorted(int(d) for d in env.get("TIMING_DEPTHS", "20").split(","))
    threads = int(env.get("TIMING_THREADS", "2"))
    repeats, min_repeats = int(env.get("TIMING_REPEATS", "30")), int(env.get("TIMING_MIN_REPEATS", "5"))
    budget = float(env.get("TIMING_BUDGET_SECONDS", "240"))
    os.environ.update(SETTINGS)
    os.environ.update({"RERANKER_MODEL": model, "RERANKER_MAX_LENGTH": str(max_length), "OMP_NUM_THREADS": str(threads)})
    import torch

    torch.set_num_threads(threads)
    steps = {}
    import main  # noqa: F401  the public app, as the container imports it

    steps["after_import_main"] = memory()
    from llama_index.core import Settings

    from app.settings import init_retrieval_settings

    init_retrieval_settings()
    for _ in range(8):
        Settings.embed_model.get_query_embedding(QUERY)
    Settings.embed_model.get_text_embedding_batch([PASSAGE * 60] * 32)
    steps["after_embedder"] = memory()
    from app.engine.retriever import load_reranker

    started = time.perf_counter()
    reranker, info = load_reranker()
    load_seconds = round(time.perf_counter() - started, 2)
    steps["after_reranker_load"] = memory()
    shapes = []
    for depth in depths:
        pairs = texts(0, depth)
        started = time.perf_counter()
        reranker.predict(pairs, batch_size=len(pairs))  # warm-up, and the estimate of what one call costs
        first = time.perf_counter() - started
        n = max(min_repeats, min(repeats, int(budget / max(first, 1e-6))))
        samples = []
        for k in range(1, n + 1):
            pairs = texts(k, depth)
            started = time.perf_counter()
            reranker.predict(pairs, batch_size=len(pairs))
            samples.append((time.perf_counter() - started) * 1000.0)
        shapes.append(dict(summary_ms(samples), depth=depth, warm_up_ms=round(first * 1000.0, 1), memory_mib=memory(), cgroup_peak_mib=cgroup_peak_mib()))
    return {"max_length": max_length, "threads": threads, "torch_threads": torch.get_num_threads(), "torch": torch.__version__,
            "cpu_count": os.cpu_count(), "memory_mib": steps, "reranker_load_seconds": load_seconds,
            "reranker": {key: info.get(key) for key in ("revision", "precision", "max_length", "device", "dtype")}, "shapes": shapes}


if __name__ == "__main__":
    print("RESULT " + json.dumps(measure(os.environ)))
    sys.stdout.flush()
