"""Time one ColBERT model inside the public demo's API process: p50 and p95 per call for each shape, and memory (M49).

Not run by hand: scripts/measure_late_interaction_cpu.py starts it in a fresh process per model, hands it its settings
in the environment (TIMING_*) and reads the one line it prints. The sibling of scripts/rerank_timing_payload.py, with
the same method: that one loads a CrossEncoder through the retriever's load_reranker, which cannot load a multi-vector
model, so this one loads the model through the repository's adapter (app.eval.late_interaction.load_encoder).

In order, in one process:
  1. import main with PUBLIC_DEMO=1: the public app, as the container imports it;
  2. build the embedder as start-up does (bge-small), embed 8 queries and a batch of 32 passages;
  3. load the ColBERT model at its saved document length (fp32, CPU);
  4. the shapes, each with one warm-up call and then repeated calls, each another seeded query:
       query_encode        encode one query (the [MASK]-expanded query matrix)
       rerank_on_the_fly   encode one query and `depth` passages, every passage longer than the document cap, and MaxSim
                           each (depths 5, 10, 20): a served reranker without a token index
       rerank_stored       encode one query, MaxSim over `depth` stored passage matrices (encoded once, untimed): the
                           precomputed-index shape
       first_stage         encode one query, exhaustive maxsim_scores over the fixture's chunk token store (71 chunks,
                           encoded once, untimed), rank: role (a) on the fixture
       parent_first_stage  bge-small: embed one query, cosine over the same chunks' vectors, rank: the parent's cost
After each step: the process's resident set and its peak so far (/proc/self/status, MiB) and the cgroup's peak where
the kernel reports one. The peak never falls. The queries are seeded word lists, never a benchmark question.
"""

import json
import os
import random
import sys
import time

SETTINGS = {
    "PUBLIC_DEMO": "1", "ENVIRONMENT": "prod", "MODEL_PROVIDER": "ollama", "EMBEDDING_PROVIDER": "huggingface",
    "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5", "EMBEDDING_DIM": "384", "CUDA_VISIBLE_DEVICES": "",
}
WORDS = ("learning", "rate", "warmup", "schedule", "encoder", "fusion", "checkpoint", "validation", "loss", "batch", "epoch", "seed",
         "config", "slurm", "trellis", "stereo", "bark", "render", "baseline", "ablation", "metric", "dataset", "split", "optimizer")
PASSAGE_WORDS = 600  # past every model's document cap (180, 300): each passage is cut, the most a shape can cost
DEPTHS = (5, 10, 20)


def memory():
    with open("/proc/self/status") as fh:
        fields = dict(line.split(":", 1) for line in fh if ":" in line)
    return {name: round(int(fields[name].split()[0]) / 1024) for name in ("VmRSS", "VmHWM", "RssAnon", "RssFile")}


def cgroup_peak_mib():
    for path in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            with open(path) as fh:
                return round(int(fh.read().strip()) / 2 ** 20)
        except (OSError, ValueError):
            continue
    return None


def query(seed):
    rng = random.Random(seed)
    return "what " + " ".join(rng.sample(WORDS, 7)) + " did run %d use" % (seed % 97)


def passages(seed, depth):
    rng = random.Random(10_000 + seed)
    return [" ".join(rng.choice(WORDS) for _ in range(PASSAGE_WORDS)) for _ in range(depth)]


def percentile(samples, q):
    ordered = sorted(samples)
    at = (len(ordered) - 1) * q / 100.0
    low = int(at)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (at - low)


def timed(fn, repeats, min_repeats, budget):
    """One warm-up call (its time sets how many calls fit the budget), then n calls with seeds 1..n: p50, p95, n."""
    started = time.perf_counter()
    fn(0)
    first = time.perf_counter() - started
    n = max(min_repeats, min(repeats, int(budget / max(first, 1e-6))))
    samples = []
    for k in range(1, n + 1):
        started = time.perf_counter()
        fn(k)
        samples.append((time.perf_counter() - started) * 1000.0)
    return {"p50_ms": round(percentile(samples, 50), 1), "p95_ms": round(percentile(samples, 95), 1), "min_ms": round(min(samples), 1),
            "max_ms": round(max(samples), 1), "n": len(samples), "warm_up_ms": round(first * 1000.0, 1), "memory_mib": memory(),
            "cgroup_peak_mib": cgroup_peak_mib()}


def rerank_calls(enc, maxsim, stored):
    """The two rerank shapes as functions of the depth, each giving a call of the seed k: on the fly (encode the query
    once, encode `depth` passages, MaxSim each) and from stored matrices (encode the query once, MaxSim over the
    `depth` matrices encoded beforehand). One query encode per call: the first J4 run (22150853) encoded it once per
    passage, which overstated both shapes; that file was never committed."""
    def on_the_fly(depth):
        def call(k):
            q = enc.encode_queries([query(k)])[0]
            return [maxsim(q, m) for m in enc.encode_documents(passages(k, depth))]
        return call

    def from_store(depth):
        def call(k):
            q = enc.encode_queries([query(k)])[0]
            return [maxsim(q, m) for m in stored[depth]]
        return call

    return on_the_fly, from_store

def measure(env):
    tag = env["TIMING_TAG"]
    threads = int(env.get("TIMING_THREADS", "2"))
    repeats, min_repeats = int(env.get("TIMING_REPEATS", "30")), int(env.get("TIMING_MIN_REPEATS", "5"))
    budget = float(env.get("TIMING_BUDGET_SECONDS", "120"))
    os.environ.update(SETTINGS)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    import numpy as np
    import torch

    torch.set_num_threads(threads)
    steps = {}
    import main  # noqa: F401  the public app, as the container imports it

    steps["after_import_main"] = memory()
    from llama_index.core import Settings

    from app.settings import init_retrieval_settings

    init_retrieval_settings()
    for _ in range(8):
        Settings.embed_model.get_query_embedding(query(0))
    Settings.embed_model.get_text_embedding_batch(passages(0, 32))
    steps["after_embedder"] = memory()

    from pathlib import Path

    from app.eval import late_interaction as li
    from app.eval.index_loader import load_chunks
    from app.multimodal.late_interaction import TokenStore, maxsim, maxsim_scores
    from app.retrieval.hybrid import DENSE_SCORE_DECIMALS, dense_scores, rank_by_score, unit_rows

    length = li.MODELS[tag]["saved"]["document_length"]
    started = time.perf_counter()
    enc = li.load_encoder(tag, length, "cpu")
    load_seconds = round(time.perf_counter() - started, 2)
    steps["after_model_load"] = memory()
    root = Path(env["TIMING_ROOT"])
    chunks = load_chunks(root / "bench" / "corpus" / "files", mtimes="manifest")
    started = time.perf_counter()
    store = TokenStore.from_matrices(enc.encode_documents([c.text for c in chunks]), enc.dim)
    store_seconds = round(time.perf_counter() - started, 2)
    parent_unit = unit_rows(np.asarray(Settings.embed_model.get_text_embedding_batch([c.text for c in chunks]), dtype=np.float32))
    stored = {d: enc.encode_documents(passages(0, d)) for d in DEPTHS}
    steps["after_stores"] = memory()

    on_the_fly, from_store = rerank_calls(enc, maxsim, stored)

    def first_stage(k):
        scores = maxsim_scores(enc.encode_queries([query(k)]), store, block_bytes=li.MAXSIM_BLOCK_BYTES)
        return rank_by_score(np.round(scores[0], DENSE_SCORE_DECIMALS), 50)

    def parent(k):
        return rank_by_score(dense_scores(np.asarray(Settings.embed_model.get_query_embedding(query(k))), parent_unit, unit=True), 50)

    shapes = [{"shape": "query_encode", **timed(lambda k: enc.encode_queries([query(k)]), repeats, min_repeats, budget)}]
    shapes += [{"shape": "rerank_on_the_fly", "depth": d, **timed(on_the_fly(d), repeats, min_repeats, budget)} for d in DEPTHS]
    shapes += [{"shape": "rerank_stored", "depth": d, **timed(from_store(d), repeats, min_repeats, budget)} for d in DEPTHS]
    shapes.append({"shape": "first_stage", "documents": len(store), **timed(first_stage, repeats, min_repeats, budget)})
    shapes.append({"shape": "parent_first_stage", "documents": len(chunks), **timed(parent, repeats, min_repeats, budget)})
    prov = enc.provenance()
    return {"tag": tag, "document_length": length, "threads": threads, "torch_threads": torch.get_num_threads(), "torch": torch.__version__,
            "cpu_count": os.cpu_count(), "memory_mib": steps, "model_load_seconds": load_seconds, "fixture_store_seconds": store_seconds,
            "fixture_store": {"documents": len(store), "scored_vectors": int(store.lengths().sum()), "bytes_fp32": store.nbytes},
            "model": {k: prov.get(k) for k in ("model", "revision", "weights_sha256", "dtype", "device", "versions")}, "shapes": shapes}


if __name__ == "__main__":
    print("RESULT " + json.dumps(measure(os.environ)))
    sys.stdout.flush()
