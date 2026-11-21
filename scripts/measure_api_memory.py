"""Peak memory of the public demo's API on CPU, without and with the reranker, as the API loads its models.

    python scripts/measure_api_memory.py [--out bench/results/api_memory_cpu.json] [--threads 2]

The single host's sizing rests on this record (doc/deploy.md, "The live demo: one host on the AWS Free plan"). In one
process, in the order the container meets them:
  1. import main with PUBLIC_DEMO=1: the public app, as the container imports it (no lifespan, so no database);
  2. build the embedder as start-up does (init_retrieval_settings: BAAI/bge-small-en-v1.5), then embed 8 queries and
     a batch of 32 passages;
  3. load the reranker as the retriever does (load_reranker: BAAI/bge-reranker-v2-m3, fp32, max_length 512), then
     score one query's 20 pairs (RERANK_DEPTH=20) whose passages are long enough to be cut at 512 tokens, timed.
After each step it records the process's resident set and its peak so far (VmRSS, VmHWM, and the anonymous and
file-backed parts of the resident set) from /proc/self/status, in MiB (the kernel's kB are KiB). CUDA is hidden: with
a CUDA build of torch the numbers are an upper bound for the image's CPU wheel, whose libraries are smaller. Linux
only. Needs the ml extra and both models in the Hugging Face cache (or BGE_ALLOW_DOWNLOAD=1 to download them).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/measure_api_memory.py"
# The single host's retrieval settings (deploy/single-host/compose.yml), with the reranker configured; CUDA hidden.
SETTINGS = {
    "PUBLIC_DEMO": "1", "ENVIRONMENT": "prod", "MODEL_PROVIDER": "ollama", "EMBEDDING_PROVIDER": "huggingface",
    "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5", "EMBEDDING_DIM": "384", "RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
    "RERANKER_PRECISION": "fp32", "RERANKER_MAX_LENGTH": "512", "RERANK_DEPTH": "20", "RERANKER_REQUIRED": "true",
    "CUDA_VISIBLE_DEVICES": "",
}
QUERY = "What learning rate did run 47 use?"
PASSAGE = "learning rate warmup schedule "


def memory() -> dict[str, int]:
    """VmRSS, VmHWM, RssAnon and RssFile of this process, MiB."""
    with open("/proc/self/status") as fh:
        fields = dict(line.split(":", 1) for line in fh if ":" in line)
    return {name: round(int(fields[name].split()[0]) / 1024) for name in ("VmRSS", "VmHWM", "RssAnon", "RssFile")}


def measure(threads: int) -> dict:
    os.environ.update(SETTINGS)
    os.environ["OMP_NUM_THREADS"] = str(threads)
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

    reranker, info = load_reranker()
    pairs = [(QUERY, PASSAGE * 120)] * int(SETTINGS["RERANK_DEPTH"])
    started = time.perf_counter()
    reranker.predict(pairs, batch_size=len(pairs))
    seconds = round(time.perf_counter() - started, 2)
    steps["after_reranker"] = memory()
    return {"threads": threads, "memory_mib": steps, "rerank_pairs": len(pairs), "rerank_seconds": seconds,
            "reranker": {key: info.get(key) for key in ("model", "revision", "precision", "max_length", "device")}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "bench" / "results" / "api_memory_cpu.json")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for torch (OMP_NUM_THREADS)")
    args = parser.parse_args(argv)
    if str(ROOT) not in sys.path:  # run as `python scripts/measure_api_memory.py`
        sys.path.insert(0, str(ROOT))
    from app.eval.provenance import command_line, device_info, git_sha, git_state, loaded_sources, run_provenance, scrub

    started = git_state(ROOT)  # before anything runs, as every results writer does
    result = measure(args.threads)
    record = {"git_sha": git_sha(ROOT),
              **run_provenance(root=ROOT, started=started, sources=sorted({SCRIPT, *loaded_sources()})),
              "command": command_line(), "device": device_info("cpu"), "settings": SETTINGS, **result}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scrub(record), indent=2) + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
