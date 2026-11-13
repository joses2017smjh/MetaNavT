"""Task-local real Postgres/pgvector + original API, then bounded HTTP load.

Run inside an allocated Slurm CPU/GPU job. Does not contact an LLM, publish
the service or alter the existing database. All servers terminate on exit.

The services come from app/eval/service.py: Postgres runs in the Apptainer image
with only its own scratch directory bound (--scratch, default a temporary
directory under TMPDIR), and the connection strings name no user. The schema
migrations run before the API starts (the deploy step; the API applies none). Everything this
script writes under --out-dir (http_load.json, setup_failure.json) passes through
app/eval/provenance.scrub, so it carries no absolute path, user or host name; the
service logs stay in the scratch directory, which is not published. http_load.json
is written by app/eval/http_load.write_report and starts with the code's provenance:
the commit, whether the working tree had uncommitted changes when this script
started (read before the services start or anything is written), and the sha256 of
the files the deployment block lists (the API's code and this client's).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eval.provenance import git_state, sanitize_path, scrub  # noqa: E402
from app.eval.service import ApiServer, TaskPostgres, base_environment, migrate  # noqa: E402


def main(argv: list[str] | None = None, *, root: Path | None = None) -> int:
    """`root`: the checkout whose commit and working tree the results file records (default: this repository)."""
    git_root = root or ROOT
    tree_at_start = git_state(git_root)  # the tree as it was when the run began, before anything below writes
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--rerank-depth", type=int, default=20)
    parser.add_argument("--scratch", type=Path, default=None, help="service data and logs (default: a temporary directory under TMPDIR, removed at exit)")
    args = parser.parse_args(argv)
    sources = ["main.py", "app/api/routers/retrieve.py", "app/engine/retriever.py", "app/settings.py", "app/eval/http_load.py", "app/eval/service.py", "app/database/migrate.py", "scripts/revamp_http_service.py"]
    source_sha256 = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    corpus_identity = hashlib.sha256(json.dumps([(str(p.relative_to(args.data)), hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(args.data.rglob("*")) if p.is_file()], separators=(",", ":")).encode()).hexdigest()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    settings = {"ENVIRONMENT": "prod", "MODEL_PROVIDER": "ollama", "EMBEDDING_PROVIDER": "huggingface", "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5", "EMBEDDING_DIM": "384", "EMBEDDING_DEVICE": args.device, "RETRIEVAL_REQUIRE_DEVICE": args.device, "RERANKER_DEVICE": args.device, "RERANKER_MODEL": "BAAI/bge-reranker-v2-m3", "RERANKER_REQUIRED": "true", "RERANKER_PRECISION": args.precision, "RERANKER_MAX_LENGTH": str(args.max_length), "RERANK_DEPTH": str(args.rerank_depth), "RERANK_TOP_N": "8", "DATA_DIR": str(args.data.resolve()), "INDEX_ON_START": "true", "CHUNK_SIZE": "512", "CHUNK_OVERLAP": "50", "RETRIEVE_K": "50", "RETRIEVAL_MODE": "sql", "ENABLE_ROUTER": "true", "BGE_ALLOW_DOWNLOAD": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "LOG_FORMAT": "json"}
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="metanavit-service-") as temporary:
        scratch = (args.scratch or Path(temporary)).resolve()
        postgres = TaskPostgres(args.container, scratch / "postgres")
        api = None
        try:
            postgres.start()
            environment = base_environment({**settings, **postgres.env()})
            migrate(environment, cwd=ROOT, log_path=scratch / "migrate.log")  # the deploy step: the API refuses pending migrations
            api = ApiServer(environment, cwd=ROOT, log_path=scratch / "api.log").start()
            health, _ = api.wait_ready(timeout=300)
            startup_seconds = time.monotonic() - started
            from app.eval.http_load import write_report
            queries = []
            for line in args.queries.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    queries.append((str(row.get("id", len(queries))), row.get("question") or row.get("query") or row["text"]))
            deployment = scrub({"backend": "original FastAPI main:app + task-local PostgreSQL16/pgvector SQL retrieval", "data_dir": sanitize_path(str(args.data.resolve())), "corpus_kind": "existing deterministic fixture" if args.data.resolve() == (ROOT / "bench/corpus/files").resolve() else "caller supplied corpus", "data_files": sum(p.is_file() for p in args.data.rglob("*")), "corpus_sha256": corpus_identity, "source_sha256": source_sha256, "startup_seconds": startup_seconds, "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "precision": args.precision, "max_length": args.max_length, "rerank_depth": args.rerank_depth, "observed_health": health, "container": sanitize_path(str(args.container.resolve()))})
            report = write_report(api.url, queries, args.out_dir / "http_load.json", root=git_root, started=tree_at_start, sources=sorted(sources),
                                  deployment=deployment, requests_per_level=args.requests, warmup_requests=2, timeout=120,
                                  require_reranker=True, require_device=args.device, fresh_service=True)
            print(json.dumps({"outcome": report["outcome"], "all_attempts": report["all_attempts"], "out": sanitize_path(str(args.out_dir / "http_load.json"))}))
            return 0 if report["outcome"] == "ok" else 2
        except Exception as exc:
            failure = {"error": f"{type(exc).__name__}: {exc}", "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "device": args.device, "elapsed_seconds": time.monotonic() - started}
            (args.out_dir / "setup_failure.json").write_text(json.dumps(scrub(failure), indent=2) + "\n")
            raise
        finally:
            if api is not None:
                api.stop()
            postgres.stop()


if __name__ == "__main__":
    raise SystemExit(main())
