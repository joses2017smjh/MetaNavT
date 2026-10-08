"""Task-local real Postgres/pgvector + original API, then bounded HTTP load.

Run inside an allocated Slurm CPU/GPU job. Does not contact an LLM, publish
the service or alter the existing database. All servers terminate on exit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_health(url: str, process: subprocess.Popen, timeout: int = 300) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"API exited {process.returncode}; see service log")
        try:
            with urllib.request.urlopen(url + "/health", timeout=10) as response:
                health = json.load(response)
                if health.get("status") == "ok":
                    return health
                last = health
        except urllib.error.HTTPError as exc:
            try:
                last = json.load(exc)
            except (ValueError, OSError):
                last = str(exc)
            if isinstance(last, dict) and last.get("status") == "unavailable" and last.get("error"):
                raise RuntimeError(f"API startup failed: {last['error']}") from exc
        except (urllib.error.URLError, ValueError, TimeoutError) as exc:
            last = str(exc)
        time.sleep(1)
    raise RuntimeError(f"API not ready after {timeout}s: {last}")


def main() -> int:
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
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = ["main.py", "app/api/routers/retrieve.py", "app/engine/retriever.py", "app/settings.py", "app/eval/http_load.py", "scripts/revamp_http_service.py"]
    source_sha256 = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
    corpus_identity = hashlib.sha256(json.dumps([(str(p.relative_to(args.data)), hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(args.data.rglob("*")) if p.is_file()], separators=(",", ":")).encode()).hexdigest()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pg_port, api_port = free_port(), free_port()
    env = dict(os.environ)
    env.update({"PG_CONNECTION_STRING": f"postgresql://{os.environ['USER']}@127.0.0.1:{pg_port}/metanavit", "PSYCOPG2_CONNECTION_STRING": f"dbname=metanavit user={os.environ['USER']} host=127.0.0.1 port={pg_port}", "ENVIRONMENT": "prod", "MODEL_PROVIDER": "ollama", "EMBEDDING_PROVIDER": "huggingface", "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5", "EMBEDDING_DIM": "384", "EMBEDDING_DEVICE": args.device, "RETRIEVAL_REQUIRE_DEVICE": args.device, "RERANKER_DEVICE": args.device, "RERANKER_MODEL": "BAAI/bge-reranker-v2-m3", "RERANKER_REQUIRED": "true", "RERANKER_PRECISION": args.precision, "RERANKER_MAX_LENGTH": str(args.max_length), "RERANK_DEPTH": str(args.rerank_depth), "RERANK_TOP_N": "8", "DATA_DIR": str(args.data.resolve()), "INDEX_ON_START": "true", "CHUNK_SIZE": "512", "CHUNK_OVERLAP": "50", "RETRIEVE_K": "50", "RETRIEVAL_MODE": "sql", "ENABLE_ROUTER": "true", "BGE_ALLOW_DOWNLOAD": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "LOG_FORMAT": "json"})
    env.pop("FRONTEND_ENDPOINT", None)
    invocation = ["apptainer", "exec", "--cleanenv", "--bind", "/nfs/stak/users/sanchej7/hpc-share", str(args.container.resolve())]
    postgres = api = None
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="metanavit-service-") as scratch, (args.out_dir / "postgres.log").open("w") as pg_log, (args.out_dir / "api.log").open("w") as api_log:
        data = Path(scratch) / "pgdata"
        run = Path(scratch) / "socket"
        run.mkdir()
        try:
            subprocess.run([*invocation, "/usr/lib/postgresql/16/bin/initdb", "-D", str(data), "--auth=trust", "--no-locale", "--encoding=UTF8"], stdout=pg_log, stderr=subprocess.STDOUT, check=True)
            postgres = subprocess.Popen([*invocation, "/usr/lib/postgresql/16/bin/postgres", "-D", str(data), "-h", "127.0.0.1", "-p", str(pg_port), "-k", str(run)], stdout=pg_log, stderr=subprocess.STDOUT)
            for _ in range(30):
                ready = subprocess.run([*invocation, "/usr/lib/postgresql/16/bin/pg_isready", "-h", "127.0.0.1", "-p", str(pg_port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if ready.returncode == 0:
                    break
                if postgres.poll() is not None:
                    raise RuntimeError("task-local PostgreSQL exited; see postgres.log")
                time.sleep(1)
            else:
                raise RuntimeError("task-local PostgreSQL not ready")
            subprocess.run([*invocation, "/usr/lib/postgresql/16/bin/createdb", "-h", "127.0.0.1", "-p", str(pg_port), "metanavit"], check=True)
            api = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(api_port)], cwd=root, env=env, stdout=api_log, stderr=subprocess.STDOUT)
            url = f"http://127.0.0.1:{api_port}"
            health = wait_health(url, api)
            startup_seconds = time.monotonic() - started
            from app.eval.http_load import run_load
            import asyncio
            queries = []
            for line in args.queries.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    queries.append((str(row.get("id", len(queries))), row.get("question") or row.get("query") or row["text"]))
            deployment = {"backend": "original FastAPI main:app + task-local PostgreSQL16/pgvector SQL retrieval", "data_dir": str(args.data.resolve()), "corpus_kind": "existing deterministic fixture" if args.data.resolve() == (root / "bench/corpus/files").resolve() else "caller supplied corpus", "data_files": sum(p.is_file() for p in args.data.rglob("*")), "corpus_sha256": corpus_identity, "source_sha256": source_sha256, "startup_seconds": startup_seconds, "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(), "precision": args.precision, "max_length": args.max_length, "rerank_depth": args.rerank_depth, "observed_health": health, "container": str(args.container.resolve())}
            def checkpoint(report):
                report["deployment"] = deployment
                target = args.out_dir / "http_load.json"
                temporary = target.with_suffix(".json.partial")
                temporary.write_text(json.dumps(report, indent=2) + "\n")
                temporary.replace(target)
            report = asyncio.run(run_load(url, queries, requests_per_level=args.requests, warmup_requests=2, timeout=120, require_reranker=True, require_device=args.device, fresh_service=True, checkpoint=checkpoint))
            (args.out_dir / "http_load.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"outcome": report["outcome"], "all_attempts": report["all_attempts"], "out": str(args.out_dir / "http_load.json")}))
            return 0 if report["outcome"] == "ok" else 2
        except Exception as exc:
            (args.out_dir / "setup_failure.json").write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}", "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "device": args.device, "elapsed_seconds": time.monotonic() - started}, indent=2) + "\n")
            raise
        finally:
            for process in [api, postgres]:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
