"""Two API processes on one Postgres: what two replicas share (C56).

    PG_TEST_DSN=postgresql://... python -m app.eval.two_replicas      (make two-replicas)

One database of its own on the server PG_TEST_DSN names (created here, dropped at the end), one copy of fixture v1 as
DATA_DIR (a volume both replicas mount), and an operator key pair outside it. The deploy step (schema migrations) and
the index step run once, then two `python -m uvicorn main:app` processes start on that database, each on its own
port, with docker-compose.yml's API environment (app/eval/served_stack.py: INDEX_ON_START off, the hash embedder, no
API key, no request budget), the operator's public key and METANAVIT_APPROVAL_LEDGER=postgres. Then, as HTTP calls:

  index_once             the index step ran once; neither replica indexed at start, and both /health count the rows
                         the step reported
  both_serve_retrieve    one query on each replica: 200 from both, the same paths in the same order
  plan_shared            a plan proposed on replica A is listed and read on replica B, with the same review
  refused_without_a_valid_capability
                         approve without a capability on either replica, and with one signed by another key: 403,
                         the file stays
  approved_on_the_other  approved on B with a capability signed over the review A returned: 200 applied, the file
                         moved, and A reports it applied
  replay_after_apply     the same capability on A: 409, the file moved once
  rejection_shared       another plan rejected on A: B reports it rejected and answers 409 to an approval
  burnt_grant_refused_elsewhere
                         a capability consumed on B by an approval whose move then failed (its destination directory
                         read-only for that attempt) is refused on A by the shared ledger (403), the file stays; a
                         fresh capability then applies on A. Not run as root, which a read-only directory does not stop.
  decision_log_shared    both replicas read the same decision log

The results file holds status codes, booleans and counts, with the commit, the command and the machine description
from app/eval/provenance.py (no host name, user name, absolute path, port or connection string); no review (a review
carries device and inode numbers) and no plan id. The committed file is written only from a tree without
uncommitted changes. The request budget is not part of the run: it is per replica by design (a cross-replica limit
belongs at the proxy).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.eval.provenance import command_line, device_info, git_sha, scrub
from app.eval.served_stack import (DEPLOY_STEP, INDEX_STEP, ROOT, api_env, copy_fixture, create_database, database_dsn,
                                   drop_database, free_port, get_json, public_env)

OUT = ROOT / "bench" / "results" / "two_replicas.json"
SCHEMA = "two-replicas-v1"
COMPOSE_MODEL = "qwen2.5:14b"  # docker-compose.yml's MODEL; no chat call is made, so no model runs
NO_OLLAMA = "http://127.0.0.1:9"  # nothing listens there: only /api/chat would call it
QUERY = "learning rate for run 47"
MOVES = {  # fixture v1 files and where each plan moves them
    "approved": ("configs/run_047.yaml", "archive/run_047.yaml"),
    "rejected": ("configs/run_046.yaml", "archive/run_046.yaml"),
    "burnt": ("configs/run_045.yaml", "readonly/run_045.yaml"),
}


class Replica:
    """`python -m uvicorn main:app` with `env` on a free loopback port; its output goes to `log`."""

    def __init__(self, name: str, env: dict, log: Path, startup_timeout: float = 180):
        self.name, self.log = name, log
        self.url = f"http://127.0.0.1:{free_port()}"
        with log.open("w") as out:
            self.process = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1",
                                             "--port", self.url.rsplit(":", 1)[1]], cwd=ROOT, env=env, stdout=out,
                                            stderr=subprocess.STDOUT)
        deadline, last = time.monotonic() + startup_timeout, None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"replica {name} exited with {self.process.returncode}: {self.tail()}")
            try:
                status, body = get_json(self.url + "/health")
            except OSError as exc:
                status, body = None, str(exc)
            if status == 200 and isinstance(body, dict) and body.get("status") == "ok":
                self.health = body
                return
            last = body
            time.sleep(0.5)
        raise RuntimeError(f"replica {name}: no healthy /health after {startup_timeout} s: {scrub(str(last))[:300]}")

    def call(self, method: str, route: str, payload: Any = None) -> tuple[int, Any]:
        if method == "GET":
            return get_json(self.url + route, timeout=60)
        return get_json(self.url + route, timeout=60, payload={} if payload is None else payload)

    def tail(self, lines: int = 30) -> str:
        try:
            return scrub("\n".join(self.log.read_text(errors="replace").splitlines()[-lines:]))
        except OSError:
            return ""

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=30)


def _observation(name: str, checks: dict[str, bool], **observed) -> dict:
    """One observation: what was seen (status codes, booleans, counts) and whether every check held."""
    return {"name": name, **observed, "checks": checks, "passed": all(checks.values())}


def _versions(dsn: str) -> dict:
    import psycopg2  # the app extra; only a run that starts the API needs it

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT current_setting('server_version'), (SELECT extversion FROM pg_extension WHERE extname = 'vector');")
            server, vector = cur.fetchone()
    finally:
        conn.close()
    return {"postgres": server.split()[0], "pgvector": vector}


def observe(a: Replica, b: Replica, data_dir: Path, operator: Path, index_report: dict) -> list[dict]:
    """The calls of the run, in order; each returns one observation."""
    from app.mcp.approvals import create_key, sign_review

    def sign(review: dict) -> str:
        return sign_review(review, operator / "operator.key", actor="replica-run-operator")

    def exists(rel: str) -> bool:
        return (data_dir / rel).is_file()

    observations = []
    health = [replica.call("GET", "/health") for replica in (a, b)]
    rows = [body.get("n_nodes") for _, body in health]
    observations.append(_observation(
        "index_once",
        {"index step exited 0": index_report["exit"] == 0,
         "both healthy": [status for status, _ in health] == [200, 200],
         "neither replica indexed at start": [body["indexing"].get("indexed") for _, body in health] == [False, False],
         "both count the rows the index step wrote": rows == [index_report["n_nodes"]] * 2},
        index_step_runs=1, health=[status for status, _ in health], indexed_at_start=[body["indexing"].get("indexed") for _, body in health],
        rows=rows))

    answers = [replica.call("POST", "/api/retrieve/", {"query": QUERY, "k": 5}) for replica in (a, b)]
    paths = [[hit["path"] for hit in body.get("hits", [])] if status == 200 else None for status, body in answers]
    observations.append(_observation(
        "both_serve_retrieve",
        {"both answer 200": [status for status, _ in answers] == [200, 200],
         "hits on both": bool(paths[0]) and bool(paths[1]),
         "the same paths in the same order": paths[0] == paths[1]},
        statuses=[status for status, _ in answers], hits=[len(p or []) for p in paths]))

    src, dst = MOVES["approved"]
    created_status, created = a.call("POST", "/api/plans/", {"src": src, "dst": dst})
    plan_id = created.get("plan_id") if isinstance(created, dict) else None
    listed_status, listed = b.call("GET", "/api/plans/")
    read_status, read = b.call("GET", f"/api/plans/{plan_id}")
    observations.append(_observation(
        "plan_shared",
        {"proposed on A (201)": created_status == 201,
         "listed on B": listed_status == 200 and plan_id in [p["plan_id"] for p in listed],
         "read on B with the same review": read_status == 200 and read.get("review") == created.get("review"),
         "pending, nothing moved": read.get("status") == "pending_approval" and exists(src) and not exists(dst)},
        statuses={"propose_on_a": created_status, "list_on_b": listed_status, "read_on_b": read_status}))

    with tempfile.TemporaryDirectory(prefix="replicas-stranger-") as temporary:  # a key pair that is not the operator's
        create_key(Path(temporary) / "other.key", Path(temporary) / "other.pub")
        forged = sign_review(created["review"], Path(temporary) / "other.key", actor="someone-else")
    refusals = {
        "no capability on A": a.call("POST", f"/api/plans/{plan_id}/approve", {"actor": "web-ui"})[0],
        "no capability on B": b.call("POST", f"/api/plans/{plan_id}/approve", {"actor": "web-ui"})[0],
        "another key's capability on B": b.call("POST", f"/api/plans/{plan_id}/approve", {"approval_token": forged})[0],
    }
    observations.append(_observation(
        "refused_without_a_valid_capability",
        {"every attempt 403": set(refusals.values()) == {403},
         "nothing moved": exists(src) and not exists(dst),
         "still pending on both": [r.call("GET", f"/api/plans/{plan_id}")[1].get("status") for r in (a, b)] == ["pending_approval"] * 2},
        statuses=refusals))

    token = sign(created["review"])
    approved_status, approved = b.call("POST", f"/api/plans/{plan_id}/approve", {"actor": "operator", "approval_token": token})
    seen_on_a = a.call("GET", f"/api/plans/{plan_id}")[1].get("status")
    observations.append(_observation(
        "approved_on_the_other",
        {"approved on B (200, applied)": approved_status == 200 and approved.get("status") == "applied",
         "the file moved": not exists(src) and exists(dst),
         "A reports it applied": seen_on_a == "applied"},
        statuses={"approve_on_b": approved_status}, status_on_a=seen_on_a))

    replay_status, _ = a.call("POST", f"/api/plans/{plan_id}/approve", {"actor": "operator", "approval_token": token})
    observations.append(_observation(
        "replay_after_apply",
        {"the same capability on A answers 409": replay_status == 409, "moved once": not exists(src) and exists(dst)},
        statuses={"replay_on_a": replay_status}))

    src2, dst2 = MOVES["rejected"]
    second_status, second = a.call("POST", "/api/plans/", {"src": src2, "dst": dst2})
    reject_status, _ = a.call("POST", f"/api/plans/{second['plan_id']}/reject", {"actor": "operator", "note": "keep it"})
    second_on_b = b.call("GET", f"/api/plans/{second['plan_id']}")[1]
    late_status, _ = b.call("POST", f"/api/plans/{second['plan_id']}/approve", {"approval_token": sign(second["review"])})
    observations.append(_observation(
        "rejection_shared",
        {"rejected on A (200)": reject_status == 200,
         "B reports it rejected, with its actor and note": (second_on_b.get("status"), second_on_b.get("actor"), second_on_b.get("note")) == ("rejected", "operator", "keep it"),
         "an approval on B answers 409": late_status == 409,
         "nothing moved": exists(src2) and not exists(dst2)},
        statuses={"propose_on_a": second_status, "reject_on_a": reject_status, "approve_on_b": late_status}))

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        observations.append({"name": "burnt_grant_refused_elsewhere", "passed": None,
                             "skipped": "running as root, which a read-only directory does not stop"})
    else:
        src3, dst3 = MOVES["burnt"]
        locked = data_dir / Path(dst3).parent
        locked.mkdir()
        third_status, third = b.call("POST", "/api/plans/", {"src": src3, "dst": dst3})
        burnt = sign(third["review"])
        locked.chmod(0o555)
        try:
            burn_status, _ = b.call("POST", f"/api/plans/{third['plan_id']}/approve", {"actor": "operator", "approval_token": burnt})
        finally:
            locked.chmod(0o755)
        reuse_status, reuse = a.call("POST", f"/api/plans/{third['plan_id']}/approve", {"actor": "operator", "approval_token": burnt})
        unmoved = exists(src3) and not exists(dst3)
        fresh_status, fresh = a.call("POST", f"/api/plans/{third['plan_id']}/approve", {"actor": "operator", "approval_token": sign(third["review"])})
        observations.append(_observation(
            "burnt_grant_refused_elsewhere",
            {"the failed move on B is refused (403)": burn_status == 403,
             "the same capability on A is refused by the ledger (403)": reuse_status == 403 and reuse.get("detail") == "approval already consumed",
             "nothing moved meanwhile": unmoved,
             "a fresh capability applies on A": fresh_status == 200 and fresh.get("status") == "applied" and exists(dst3)},
            statuses={"propose_on_b": third_status, "approve_on_b_move_fails": burn_status, "same_capability_on_a": reuse_status,
                      "fresh_capability_on_a": fresh_status}, refused_on_a_by="ledger" if reuse.get("detail") == "approval already consumed" else None))

    logs = [replica.call("GET", "/api/plans/decisions/recent?limit=50") for replica in (a, b)]
    observations.append(_observation(
        "decision_log_shared",
        {"both answer 200": [status for status, _ in logs] == [200, 200], "the same entries": logs[0][1] == logs[1][1]},
        statuses=[status for status, _ in logs], entries=len(logs[0][1]) if logs[0][0] == 200 else None))
    return observations


def run(server_dsn: str, workdir: Path) -> dict:
    """Deploy, index, start two replicas, observe, stop them and drop the database; the results (unsanitized)."""
    from app.mcp.approvals import create_key

    database = f"mnv_replicas_{uuid.uuid4().hex[:8]}"
    data_dir = copy_fixture(workdir / "shared" / "files")
    operator = workdir / "operator"
    operator.mkdir(mode=0o700)
    create_key(operator / "operator.key", operator / "operator.pub")
    env = api_env(server_dsn, database, data_dir, model=COMPOSE_MODEL, ollama_url=NO_OLLAMA,
                  extra={"METANAVIT_APPROVAL_PUBLIC_KEY_FILE": str(operator / "operator.pub"), "METANAVIT_APPROVAL_LEDGER": "postgres"})
    create_database(server_dsn, database)
    replicas: list[Replica] = []
    try:
        with (workdir / "steps.log").open("w") as log:
            deploy = subprocess.run([sys.executable, *DEPLOY_STEP], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if deploy.returncode != 0:
            raise RuntimeError(f"the deploy step exited with {deploy.returncode}")
        step = subprocess.run([sys.executable, *INDEX_STEP], cwd=ROOT, env=env, capture_output=True, text=True)
        if step.returncode != 0:
            raise RuntimeError(f"the index step exited with {step.returncode}: {scrub(step.stderr[-500:])}")
        report = json.loads(step.stdout.strip().splitlines()[-1])
        index_report = {"exit": step.returncode, "files_indexed": report.get("files_indexed"),
                        "chunks_embedded": report.get("chunks_embedded"), "n_nodes": report.get("n_nodes")}
        for name in ("a", "b"):
            replicas.append(Replica(name, env, workdir / f"replica_{name}.log"))
        observations = observe(replicas[0], replicas[1], data_dir, operator, index_report)
        versions = _versions(database_dsn(server_dsn, database))
    finally:
        for replica in replicas:
            replica.stop()
        drop_database(server_dsn, database)
    return {"index_step": {"runs": 1, **index_report}, "observations": observations, "environment": public_env(env), **versions}


def worktree_clean(exclude: Path | None = None) -> bool:
    from app.engine.ingest_scale import worktree_clean as clean  # the same check ingest_scale.json records

    return clean(exclude)


def results(run_result: dict) -> dict:
    """The file: provenance first, then the run; every string scrubbed of paths, user and host names."""
    observations = run_result["observations"]
    return scrub({
        "schema": SCHEMA,
        "git_sha": git_sha(ROOT),
        "worktree_clean": worktree_clean(OUT),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command_line(),
        "device": device_info("cpu"),
        "replicas": 2,
        "postgres": run_result.get("postgres"),
        "pgvector": run_result.get("pgvector"),
        "embedder": "hash-256 (fallback: deterministic hashing, no model)",
        "corpus": "a copy of fixture v1 (bench/corpus/files) shared by both replicas as DATA_DIR",
        "approval_ledger": "postgres (METANAVIT_APPROVAL_LEDGER=postgres)",
        "environment": run_result["environment"],
        "index_step": run_result["index_step"],
        "observations": observations,
        "passed": all(o["passed"] is not False for o in observations),
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    server_dsn = os.environ.get("PG_TEST_DSN")
    if not server_dsn:
        print("set PG_TEST_DSN (a postgresql:// URI of a server where this run may create a database)")
        return 2
    if args.out.resolve() == OUT.resolve() and not worktree_clean(OUT):
        print("refusing to write bench/results/two_replicas.json from a tree with uncommitted changes: commit first, so the file names the code that produced it")
        return 2
    with tempfile.TemporaryDirectory(prefix="two_replicas_") as temporary:
        data = results(run(server_dsn, Path(temporary)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n")
    for observation in data["observations"]:
        state = "skipped" if observation["passed"] is None else ("ok" if observation["passed"] else "FAILED")
        print(f"{observation['name']:40s} {state}")
    print(f"wrote {args.out}")
    return 0 if data["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
