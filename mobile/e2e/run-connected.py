"""Run optional browser integration against both actual repository servers.

Example from mobile/ (Chrome must be available):
  python e2e/run-connected.py --soccer-repo /path/to/Agentic-Soccer --meta-python /path/to/meta-venv/bin/python --soccer-python /path/to/soccer-venv/bin/python

Installs nothing. Uses generated disposable test keys and only synthetic data.
Every child server is terminated when the browser checks finish or fail.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import secrets
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def provenance(root: Path, relative_paths: list[str]) -> dict:
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "files_sha256": {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest() for relative in relative_paths},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soccer-repo", required=True, type=Path)
    parser.add_argument("--meta-python", required=True, type=Path)
    parser.add_argument("--soccer-python", required=True, type=Path)
    parser.add_argument("--meta-port", type=int, default=8000)
    parser.add_argument("--agent-port", type=int, default=8002)
    parser.add_argument("--gateway-port", type=int, default=8010)
    parser.add_argument("--adapter-port", type=int, default=8011)
    parser.add_argument("--receipt", type=Path, default=Path("/tmp/agent-field-connected-integration.json"))
    args = parser.parse_args()
    mobile = Path(__file__).resolve().parents[1]
    meta = mobile.parent
    soccer = args.soccer_repo.resolve()
    for port in (args.meta_port, args.agent_port, args.gateway_port, args.adapter_port):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    servers: list[subprocess.Popen] = []
    source_files = {
        "metanavt": (meta, ["app/mobile/demo.py", "app/api/routers/retrieve.py", "app/agent/companion.py", "app/agent/companion_api.py", "mobile/src/App.tsx", "mobile/src/components/ResearchAgent.tsx", "mobile/src/lib/agent.ts", "mobile/src/lib/client.ts", "mobile/src/lib/contracts.ts", "mobile/src/fixtures/meta.json", "mobile/src/fixtures/soccer.json", "doc/research-agent-demo.json", "mobile/e2e/connected.spec.ts", "mobile/e2e/run-connected.py"]),
        "soccer": (soccer, ["gateway/app.py", "gateway/mobile.py", "agent/graph.py", "agent/parse.py", "agent/tooling.py", "src/models/predict.py", "scripts/build_demo_artifacts.py"]),
    }
    source_before = {name: provenance(root, files) for name, (root, files) in source_files.items()}
    receipt: dict = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
                     "synthetic_data": True, "gateway_model": "v0-mobile-demo", "services": {}, "browser_exit_code": None,
                     "browser_scope": "Chromium mobile viewports; not native device testing",
                     "research_agent_scope": "Actual HTTP companion demo with request mode rewritten by test harness; scripted decisions, actual retrieval/inspection tools, no LLM inference",
                     "source_before": source_before}
    with tempfile.TemporaryDirectory(prefix="agent-field-integration-") as temporary:
        temp = Path(temporary)
        gateway_key, mobile_key = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "ODDS_API_KEY": "", "API_FOOTBALL_KEY": "", "ANTHROPIC_API_KEY": "",
               "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
        artifact_root = temp / "artifacts"
        specifications = [
            ("metanavt", args.meta_python, meta, "app.mobile.demo:app", args.meta_port,
             {"MOBILE_DEMO_ORIGINS": "http://127.0.0.1:8081,http://localhost:8081"}, "/health"),
            ("research_agent", args.meta_python, meta, "app.agent.companion_api:app", args.agent_port,
             {"AGENT_ALLOWED_ORIGINS": "http://127.0.0.1:8081,http://localhost:8081", "AGENT_API_KEY": "",
              "AGENT_RETRIEVAL_URL": "", "AGENT_OLLAMA_MODEL": "", "AGENT_OLLAMA_URL": "http://127.0.0.1:11434",
              "AGENT_OLLAMA_API_KEY": ""}, "/agent/health"),
            ("soccer_gateway", args.soccer_python, soccer, "gateway.app:app", args.gateway_port,
             {"DATA_BACKEND": "demo", "MODEL_VERSION": "v0-mobile-demo", "ARTIFACT_ROOT": str(artifact_root),
              "GATEWAY_API_KEY": gateway_key, "MEMORY_PATH": str(temp / "prediction-memory.jsonl"), "AGENT_RUNNER": "inprocess",
              "TRACE_PATH": str(temp / "traces.jsonl"), "PREDICT_RATE_LIMIT": "30/minute"}, "/health"),
            ("soccer_adapter", args.soccer_python, soccer, "gateway.mobile:app", args.adapter_port,
             {"SOCCER_GATEWAY_URL": f"http://127.0.0.1:{args.gateway_port}", "SOCCER_GATEWAY_API_KEY": gateway_key,
              "MOBILE_API_KEY": mobile_key, "MOBILE_ALLOWED_ORIGINS": "http://127.0.0.1:8081,http://localhost:8081"}, "/mobile/health"),
        ]
        try:
            # A fresh clone has no trained artifacts. Build the same seeded
            # synthetic bundle in disposable storage, never inside either repo.
            build_script = ("import sys; from pathlib import Path; "
                            "import scripts.build_demo_artifacts as demo; "
                            "demo.ARTIFACT_ROOT = Path(sys.argv[1]); "
                            "demo.build('v0-mobile-demo', seed=42)")
            build = subprocess.run([str(args.soccer_python), "-c", build_script, str(artifact_root)],
                                   cwd=soccer, env=env, text=True, capture_output=True, timeout=180)
            if build.returncode:
                raise RuntimeError(f"Synthetic artifact build failed: {build.stderr[-6000:]}")
            receipt["artifact_build"] = {
                "method": "scripts.build_demo_artifacts.build(version='v0-mobile-demo', seed=42)",
                "scope": "synthetic training data; no real-match accuracy claim",
                "files_sha256": {str(path.relative_to(artifact_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sorted(artifact_root.rglob("*")) if path.is_file()},
            }
            for name, python, cwd, module, port, updates, health_path in specifications:
                log = (temp / f"{name}.log").open("w")
                process = subprocess.Popen([str(python), "-m", "uvicorn", module, "--host", "127.0.0.1", "--port", str(port)],
                                           cwd=cwd, env={**env, **updates}, stdout=log, stderr=subprocess.STDOUT)
                servers.append(process)
                log.close()
                for _ in range(150):
                    if process.poll() is not None:
                        raise RuntimeError(f"{name} exited before readiness: {(temp / f'{name}.log').read_text()[-6000:]}")
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}{health_path}", timeout=2) as response:
                            receipt["services"][name] = json.load(response)
                        break
                    except (urllib.error.URLError, TimeoutError):
                        time.sleep(0.2)
                else:
                    raise TimeoutError(f"{name} did not become ready")
            browser = subprocess.run(["npm", "run", "test:e2e", "--", "e2e/connected.spec.ts"], cwd=mobile,
                                     env={**env, "META_E2E_URL": f"http://127.0.0.1:{args.meta_port}",
                                          "AGENT_E2E_URL": f"http://127.0.0.1:{args.agent_port}",
                                          "SOCCER_E2E_URL": f"http://127.0.0.1:{args.adapter_port}", "SOCCER_E2E_TOKEN": mobile_key})
            receipt["browser_exit_code"] = browser.returncode
        finally:
            for process in reversed(servers):
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            receipt["all_test_servers_stopped"] = all(process.poll() is not None for process in servers)
            receipt["source_after"] = {name: provenance(root, files) for name, (root, files) in source_files.items()}
            receipt["source_unchanged_during_run"] = receipt["source_before"] == receipt["source_after"]
            receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    if not receipt["source_unchanged_during_run"]:
        raise RuntimeError("Repository code changed during integration; rerun against a stable checkout")
    return int(receipt["browser_exit_code"] or 0)


if __name__ == "__main__":
    raise SystemExit(main())
