# MetaNavT research agent

The companion exposes MetaNavT to the Android/iPhone app and to MCP agent clients.
It runs a bounded, read-only search-and-inspect loop. It cannot mutate files,
execute commands, approve actions, or fetch caller-selected URLs.

`POST /agent/ask` accepts `{"question":"What is the learning rate in run_047.yaml?","mode":"demo"}`.
Demo is keyless: a **scripted decision policy**, the existing retrieval router,
five packaged synthetic files, and an extractive quoted answer. It does not run
an LLM and does not establish answer accuracy on real research questions.

Live uses an operator-selected **Ollama model with tool calling**, the configured
MetaNavT `/api/retrieve/` service, and request-local evidence. The model chooses
searches and inspections; the server enforces 4 model rounds, 6 tool calls,
8 retained sources, 800 characters per returned excerpt, a request deadline,
and an HTTP gateway concurrent-request limit. Unknown tools and extra arguments fail closed.

Each returned claim must reference an inspected source. The server assembles
`[S1]` markers and returns source paths, retrieval byte metadata, exact hashes
of returned excerpts, and executed tool events. **Citation checks establish
reference integrity**, not factual entailment, current-version accuracy or
whole-file byte verification. Failed citation checks suppress the proposed answer.

## Run the keyless API

```bash
pip install -e ".[eval,mobile,agent]"
AGENT_ALLOWED_ORIGINS=http://localhost:8081,http://127.0.0.1:8081 \
  python -m uvicorn app.agent.companion_api:app --host 127.0.0.1 --port 8001
```

The standalone gateway keeps the retrieval app independent. Set the mobile app's
research-agent URL to this service's HTTPS base URL for a device, or its loopback
URL for a local browser. The browser's offline agent demo requires no backend.

```bash
curl http://127.0.0.1:8001/agent/ask -H 'Content-Type: application/json' \
  --data '{"question":"What GPU resources does run_047.sbatch request?","mode":"demo"}'
```

`GET /agent/health` reports demo readiness and whether live settings exist. It
does not assert that Ollama is reachable, a model is installed, or that tool
calling succeeds. A live request is necessary to validate that capability.

## Connect a live model

Run your indexed MetaNavT service separately. Configure the following variables
on the agent gateway, then restart it. Provider endpoints and provider keys never
enter phone settings or tool arguments.

| Variable | Purpose |
|---|---|
| `AGENT_API_KEY` | Required gateway key; mobile sends it in `X-API-Key` for live requests |
| `AGENT_RETRIEVAL_URL` | Fixed indexed MetaNavT base URL, e.g. `http://127.0.0.1:8000` |
| `AGENT_RETRIEVAL_API_KEY` | Optional server-side retrieval key, sent only to that endpoint |
| `AGENT_OLLAMA_URL` | Ollama base URL; defaults to `http://127.0.0.1:11434` |
| `AGENT_OLLAMA_MODEL` | Installed model with tool support; required for live mode |
| `AGENT_OLLAMA_API_KEY` | Optional server-side provider bearer key |
| `AGENT_ALLOWED_ORIGINS` | Comma-separated explicit browser origins; no wildcard |
| `AGENT_TIMEOUT_SECONDS` | Entire request deadline, default 60, allowed 5–120 |
| `AGENT_MAX_CONCURRENT` | Per-process active request cap, default 4, allowed 1–8 |

Expose device-facing services over HTTPS. An API key authenticates the gateway
but does not itself encrypt transport. Model and retrieval calls use operator
configured endpoints, disable redirects and environment proxies, bound upstream
bodies, and return sanitized errors. Live is rejected before network work when
authentication or configuration is missing. Retrieved source text is untrusted
data; the prompt instructs the model to ignore instructions found in files.

The live adapter follows Ollama's documented [tool-calling loop](https://docs.ollama.com/capabilities/tool-calling)
and [`/api/chat` contract](https://docs.ollama.com/api/chat). It requests a JSON
final answer with up to three claims, validates each reference, and assembles
the displayed answer. Models that return unstructured text instead of this
contract are blocked, rather than served as grounded answers.

## Use it from an MCP agent

The `agent` extra uses the official MCP Python SDK 2.x, matching MetaNavT's
existing SDK server. Start the read-only companion entry point:

```bash
python -m app.mcp.research_companion
```

An MCP client's server configuration can launch:

```json
{
  "mcpServers": {
    "metanavt-research": {
      "command": "/absolute/path/to/venv/bin/python",
      "args": ["-m", "app.mcp.research_companion"],
      "cwd": "/absolute/path/to/MetaNavT"
    }
  }
}
```

It registers only `ask_research(question, mode="demo")`. In live mode, its
credentials come from the local server environment; they are not tool inputs.
The same bounded loop runs internally and returns the answer, sources, citation
check and trace. Existing `app.mcp.server_sdk` remains the separate full research
filesystem server, including its operator-controlled action gates.

## Validation scope

```bash
python -m pytest -q tests/eval/test_companion_agent.py tests/mcp/test_research_companion.py
```

The tests run real keyless fixture retrieval and SDK stdio calls. Live protocol,
auth, bounded execution, unavailable providers and citation failures are tested
with simulated provider responses. This release does **not** claim a completed
live-model evaluation or measured LLM answer accuracy.
