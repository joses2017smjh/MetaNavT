"""docker-compose.prod.yml, deploy/Caddyfile and deploy/prod.env.example: what the production rehearsal must keep.

There is no Docker on the development host, and CI only checks that the file resolves (`docker compose config` in the
docker-smoke job); nothing starts this stack. So what can be read from the files is pinned here: every image pinned by
digest (Postgres by the digest docker-compose.yml uses), Caddy the only service that publishes ports, the api on the
deploy contract (doc/engineering-reference.md, Public demo mode) with the retrieval settings of the stack CI smokes,
health checks and restarts everywhere, and no secret value in the repository.
"""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PROD = ROOT / "docker-compose.prod.yml"
CADDYFILE = ROOT / "deploy" / "Caddyfile"
PROD_ENV = ROOT / "deploy" / "prod.env.example"
PINNED = re.compile(r"[a-z0-9][a-z0-9._/-]*:[\w.-]+@sha256:[0-9a-f]{64}")
OWN_IMAGE = "${METANAVIT_IMAGE:-ghcr.io/joses2017smjh/metanavt:main-ml}"  # the tag CI's image-ml job pushes
SECRET_NAME = re.compile(r"(KEY|PASSWORD|SECRET|TOKEN)$")  # a name ending so (not ANSWER_MAX_TOKENS)

# The deploy contract: values the topology fixes, and the defaults of the settings deploy/prod.env may change.
# test_the_contract_is_the_reference_s checks both against the contract block of doc/engineering-reference.md.
FIXED = {"PUBLIC_DEMO": "1", "REDIS_URL": "redis://redis:6379/0", "TRUSTED_PROXY_HOPS": "1"}
DEFAULTS = {
    "ANSWER_MODEL": "claude-opus-5-5",
    "ANSWER_EFFORT": "low",
    "ANSWER_MAX_TOKENS": "4096",
    "ANSWER_TIMEOUT_SECONDS": "30",
    "ANSWER_MAX_RETRIES": "2",
    "ANSWER_DAILY_BUDGET_USD": "2.00",
    "ANSWER_CACHE_TTL_SECONDS": "86400",
    "ANSWER_LIMIT_PER_MINUTE": "5",
    "ANSWER_LIMIT_PER_DAY": "50",
    "MAX_BODY_BYTES": "8192",
    "WEB_ORIGIN": "https://${DOMAIN}",  # the origin this stack serves the web client from
    "ANTHROPIC_API_KEY": "",  # the secret: from deploy/prod.env only; empty means answers are off
}


def _compose(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _services() -> dict:
    return _compose(PROD)["services"]


def _default(value: str) -> tuple[str, str]:
    """`${NAME:-default}` -> (NAME, default); the default may hold another `${...}`."""
    match = re.fullmatch(r"\$\{(\w+):-(.*)\}", value)
    assert match, value
    return match.group(1), match.group(2)


def _interpolated_names(text: str) -> set[str]:
    return set(re.findall(r"\$\{(\w+)", text))


def _required_names(text: str) -> set[str]:
    return set(re.findall(r"\$\{(\w+):\?", text))


def _example_names() -> set[str]:
    """The names deploy/prod.env.example sets or lists (commented out, with its default)."""
    return set(re.findall(r"^#?\s?([A-Z][A-Z0-9_]*)=", PROD_ENV.read_text(), re.M))


def test_the_stack_is_four_published_images_pinned_by_digest():
    services = _services()
    assert set(services) == {"api", "db", "redis", "caddy"}
    assert not any("build" in service for service in services.values())  # runs published images only
    base = _compose(ROOT / "docker-compose.yml")["services"]["db"]["image"]
    ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    assert services["db"]["image"] == base == ci["jobs"]["parity"]["services"]["postgres"]["image"]  # one Postgres
    assert services["redis"]["image"].startswith("redis:7.") and PINNED.fullmatch(services["redis"]["image"])
    assert services["caddy"]["image"].startswith("caddy:2.") and PINNED.fullmatch(services["caddy"]["image"])
    assert services["api"]["image"] == OWN_IMAGE
    (push,) = [s["run"] for s in ci["jobs"]["image-ml"]["steps"] if "docker push" in s.get("run", "")]
    assert 'docker push "$IMAGE:main-ml"' in push.splitlines()  # the default tag is one CI publishes
    assert _compose(PROD)["name"] == "metanavit-prod"  # its own project: never the local stack's volumes


def test_only_caddy_publishes_ports_http_and_https():
    services = _services()
    for name, service in services.items():
        assert service.get("network_mode") is None, name
        if name != "caddy":
            assert "ports" not in service, name  # db, redis and the api answer only on the stack's own network
    assert set(services["caddy"]["ports"]) == {"80:80", "443:443", "443:443/udp"}


def test_the_api_runs_the_deploy_contract():
    api = _services()["api"]
    env = api["environment"]
    for name, value in FIXED.items():
        assert env[name] == value, name  # fixed by the topology, never read from the env file
    for name, value in DEFAULTS.items():
        assert _default(env[name]) == (name, value), name
    assert env["ENVIRONMENT"] == "prod"  # main.py serves the web client's export at / only outside dev
    assert api["command"][-1].endswith("exec uvicorn main:app --host 0.0.0.0 --port 8000")
    assert "http://127.0.0.1:8000/health" in " ".join(api["healthcheck"]["test"])
    assert re.search(r"^\s*reverse_proxy api:8000$", CADDYFILE.read_text(), re.M)
    assert {"db": {"condition": "service_healthy"}, "redis": {"condition": "service_healthy"}} == api["depends_on"]


def test_the_contract_is_the_reference_s():
    """The contract block under Public demo mode lists the names the production Compose file sets: every one is set
    here, fixed or with the block's default (a placeholder such as <secret> or <site> stands for the host's value)."""
    text = (ROOT / "doc" / "engineering-reference.md").read_text()
    section = text.split("### Public demo mode", 1)[1]
    block = section.split("```text", 1)[1].split("```", 1)[0]
    contract = dict(re.findall(r"(?:^|\s)([A-Z][A-Z0-9_]+)=(\S+)", block, re.M))
    # CloudFront's origin header is checked where CloudFront reaches the API directly (the single host); here Caddy is
    # the only entry, so the rehearsal leaves the check off.
    assert contract.keys() - {"ORIGIN_VERIFY_SECRET"} == FIXED.keys() | DEFAULTS.keys()
    assert "ORIGIN_VERIFY_SECRET" not in PROD.read_text()
    for name, value in contract.items():
        if "<" not in value:
            assert (FIXED.get(name) or DEFAULTS.get(name)) == value, name
    assert contract["ANTHROPIC_API_KEY"] == "<secret>" and DEFAULTS["ANTHROPIC_API_KEY"] == ""
    assert contract["WEB_ORIGIN"] == "https://<site>" and DEFAULTS["WEB_ORIGIN"] == "https://${DOMAIN}"
    assert "the API listens on port 8000; health: GET /health" in block


def test_the_api_serves_the_retrieval_stack_ci_smokes():
    """docker-compose.yml with docker-compose.ml.yml is the stack CI's image-ml job builds and smokes: the prod api
    keeps every setting of it but the database's address, the log format, the deploy contract's own names (a local
    stack may set those differently) and the Ollama address: no Ollama runs here. The chat provider stays, because the
    index step the command runs before the server calls init_settings, which stops without one."""
    base = _compose(ROOT / "docker-compose.yml")["services"]["api"]
    ml = _compose(ROOT / "docker-compose.ml.yml")["services"]["api"]
    api = _services()["api"]
    smoked = {**base["environment"], **ml["environment"]}
    own = {"PG_CONNECTION_STRING", "PSYCOPG2_CONNECTION_STRING", "LOG_FORMAT", "ENVIRONMENT", "OLLAMA_BASE_URL",
           *FIXED, *DEFAULTS}
    assert {k: api["environment"].get(k) for k in smoked.keys() - own} == {k: smoked[k] for k in smoked.keys() - own}
    assert "OLLAMA_BASE_URL" not in api["environment"]
    index_step = (ROOT / "app" / "engine" / "ingest.py").read_text().split("def run_index_step", 1)[1].split("\ndef ", 1)[0]
    if "init_settings()" in index_step:  # it raises without MODEL_PROVIDER, and the container would restart in a loop
        assert api["environment"]["MODEL_PROVIDER"]
    assert api["command"] == base["command"]  # migrations, the index step, then the server
    assert api["healthcheck"] == {**base["healthcheck"], **ml["healthcheck"]}  # the first start's long grace period
    dockerfile = (ROOT / "Dockerfile").read_text().splitlines()
    assert not any(line.startswith("USER ") for line in dockerfile)  # root: the volume is the default HF cache
    assert api["volumes"] == ["metanavit_models:/root/.cache/huggingface"]


def test_redis_caps_memory_evicts_only_expiring_keys_and_keeps_snapshots():
    redis = _services()["redis"]
    args = redis["command"]
    assert args[0] == "redis-server"
    assert args[args.index("--maxmemory") + 1].endswith("mb")
    policy = args[args.index("--maxmemory-policy") + 1]
    assert policy.startswith("volatile-")  # never allkeys-*: the spend counter must not be evicted for space
    assert args[args.index("--appendonly") + 1] == "no"
    assert args[args.index("--save") + 1 : args.index("--save") + 3] == ["60", "1"]  # a snapshot a minute
    assert redis["volumes"] == ["metanavit_redis:/data"]  # in a volume: a restart keeps the counters


def test_every_service_checks_its_health_and_restarts():
    for name, service in _services().items():
        assert service["healthcheck"]["test"], name
        assert service["restart"] == "unless-stopped", name
        assert service["logging"]["options"]["max-size"], name  # rotated logs on a long-running host


def test_no_secret_value_is_in_the_repository():
    text = PROD.read_text() + CADDYFILE.read_text() + PROD_ENV.read_text()
    for marker in ("sk-ant-", "AKIA", "PRIVATE KEY", "ghp_", "github_pat_"):
        assert marker not in text, marker
    for name, service in _services().items():
        for key, value in service.get("environment", {}).items():
            if SECRET_NAME.search(key):  # from the env file: required, or empty by default; never a value here
                assert re.fullmatch(rf"\$\{{{key}(:-|:\?[^}}]*)\}}", value), (name, key, value)
            if "password=" in value or "@db:" in value:  # a connection string: the password from the env file
                assert "${POSTGRES_PASSWORD:?" in value, (name, key)
    for line in PROD_ENV.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not line.startswith("#") and SECRET_NAME.search(key):
            assert value == "", key  # filled in on the host, never here


def test_the_settings_come_from_deploy_prod_env_never_from_the_local_env():
    """Compose reads the project directory's .env when --env-file is not given. The names this file requires are not
    set by .env.example (the local app's settings), so starting it without --env-file deploy/prod.env stops instead
    of running on them; every name it reads is listed in deploy/prod.env.example."""
    text = PROD.read_text()
    assert _required_names(text) == {"POSTGRES_PASSWORD", "DOMAIN"}
    local = set(re.findall(r"^\s*#?\s*([A-Z][A-Z0-9_]*)=", (ROOT / ".env.example").read_text(), re.M))
    assert not _required_names(text) & local
    assert _interpolated_names(text) <= _example_names()
    example = PROD_ENV.read_text()
    for name, value in DEFAULTS.items():  # a default the example shows is the file's default
        shown = re.search(rf"^#\s?{name}=(.*)$", example, re.M)
        if shown and "${" not in value:
            assert shown.group(1) == value, name
    assert re.search(r"^#\s?METANAVIT_IMAGE=(.*)$", example, re.M).group(1) == _default(OWN_IMAGE)[1]


def test_the_caddyfile_serves_domain_caps_bodies_and_sets_headers():
    caddyfile = CADDYFILE.read_text()
    lines = [line.strip() for line in caddyfile.splitlines() if line.strip() and not line.strip().startswith("#")]
    assert "{$DOMAIN} {" in lines  # TLS for DOMAIN: Let's Encrypt for a public name, the local CA for localhost
    caddy = _services()["caddy"]
    assert caddy["environment"]["DOMAIN"].startswith("${DOMAIN:?")
    assert "./deploy/Caddyfile:/etc/caddy/Caddyfile:ro" in caddy["volumes"]
    assert any(volume.endswith(":/data") for volume in caddy["volumes"])  # certificates survive a restart
    (cap,) = re.findall(r"^max_size (\d+)KiB$", "\n".join(lines), re.M)
    api_cap = int(_default(_services()["api"]["environment"]["MAX_BODY_BYTES"])[1])
    assert int(cap) * 1024 >= api_cap  # a backstop above the api's own cap, so the api's error answers first
    assert re.search(rf"Above {int(cap) * 1024}, raise request_body max_size", PROD_ENV.read_text())
    for header in ("X-Content-Type-Options nosniff", "X-Frame-Options DENY", "-Server"):
        assert header in lines, header
    assert any(line.startswith("Referrer-Policy ") for line in lines)
    assert "@public not host localhost" in lines  # HSTS for a public name only
    assert any(line.startswith('header @public Strict-Transport-Security "max-age=') for line in lines)
    assert not any("trusted_proxies" in line for line in lines)  # so Caddy replaces a client's X-Forwarded-For


def test_ci_checks_that_the_file_resolves_in_an_always_run_job():
    ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    job = ci["jobs"]["docker-smoke"]
    assert job["if"] == "${{ github.event_name != 'schedule' }}"  # every pull request: a required check already
    runs = [step.get("run", "") for step in job["steps"]]
    (check,) = [i for i, run in enumerate(runs) if "docker-compose.prod.yml" in run]
    assert "docker compose -f docker-compose.prod.yml --env-file deploy/prod.env.example config --quiet" in runs[check]
    assert job["steps"][check]["env"]["POSTGRES_PASSWORD"]  # required, and empty in the example
    first_up = next(i for i, run in enumerate(runs) if "docker compose up" in run or "cp .env.example .env" in run)
    assert check < first_up  # before the local stack's .env exists


def test_the_filled_in_env_file_stays_out_of_git_and_the_image():
    assert "/deploy/prod.env" in (ROOT / ".gitignore").read_text().splitlines()
    assert "deploy/prod.env" in (ROOT / ".dockerignore").read_text().splitlines()  # COPY . . would bake it in
