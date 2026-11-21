"""The live demo's single host (infra/modules/single-host, infra/envs/single-host, deploy/single-host), from the files.

What a reviewer would ask first, each pinned to the code that answers it: port 80 is open to CloudFront's addresses
only and there is no SSH; the instance asks for IMDSv2 with one hop, encrypts its disk, runs on demand, may be only a
Free Tier eligible x86_64 type and uses standard CPU credits; every image runs by digest; the Compose environment is the
deploy contract with the live demo's values; no secret value is in the code. The host's deploy script runs here
against fake `docker` and `aws` commands, so its order and what stops it are exercised, not just read. Nothing here
calls AWS or Docker.
"""

import base64
import gzip
import importlib.util
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
INFRA = ROOT / "infra"
HOST_FILES = ROOT / "deploy" / "single-host"
_spec = importlib.util.spec_from_file_location("infra_hcl", Path(__file__).with_name("infra_hcl.py"))
hcl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hcl)

MODULE = hcl.read_dir(INFRA / "modules" / "single-host")
ENV = hcl.read_dir(INFRA / "envs" / "single-host")
DEMO = hcl.read_dir(INFRA / "modules" / "demo")
TEMPLATE = (INFRA / "modules" / "single-host" / "cloud-init.yaml.tftpl").read_text()
COMPOSE = yaml.safe_load((HOST_FILES / "compose.yml").read_text())
DEPLOY_SH = HOST_FILES / "deploy.sh"
PINNED = re.compile(r"[a-z0-9][a-z0-9._/-]*:[\w.-]+@sha256:[0-9a-f]{64}")  # as tests/eval/test_image_build.py
# Free Tier eligible instance types for accounts opened on or after 2025-07-15 (EC2 User Guide, "Track your Free Tier
# usage for Amazon EC2", read 2026-11-21).
FREE_TIER_TYPES = ["t3.micro", "t3.small", "t4g.micro", "t4g.small", "c7i-flex.large", "m7i-flex.large"]


def _resources(kind: str, text: str = MODULE) -> dict[str, str]:
    return {labels[1]: body for labels, body in hcl.blocks(text, "resource", kind)}


def _locals(text: str = MODULE) -> dict[str, str]:
    values = {}
    for _, body in hcl.blocks(text, "locals"):
        for name in hcl.object_keys("{" + body + "}"):
            values[name] = hcl.attr(body, name)
    return values


def _default(variable: str) -> str | None:
    return hcl.attr(hcl.block(MODULE, "variable", variable), "default")


def _contract() -> dict[str, str]:
    """NAME -> value of the deploy contract block of doc/engineering-reference.md."""
    text = (ROOT / "doc" / "engineering-reference.md").read_text()
    section = text.split("### Public demo mode", 1)[1].split("**Full application locally**", 1)[0]
    block = section.split("```text", 1)[1].split("```", 1)[0]
    return dict(re.findall(r"(?:^|\s)([A-Z][A-Z_]+)=(\S+)", block, flags=re.M))


def _app_env() -> dict[str, str]:
    """local.app_env, the host's app.env: NAME -> expression."""
    body = _locals()["app_env"].strip()[1:-1]
    return {name: hcl.attr(body, name) for name in hcl.object_keys(_locals()["app_env"])}


def _module_call() -> str:
    return hcl.block(ENV, "module", "single_host")


def _instance_memory() -> dict[str, int]:
    """local.instance_memory_mib: instance type -> MiB (its keys are quoted names with dots)."""
    return {name: int(mib) for name, mib in re.findall(r'"([\w.-]+)"\s*=\s*(\d+)', _locals()["instance_memory_mib"])}


# ---- network: port 80 from CloudFront only, no SSH, no NAT ------------------------------------------------------

def test_port_80_is_open_only_to_cloudfronts_origin_facing_addresses():
    ingress = _resources("aws_vpc_security_group_ingress_rule")
    assert set(ingress) == {"http_from_cloudfront"}  # the one way in
    rule = ingress["http_from_cloudfront"]
    assert hcl.attr(rule, "security_group_id") == "aws_security_group.host.id"
    assert hcl.attr(rule, "prefix_list_id") == "data.aws_ec2_managed_prefix_list.cloudfront.id"
    assert (hcl.attr(rule, "ip_protocol"), hcl.attr(rule, "from_port"), hcl.attr(rule, "to_port")) == ('"tcp"', "80", "80")
    assert hcl.attr(rule, "cidr_ipv4") is None and hcl.attr(rule, "cidr_ipv6") is None
    prefix_list = hcl.block(MODULE, "data", "aws_ec2_managed_prefix_list", "cloudfront")
    assert hcl.strings(hcl.attr(prefix_list, "name")) == ["com.amazonaws.global.cloudfront.origin-facing"]
    for body in _resources("aws_security_group").values():  # every rule is its own resource
        assert not hcl.nested(body, "ingress") and not hcl.nested(body, "egress")
    assert "aws_security_group_rule" not in MODULE
    assert set(_resources("aws_security_group")) == {"host"}
    assert hcl.attr(_resources("aws_network_interface")["host"], "security_groups") == "[aws_security_group.host.id]"


def test_there_is_no_port_22_no_key_pair_and_no_ssh_anywhere():
    for text in (MODULE, ENV):
        assert "key_name" not in text and "aws_key_pair" not in text
        for kind in ("aws_vpc_security_group_ingress_rule", "aws_vpc_security_group_egress_rule"):
            for body in _resources(kind, text).values():
                assert hcl.attr(body, "from_port") != "22" and hcl.attr(body, "to_port") != "22"
    assert not re.search(r"\b(ssh|sshd|authorized_keys)\b", TEMPLATE, re.I)
    instance = _resources("aws_instance")["host"]
    assert hcl.attr(instance, "iam_instance_profile") == "aws_iam_instance_profile.host.name"
    attachment = _resources("aws_iam_role_policy_attachment")["ssm_core"]
    assert hcl.strings(hcl.attr(attachment, "policy_arn")) == ["arn:${local.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore"]


def test_one_public_subnet_routes_out_through_the_internet_gateway_and_there_is_no_nat():
    assert set(_resources("aws_subnet")) == {"public"}
    assert hcl.attr(_resources("aws_subnet")["public"], "map_public_ip_on_launch") == "false"
    (route,) = _resources("aws_route").values()
    assert hcl.attr(route, "gateway_id") == "aws_internet_gateway.this.id"
    assert hcl.strings(hcl.attr(route, "destination_cidr_block")) == ["0.0.0.0/0"]
    assert "aws_nat_gateway" not in MODULE and "nat_gateway_id" not in MODULE
    # The Elastic IP is on the instance's own interface before the instance boots, and is CloudFront's origin.
    eip = _resources("aws_eip")["host"]
    assert hcl.attr(eip, "network_interface") == "aws_network_interface.host.id"
    assert hcl.attr(eip, "depends_on") == "[aws_internet_gateway.this]"
    instance = _resources("aws_instance")["host"]
    (interface,) = hcl.nested(instance, "primary_network_interface")
    assert hcl.attr(interface, "network_interface_id") == "aws_network_interface.host.id"


# ---- the instance -------------------------------------------------------------------------------------------------

def test_the_instance_requires_imdsv2_with_one_hop_and_encrypts_its_gp3_root_volume():
    instance = _resources("aws_instance")["host"]
    (metadata,) = hcl.nested(instance, "metadata_options")
    assert hcl.strings(hcl.attr(metadata, "http_tokens")) == ["required"]
    assert hcl.attr(metadata, "http_put_response_hop_limit") == "1"  # a container is one hop further: no credentials
    assert hcl.strings(hcl.attr(metadata, "http_endpoint")) == ["enabled"]
    (root,) = hcl.nested(instance, "root_block_device")
    assert hcl.attr(root, "encrypted") == "true"
    assert hcl.strings(hcl.attr(root, "volume_type")) == ["gp3"]
    assert hcl.attr(root, "volume_size") == "var.root_volume_gb" and _default("root_volume_gb") == "20"


def test_the_instance_is_on_demand_only():
    instance = _resources("aws_instance")["host"]
    assert not hcl.nested(instance, "instance_market_options")
    assert not re.search(r"spot", MODULE + ENV, re.I)
    assert set(_resources("aws_instance")) == {"host"}
    assert "aws_launch_template" not in MODULE and "aws_autoscaling_group" not in MODULE


def test_the_instance_type_must_be_free_tier_eligible_and_x86():
    variable = hcl.block(MODULE, "variable", "instance_type")
    eligible, x86 = hcl.nested(variable, "validation")
    listed = re.match(r"contains\((\[[^\]]*\]), var\.instance_type\)", hcl.attr(eligible, "condition"))
    assert hcl.strings(listed.group(1)) == FREE_TIER_TYPES
    assert hcl.attr(x86, "condition") == '!startswith(var.instance_type, "t4g.")'
    assert hcl.strings(hcl.attr(variable, "default")) == ["t3.small"]
    call = _module_call()
    assert hcl.strings(hcl.attr(call, "instance_type")) == ["t3.small"]  # the owner's decision (doc/deploy.md)
    assert hcl.attr(call, "serve_reranker") == "false"
    assert list(_instance_memory()) == [t for t in FREE_TIER_TYPES if not t.startswith("t4g.")]  # each allowed type
    # the AMI is x86_64 too, from AWS's public parameter, and a newer one never replaces a running host
    ami = hcl.block(MODULE, "data", "aws_ssm_parameter", "al2023")
    assert hcl.strings(hcl.attr(ami, "name")) == ["/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"]
    instance = _resources("aws_instance")["host"]
    assert hcl.attr(instance, "ami") == "data.aws_ssm_parameter.al2023.insecure_value"
    (lifecycle,) = hcl.nested(instance, "lifecycle")
    assert hcl.attr(lifecycle, "ignore_changes") == "[ami]"


def test_burstable_types_use_standard_cpu_credits():
    instance = _resources("aws_instance")["host"]
    (dynamic,) = [body for labels, body in hcl.blocks(instance, "dynamic", "credit_specification")]
    assert hcl.attr(dynamic, "for_each") == 'startswith(var.instance_type, "t") ? [1] : []'
    (content,) = hcl.nested(dynamic, "content")
    assert hcl.strings(hcl.attr(content, "cpu_credits")) == ["standard"]  # never "unlimited": no surplus credit charges
    assert "unlimited" not in MODULE


def test_the_memory_limits_and_the_swap_fit_a_t3_small():
    """The module's precondition, with its defaults and the instance the root picks: the three limits and the host's
    reserve fit in the memory, with the swap file as a margin under them. The one-off migrate container has the api's
    limit and runs only while the api is stopped (deploy.sh), so it needs no room of its own."""
    memory = _instance_memory()
    api, db, redis, reserve, swap = (int(_default(name)) for name in
                                     ("api_memory_mib", "db_memory_mib", "redis_memory_mib", "host_reserve_mib", "swap_mib"))
    assert swap == 2048
    ram = memory["t3.small"]
    assert ram == 2048
    assert api + db + redis + reserve <= ram
    (lifecycle,) = hcl.nested(_resources("aws_instance")["host"], "lifecycle")
    conditions = [hcl.attr(p, "condition") for p in hcl.nested(lifecycle, "precondition")]
    assert conditions == [
        "var.api_memory_mib + var.db_memory_mib + var.redis_memory_mib + var.host_reserve_mib <= "
        "local.instance_memory_mib[var.instance_type]",
        "!var.serve_reranker || (local.instance_memory_mib[var.instance_type] >= 4096 && var.api_memory_mib >= 3072)"]
    # the limits reach Compose through app.env, and every service that runs has one
    env = _app_env()
    assert (env["API_MEMORY"], env["DB_MEMORY"], env["REDIS_MEMORY"]) == (
        '"${var.api_memory_mib}m"', '"${var.db_memory_mib}m"', '"${var.redis_memory_mib}m"')
    limits = {name: service.get("mem_limit") for name, service in COMPOSE["services"].items()}
    assert limits == {"api": "${API_MEMORY:?set in app.env}", "migrate": "${API_MEMORY:?set in app.env}",
                      "db": "${DB_MEMORY:?set in app.env}", "redis": "${REDIS_MEMORY:?set in app.env}"}


def test_the_reranker_switch_sets_the_apps_own_not_configured_value():
    assert _locals()["reranker_model"] == 'var.serve_reranker ? "BAAI/bge-reranker-v2-m3" : "none"'
    assert _app_env()["RERANKER_MODEL"] == "local.reranker_model"
    assert COMPOSE["x-api-environment"]["RERANKER_MODEL"] == "${RERANKER_MODEL:?set in app.env}"
    retriever = (ROOT / "app" / "engine" / "retriever.py").read_text()
    assert 'name.lower() != "none"' in retriever  # load_reranker: "none" means not configured, not a failed load
    assert _default("serve_reranker") == "false"


def test_the_host_reads_only_its_two_secure_strings():
    policy = hcl.block(MODULE, "data", "aws_iam_policy_document", "parameters")
    (statement,) = hcl.nested(policy, "statement")
    assert hcl.strings(hcl.attr(statement, "actions")) == ["ssm:GetParameter"]
    assert hcl.attr(statement, "resources") == "[aws_ssm_parameter.origin_verify.arn, local.anthropic_key_parameter_arn]"
    parameter = _resources("aws_ssm_parameter")["origin_verify"]
    assert hcl.strings(hcl.attr(parameter, "type")) == ["SecureString"] and hcl.attr(parameter, "key_id") is None  # aws/ssm
    assert hcl.attr(parameter, "value") == "random_password.origin_verify.result"
    assert set(_resources("aws_ssm_parameter")) == {"origin_verify"}  # the Anthropic key is the owner's, never Terraform's
    names = _locals()
    assert hcl.strings(names["origin_verify_parameter"]) == ["/${var.project}/single-host/origin-verify"]
    assert hcl.strings(names["anthropic_key_parameter"]) == ["/${var.project}/single-host/anthropic-api-key"]
    trust = hcl.block(MODULE, "data", "aws_iam_policy_document", "ec2_trust")
    (principal,) = hcl.nested(hcl.nested(trust, "statement")[0], "principals")
    assert hcl.strings(hcl.attr(principal, "identifiers")) == ["ec2.amazonaws.com"]


def test_the_instance_carries_the_tags_the_deploy_role_requires():
    tags = _locals()["instance_tags"]
    body = tags.strip()[1:-1]
    assert hcl.object_keys(tags) == ["Name", "Project", "Target"]
    assert (hcl.attr(body, "Project"), hcl.strings(hcl.attr(body, "Target"))) == ("var.project", ["single-host"])
    assert hcl.attr(_resources("aws_instance")["host"], "tags") == "local.instance_tags"


# ---- CloudFront ---------------------------------------------------------------------------------------------------

def test_cloudfront_reaches_the_host_over_http_with_its_header_and_caches_only_the_hashed_assets():
    distribution = _resources("aws_cloudfront_distribution")["this"]
    (origin,) = hcl.nested(distribution, "origin")
    assert hcl.attr(origin, "domain_name") == "aws_eip.host.public_dns"
    (custom,) = hcl.nested(origin, "custom_origin_config")
    assert hcl.strings(hcl.attr(custom, "origin_protocol_policy")) == ["http-only"] and hcl.attr(custom, "http_port") == "80"
    (header,) = hcl.nested(origin, "custom_header")
    assert (hcl.attr(header, "name"), hcl.attr(header, "value")) == ("local.origin_header_name", "random_password.origin_verify.result")
    assert hcl.strings(_locals()["origin_header_name"]) == ["X-Origin-Verify"]
    assert hcl.strings(hcl.attr(distribution, "price_class") or "") == [] and _default("price_class") == '"PriceClass_100"'
    assert hcl.attr(distribution, "web_acl_id") is None  # no WAF: not free
    (certificate,) = hcl.nested(distribution, "viewer_certificate")
    assert hcl.attr(certificate, "cloudfront_default_certificate") == "true"
    policies = {labels[1]: hcl.strings(hcl.attr(body, "name"))[0] for labels, body in hcl.blocks(MODULE, "data")
                if labels[0].startswith("aws_cloudfront_")}
    assert policies == {"disabled": "Managed-CachingDisabled", "optimized": "Managed-CachingOptimized",
                        "all_viewer_except_host": "Managed-AllViewerExceptHostHeader",
                        "security": "Managed-SecurityHeadersPolicy"}
    disabled, optimized = "data.aws_cloudfront_cache_policy.disabled.id", "data.aws_cloudfront_cache_policy.optimized.id"
    security = "data.aws_cloudfront_response_headers_policy.security.id"
    (default,) = hcl.nested(distribution, "default_cache_behavior")
    assert hcl.attr(default, "cache_policy_id") == disabled and hcl.attr(default, "response_headers_policy_id") == security
    assert hcl.strings(hcl.attr(default, "viewer_protocol_policy")) == ["redirect-to-https"]
    assert hcl.strings(_locals()["api_paths"]) == ["/api/*", "/health"]
    (dynamic,) = [body for labels, body in hcl.blocks(distribution, "dynamic", "ordered_cache_behavior")]
    (api,) = hcl.nested(dynamic, "content")
    assert hcl.attr(api, "cache_policy_id") == disabled
    assert hcl.attr(api, "origin_request_policy_id") == "data.aws_cloudfront_origin_request_policy.all_viewer_except_host.id"
    assert hcl.attr(api, "response_headers_policy_id") == security
    (static,) = hcl.nested(distribution, "ordered_cache_behavior")
    assert hcl.strings(hcl.attr(static, "path_pattern")) == ["/_next/static/*"]
    assert hcl.attr(static, "cache_policy_id") == optimized and hcl.attr(static, "response_headers_policy_id") == security
    for behaviour in (default, api, static):
        assert hcl.strings(hcl.attr(behaviour, "target_origin_id")) == ["host"]
        assert hcl.strings(hcl.attr(behaviour, "viewer_protocol_policy"))[0] in {"redirect-to-https", "https-only"}


def test_the_status_check_alarm_is_optional_and_emails_through_sns():
    alarm = _resources("aws_cloudwatch_metric_alarm")["status_check"]
    assert hcl.attr(alarm, "count") == "local.alarm ? 1 : 0"
    assert hcl.strings(hcl.attr(alarm, "metric_name")) == ["StatusCheckFailed"]
    assert hcl.attr(alarm, "dimensions") == "{ InstanceId = aws_instance.host.id }"
    assert _locals()["alarm"] == 'var.alert_email != null && var.alert_email != ""'
    assert hcl.strings(hcl.attr(_resources("aws_sns_topic_subscription")["alarm_email"], "protocol")) == ["email"]
    assert _default("alert_email") == "null"


def test_the_outputs_and_the_roots_backend():
    for name in ("public_url", "instance_id", "cloudfront_distribution_id", "github_environment_variables"):
        hcl.block(MODULE, "output", name)
        assert hcl.attr(hcl.block(ENV, "output", name), "value") == f"module.single_host.{name}"
    variables = hcl.object_keys(hcl.attr(hcl.block(MODULE, "output", "github_environment_variables"), "value"))
    assert variables == ["PUBLIC_URL", "SINGLE_HOST_NAME", "SERVE_RERANKER"]
    assert hcl.attr(hcl.block(ENV, "output", "image_digest"), "value") == "var.image_digest"  # the plan job reads it
    backend = hcl.block(hcl.block(ENV, "terraform"), "backend", "s3")
    assert hcl.strings(hcl.attr(backend, "key")) == ["single-host/terraform.tfstate"]
    assert hcl.attr(backend, "use_lockfile") == "true" and hcl.attr(backend, "bucket") is None  # given at init
    assert hcl.strings(hcl.attr(_module_call(), "source")) == ["../../modules/single-host"]
    assert hcl.attr(_module_call(), "image_digest") == "var.image_digest"


# ---- images by digest -------------------------------------------------------------------------------------------

def test_every_image_runs_by_digest():
    services = COMPOSE["services"]
    base = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]["db"]["image"]
    rehearsal = yaml.safe_load((ROOT / "docker-compose.prod.yml").read_text())["services"]["redis"]["image"]
    assert services["db"]["image"] == base and PINNED.fullmatch(base)  # the Postgres docker-compose.yml pins
    assert services["redis"]["image"] == rehearsal and PINNED.fullmatch(rehearsal)  # the Redis the rehearsal pins
    # the api's: METANAVIT_IMAGE, which deploy.sh writes as <repository>@<digest> after checking the digest's form
    assert services["api"]["image"] == services["migrate"]["image"] == "${METANAVIT_IMAGE:?set by deploy.sh in image.env}"
    script = DEPLOY_SH.read_text()
    assert "DIGEST='^sha256:[0-9a-f]{64}$'" in script and '[[ $digest =~ $DIGEST ]]' in script
    assert 'image="$(setting IMAGE_REPOSITORY)@$digest"' in script
    assert "printf 'METANAVIT_IMAGE=%s\\n' \"$image\"" in script
    variable = hcl.block(MODULE, "variable", "image_digest")
    (validation,) = hcl.nested(variable, "validation")
    assert hcl.attr(validation, "condition") == 'can(regex("^sha256:[0-9a-f]{64}$", var.image_digest))'
    assert hcl.attr(variable, "default") is None
    assert "/opt/metanavit/deploy.sh ${image_digest}" in TEMPLATE
    # the Compose plugin: a release by version, checked against its sha256 before it is made executable
    values = _locals()
    assert re.fullmatch(r"v\d+\.\d+\.\d+", hcl.strings(values["compose_version"])[0])
    assert re.fullmatch(r"[0-9a-f]{64}", hcl.strings(values["compose_sha256"])[0])
    runcmd = _rendered()["runcmd"]
    check = next(i for i, line in enumerate(runcmd) if "sha256sum --check --strict" in line)
    download = next(i for i, line in enumerate(runcmd) if line.startswith("curl ") and "docker-compose" in line)
    executable = next(i for i, line in enumerate(runcmd) if line.startswith("chmod 755 ") and "docker-compose" in line)
    assert download < check < executable


# ---- the Compose environment is the deploy contract's, with the live demo's values -------------------------------

# The names the host sets beyond the contract: the same as the ECS task's (infra/modules/demo, platform_environment)
# but PGSSLMODE (the database is a container on the host's own network, not RDS).
PLATFORM = {"ENVIRONMENT", "LOG_FORMAT", "PG_CONNECTION_STRING", "PSYCOPG2_CONNECTION_STRING", "RETRIEVE_TIMEOUT_SECONDS",
            "MODEL_PROVIDER", "EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_DIM", "RERANKER_MODEL",
            "RERANKER_PRECISION", "RERANKER_MAX_LENGTH", "RERANK_DEPTH", "RERANKER_REQUIRED", "BGE_ALLOW_DOWNLOAD",
            "RETRIEVAL_MODE", "DATA_DIR", "INDEX_ON_START", "INGEST_MAX_REJECTED", "RETRIEVE_K", "RERANK_TOP_N",
            "ENABLE_ROUTER"}


def test_the_api_environment_is_the_contract_and_only_listed_extras():
    contract = _contract()
    env = COMPOSE["services"]["api"]["environment"]
    assert contract.keys() <= env.keys()  # every name of the contract, ORIGIN_VERIFY_SECRET included
    assert env.keys() - contract.keys() == PLATFORM
    ecs = hcl.object_keys(_locals(DEMO)["platform_environment"])
    assert PLATFORM == set(ecs) - {"PGSSLMODE"}
    assert COMPOSE["services"]["migrate"]["environment"] == env  # one environment (a YAML anchor) for both


def test_the_contract_values_are_the_live_demos():
    contract, env = _contract(), COMPOSE["services"]["api"]["environment"]
    assert env["PUBLIC_DEMO"] == "1"
    assert env["ENVIRONMENT"] == "prod"  # main.py serves the web client's export at /
    assert env["TRUSTED_PROXY_HOPS"] == "1"  # CloudFront appends the viewer; nothing else is in front
    assert env["REDIS_URL"] == "redis://redis:6379/0" == contract["REDIS_URL"]
    for name in ("ANSWER_EFFORT", "ANSWER_MAX_TOKENS", "ANSWER_CACHE_TTL_SECONDS", "ANSWER_LIMIT_PER_MINUTE",
                 "ANSWER_LIMIT_PER_DAY", "MAX_BODY_BYTES"):
        assert env[name] == contract[name], name
    # WEB_ORIGIN, the answer model and the daily budget come from app.env, which Terraform writes
    app_env = _app_env()
    for name in ("WEB_ORIGIN", "ANSWER_MODEL", "ANSWER_DAILY_BUDGET_USD"):
        assert env[name] == f"${{{name}:?set in app.env}}", name
    assert app_env["WEB_ORIGIN"] == "local.public_url"
    assert hcl.strings(_locals()["public_url"]) == ["https://${aws_cloudfront_distribution.this.domain_name}"]
    assert app_env["ANSWER_MODEL"] == "var.answer_model" and hcl.strings(_default("answer_model")) == ["claude-haiku-5-5"]
    assert app_env["ANSWER_DAILY_BUDGET_USD"] == 'format("%.2f", var.answer_daily_budget_usd)'
    assert float(_default("answer_daily_budget_usd")) == 0.75
    call = _module_call()
    assert hcl.strings(hcl.attr(call, "answer_model")) == ["claude-haiku-5-5"]
    assert float(hcl.attr(call, "answer_daily_budget_usd")) == 0.75
    # the secrets: the key optional (empty: answers off), the origin value required (the stack does not start without)
    assert env["ANTHROPIC_API_KEY"] == "${ANTHROPIC_API_KEY:-}"
    assert env["ORIGIN_VERIFY_SECRET"].startswith("${ORIGIN_VERIFY_SECRET:?")
    # an answer ends inside CloudFront's origin timeout: retrieval budget + every attempt of the Claude call + backoff
    retrieve, timeout = float(env["RETRIEVE_TIMEOUT_SECONDS"]), float(env["ANSWER_TIMEOUT_SECONDS"])
    retries, origin = int(env["ANSWER_MAX_RETRIES"]), float(_default("origin_read_timeout"))
    assert retrieve + timeout * (retries + 1) + 5 <= origin, (retrieve, timeout, retries, origin)


def test_every_interpolated_name_has_a_source_on_the_host():
    """app.env (Terraform), image.env and secrets.env (deploy.sh) between them give every ${NAME} compose.yml reads,
    and a required one stops Compose instead of starting a service without it."""
    text = (HOST_FILES / "compose.yml").read_text()
    used = set(re.findall(r"\$\{(\w+)", text))
    secrets = set(re.findall(r"^(?:\s*)printf '((?:[A-Z_]+=%s\\n)+)'", DEPLOY_SH.read_text(), re.M)[0].replace("=%s\\n", " ").split())
    assert secrets == {"ORIGIN_VERIFY_SECRET", "ANTHROPIC_API_KEY", "POSTGRES_PASSWORD"}
    assert used == (set(_app_env()) - {"IMAGE_REPOSITORY", "AWS_REGION", "ORIGIN_VERIFY_PARAMETER",
                                       "ANTHROPIC_KEY_PARAMETER"}) | {"METANAVIT_IMAGE"} | secrets
    optional = set(re.findall(r"\$\{(\w+):-", text))
    assert optional == {"ANTHROPIC_API_KEY"}  # everything else is required (:?)


def test_only_the_api_publishes_a_port_and_every_long_running_service_checks_its_health_and_restarts():
    services = COMPOSE["services"]
    assert set(services) == {"api", "migrate", "db", "redis"}
    assert services["api"]["ports"] == ["80:8000"]
    assert [name for name, s in services.items() if "ports" in s] == ["api"]
    assert not any("build" in s or s.get("network_mode") for s in services.values())
    for name in ("api", "db", "redis"):
        assert services[name]["healthcheck"]["test"], name
        assert services[name]["restart"] == "unless-stopped", name
        assert services[name]["logging"]["options"]["max-size"], name
    assert "http://127.0.0.1:8000/health" in " ".join(services["api"]["healthcheck"]["test"])  # loopback: no header needed
    assert services["api"]["volumes"] == ["models:/root/.cache/huggingface"]
    assert services["api"]["command"][:6] == ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    keepalive = int(services["api"]["command"][services["api"]["command"].index("--timeout-keep-alive") + 1])
    distribution = _resources("aws_cloudfront_distribution")["this"]
    (custom,) = hcl.nested(hcl.nested(distribution, "origin")[0], "custom_origin_config")
    assert keepalive > int(hcl.attr(custom, "origin_keepalive_timeout"))  # CloudFront closes an idle connection first
    assert set(COMPOSE["volumes"]) == {"pgdata", "redis", "models"}


def test_the_migrate_service_runs_the_ecs_migrate_tasks_command_once_per_deploy():
    migrate = COMPOSE["services"]["migrate"]
    ecs = hcl.attr(_locals(DEMO)["migrate_container"].strip()[1:-1], "command")
    assert migrate["command"] == hcl.strings(ecs) == ["sh", "-c", "python -m app.database.migrate && python -m app.engine.ingest"]
    assert migrate["profiles"] == ["deploy"]  # `up` never starts it
    assert migrate["healthcheck"] == {"disable": True} and "restart" not in migrate
    assert migrate["depends_on"] == {"db": {"condition": "service_healthy"}}


SECRET_NAME = re.compile(r"(KEY|PASSWORD|SECRET|TOKEN)$")


def test_no_secret_value_is_in_the_host_files():
    for path in sorted(HOST_FILES.iterdir()):
        text = path.read_text()
        for marker in ("sk-ant-", "AKIA", "ASIA", "PRIVATE KEY", "ghp_", "github_pat_"):
            assert marker not in text, (path.name, marker)
        assert not re.search(r"(?<![\w.:-])\d{12}(?![\w])", text), path.name  # an account id
        assert not re.search(r"[\w.+-]+@[\w-]+\.[a-z]{2,}\b", text), path.name  # an email address
    for name, service in COMPOSE["services"].items():
        for key, value in (service.get("environment") or {}).items():
            if SECRET_NAME.search(key):
                assert re.fullmatch(rf"\$\{{{key}(:-|:\?[^}}]*)\}}", value), (name, key, value)
            if "password=" in value or "@db:" in value:
                assert "${POSTGRES_PASSWORD:?" in value, (name, key)


# ---- cloud-init -------------------------------------------------------------------------------------------------

def _rendered() -> dict:
    """The template rendered as templatefile() would render it (it holds only ${name} interpolations)."""
    names = set(re.findall(r"\$\{(\w+)\}", TEMPLATE))
    call = re.search(r"templatefile\(\"\$\{path\.module\}/cloud-init\.yaml\.tftpl\", \{(.*?)\}\)", _locals()["cloud_init"], re.S)
    assert call and set(hcl.object_keys("{" + call.group(1) + "}")) == names
    b64 = lambda data: base64.b64encode(data).decode()  # noqa: E731
    values = {"compose": b64((HOST_FILES / "compose.yml").read_bytes()), "deploy": b64(DEPLOY_SH.read_bytes()),
              "unit": b64((HOST_FILES / "metanavit.service").read_bytes()), "app_env": b64(b"WEB_ORIGIN=https://d1.cloudfront.net\n"),
              "swap_mib": "2048", "compose_version": "v5.6.1", "compose_sha256": "ab" * 32, "image_digest": "sha256:" + "cd" * 32}
    assert "%{" not in TEMPLATE and names == set(values)
    return yaml.safe_load(re.sub(r"\$\{(\w+)\}", lambda m: values[m.group(1)], TEMPLATE))


def test_cloud_init_writes_the_committed_host_files_and_embeds_them_from_the_repository():
    files = {f["path"]: f for f in _rendered()["write_files"]}
    assert set(files) == {"/opt/metanavit/compose.yml", "/opt/metanavit/deploy.sh", "/opt/metanavit/app.env",
                          "/etc/systemd/system/metanavit.service"}
    for path, source in (("/opt/metanavit/compose.yml", "compose.yml"), ("/opt/metanavit/deploy.sh", "deploy.sh"),
                         ("/etc/systemd/system/metanavit.service", "metanavit.service")):
        assert base64.b64decode(files[path]["content"]) == (HOST_FILES / source).read_bytes(), path
        assert files[path]["encoding"] == "b64" and files[path]["owner"] == "root:root"
    assert files["/opt/metanavit/deploy.sh"]["permissions"] == "0700"
    cloud_init = _locals()["cloud_init"]
    for name, source in (("compose", "compose.yml"), ("deploy", "deploy.sh"), ("unit", "metanavit.service")):
        assert f'{name} ' in cloud_init and f'filebase64("${{local.host_files}}/{source}")' in cloud_init
    assert hcl.strings(_locals()["host_files"]) == ["${path.module}/../../../deploy/single-host"]
    instance = _resources("aws_instance")["host"]
    assert hcl.attr(instance, "user_data_base64") == "base64gzip(local.cloud_init)"
    assert hcl.attr(instance, "user_data_replace_on_change") == "true"  # cloud-init runs once per instance


def test_cloud_init_makes_the_swap_installs_docker_then_enables_the_unit_and_deploys_last():
    runcmd = _rendered()["runcmd"]
    assert runcmd[0] == "set -eu"  # one failing command stops the rest

    def first(prefix):
        return next(i for i, line in enumerate(runcmd) if line.startswith(prefix))

    swap = first("dd if=/dev/zero of=/swapfile bs=1M count=")
    assert runcmd[swap].split("count=")[1].split()[0] == "2048"
    assert swap < first("mkswap /swapfile") < first("swapon /swapfile") < first("dnf install -y docker")
    assert first("dnf install -y docker") < first("systemctl enable --now docker") < first("docker compose version")
    assert first("systemctl daemon-reload") < first("systemctl enable metanavit.service") < first("/opt/metanavit/deploy.sh")
    assert runcmd[-1] == "/opt/metanavit/deploy.sh sha256:" + "cd" * 32
    assert any(line.startswith('echo "/swapfile none swap') and line.endswith(">> /etc/fstab") for line in runcmd)


def test_the_user_data_fits_ec2s_limit_once_compressed():
    """EC2 takes at most 16 KB of user data (before base64); the module gzips it (base64gzip). With full-size values
    (a 64-hex digest, an app.env of the real length) the compressed text must leave room."""
    rendered = re.sub(r"\$\{(\w+)\}", lambda m: "x" * 64, TEMPLATE)
    for source in ("compose.yml", "deploy.sh", "metanavit.service"):
        rendered += base64.b64encode((HOST_FILES / source).read_bytes()).decode()
    rendered += base64.b64encode(b"NAME=" + b"v" * 60 + b"\n" * 12).decode()
    assert len(gzip.compress(rendered.encode(), compresslevel=6)) < 14 * 1024


def test_the_unit_reads_the_secrets_then_starts_the_stack_at_boot():
    unit = (HOST_FILES / "metanavit.service").read_text()
    lines = [line.strip() for line in unit.splitlines() if line.strip() and not line.startswith("#")]
    assert "ExecStart=/opt/metanavit/deploy.sh --start" in lines
    assert "Requires=docker.service" in lines and "After=docker.service network-online.target" in lines
    assert "Type=oneshot" in lines and "RemainAfterExit=yes" in lines and "WantedBy=multi-user.target" in lines
    assert "Restart=on-failure" in lines


# ---- deploy.sh, against fake docker and aws commands --------------------------------------------------------------

DIGEST = "sha256:" + "ab" * 32
OLD = "ghcr.io/joses2017smjh/metanavt@sha256:" + "00" * 32
ORIGIN_VALUE = "originValue0123456789abcdefABCDEF0123456"
KEY_VALUE = "sk-test-not-a-real-key_0123"

FAKE_DOCKER = r"""#!/usr/bin/env bash
# records each call, with the image env file Compose was given; fails where the test asks it to
{
  printf 'docker'; printf ' %q' "$@"; printf '\n'
  prev=""
  for arg in "$@"; do
    if [ "$prev" = "--env-file" ] && [[ $arg == *image.env* ]]; then printf '  image-env: %s\n' "$(cat "$arg")"; fi
    prev=$arg
  done
} >> "$FAKE_LOG"
case " $* " in
  *" run "*) [ -z "${FAKE_FAIL_RUN:-}" ] || exit 7 ;;
  *" up "*)
    count=$(cat "$FAKE_STATE/up" 2>/dev/null || echo 0); count=$((count + 1)); echo "$count" > "$FAKE_STATE/up"
    if [ -n "${FAKE_FAIL_UP:-}" ] && [ "$count" -le "$FAKE_FAIL_UP" ]; then exit 1; fi ;;
  *" exec "*) echo '{"status": "ok"}' ;;
esac
exit 0
"""

FAKE_AWS = r"""#!/usr/bin/env bash
# aws ssm get-parameter --region R --name NAME --with-decryption ...: the value from FAKE_PARAMS (NAME=value lines)
printf 'aws %s\n' "$*" >> "$FAKE_LOG"
name=""; prev=""
for arg in "$@"; do [ "$prev" = "--name" ] && name=$arg; prev=$arg; done
if [ -n "${FAKE_AWS_DOWN:-}" ]; then echo "Could not connect to the endpoint URL" >&2; exit 255; fi
value=$(sed -n "s|^$name=||p" "$FAKE_PARAMS")
if [ -z "$value" ]; then
  echo "An error occurred (ParameterNotFound) when calling the GetParameter operation: Parameter $name not found." >&2
  exit 254
fi
printf '%s\n' "$value"
"""


@pytest.fixture
def host(tmp_path):
    if shutil.which("bash") is None or shutil.which("flock") is None:
        pytest.skip("needs bash and flock")
    home, run, state, bin_dir, fake = (tmp_path / name for name in ("opt", "run", "state", "bin", "fake"))
    for directory in (home, bin_dir, fake):
        directory.mkdir()
    (home / "compose.yml").write_text((HOST_FILES / "compose.yml").read_text())
    (home / "app.env").write_text("# written by Terraform\nIMAGE_REPOSITORY=ghcr.io/joses2017smjh/metanavt\n"
                                  "AWS_REGION=us-west-2\nORIGIN_VERIFY_PARAMETER=/metanavit/single-host/origin-verify\n"
                                  "ANTHROPIC_KEY_PARAMETER=/metanavit/single-host/anthropic-api-key\n")
    for name, body in (("docker", FAKE_DOCKER), ("aws", FAKE_AWS)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    params = fake / "params"
    params.write_text(f"/metanavit/single-host/origin-verify={ORIGIN_VALUE}\n/metanavit/single-host/anthropic-api-key={KEY_VALUE}\n")

    class Host:
        def __init__(self):
            self.home, self.run_dir, self.state, self.params, self.log = home, run, state, params, fake / "log"

        def deploy(self, *args, **env):
            environ = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "METANAVIT_HOME": str(home),
                       "METANAVIT_RUN": str(run), "METANAVIT_STATE": str(state), "METANAVIT_RETRY_SECONDS": "0",
                       "FAKE_LOG": str(self.log), "FAKE_PARAMS": str(params), "FAKE_STATE": str(fake), **env}
            return subprocess.run(["bash", str(DEPLOY_SH), *args], env=environ, capture_output=True, text=True, timeout=60)

        def calls(self) -> list[str]:
            return self.log.read_text().splitlines() if self.log.exists() else []

        def docker_calls(self) -> list[str]:
            return [line for line in self.calls() if line.startswith("docker")]

    return Host()


def _index(calls, needle):
    return next(i for i, line in enumerate(calls) if needle in line)


def test_a_deploy_pulls_by_digest_migrates_with_the_new_image_then_starts_the_stack_and_waits(host):
    (host.home / "image.env").write_text(f"METANAVIT_IMAGE={OLD}\n")
    done = host.deploy(DIGEST)
    assert done.returncode == 0, done.stdout + done.stderr
    image = f"ghcr.io/joses2017smjh/metanavt@{DIGEST}"
    calls = host.calls()
    order = [_index(calls, "aws ssm get-parameter"), _index(calls, f"docker pull --quiet {image}"),
             _index(calls, " stop api"), _index(calls, " run --rm -T migrate"),
             _index(calls, " up --detach --remove-orphans --wait"), _index(calls, " exec -T api python"),
             _index(calls, "docker image prune --all --force")]
    assert order == sorted(order)  # the api is stopped before the one-off container loads the embedder too
    # the one-off container got the new image while the api still had the old one; up got the new one
    assert calls[_index(calls, " run --rm -T migrate") + 1] == f"  image-env: METANAVIT_IMAGE={image}"
    assert calls[_index(calls, " up --detach") + 1] == f"  image-env: METANAVIT_IMAGE={image}"
    assert (host.home / "image.env").read_text() == f"METANAVIT_IMAGE={image}\n"
    up = calls[_index(calls, " up --detach")]
    assert "--wait-timeout 1200" in up and "--project-name metanavit" in up
    assert f"--env-file {host.home}/app.env" in up and f"--env-file {host.run_dir}/secrets.env" in up
    assert "DEPLOYED " + image in done.stdout


def test_the_secrets_file_is_root_only_and_holds_the_parameters_and_a_stable_postgres_password(host):
    assert host.deploy(DIGEST).returncode == 0
    secrets = host.run_dir / "secrets.env"
    assert stat.S_IMODE(secrets.stat().st_mode) == 0o600 and stat.S_IMODE(host.run_dir.stat().st_mode) == 0o700
    values = dict(line.split("=", 1) for line in secrets.read_text().splitlines())
    assert values["ORIGIN_VERIFY_SECRET"] == ORIGIN_VALUE and values["ANTHROPIC_API_KEY"] == KEY_VALUE
    assert re.fullmatch(r"[0-9a-f]{48}", values["POSTGRES_PASSWORD"])
    assert stat.S_IMODE((host.state / "postgres-password").stat().st_mode) == 0o600
    assert host.deploy("--start").returncode == 0  # made once: the database volume keeps the first one
    again = dict(line.split("=", 1) for line in secrets.read_text().splitlines())
    assert again["POSTGRES_PASSWORD"] == values["POSTGRES_PASSWORD"]
    assert ORIGIN_VALUE not in host.deploy(DIGEST).stdout  # never printed


def test_without_an_anthropic_key_the_deploy_goes_on_with_answers_off(host):
    host.params.write_text(f"/metanavit/single-host/origin-verify={ORIGIN_VALUE}\n")
    done = host.deploy(DIGEST)
    assert done.returncode == 0, done.stderr
    assert "answers are off" in done.stdout
    assert "ANTHROPIC_API_KEY=\n" in (host.run_dir / "secrets.env").read_text()


@pytest.mark.parametrize("params, env, message", [
    ("/metanavit/single-host/anthropic-api-key=k\n", {}, "cannot read CloudFront's origin header value"),
    (f"/metanavit/single-host/origin-verify={ORIGIN_VALUE}\n", {"FAKE_AWS_DOWN": "1"}, "cannot read CloudFront's"),
    ("/metanavit/single-host/origin-verify=has space\n", {}, "is not [A-Za-z0-9_-]+"),
    (f"/metanavit/single-host/origin-verify={ORIGIN_VALUE}\n/metanavit/single-host/anthropic-api-key=a'b\n", {},
     "the Anthropic key"),
])
def test_a_missing_or_malformed_secret_stops_the_deploy_before_anything_is_pulled(host, params, env, message):
    host.params.write_text(params)
    done = host.deploy(DIGEST, **env)
    assert done.returncode == 1 and message in done.stderr
    assert host.docker_calls() == []


@pytest.mark.parametrize("argument", ["main-ml", "sha256:abc", "sha256:" + "AB" * 32, "", "--deploy"])
def test_only_a_digest_or_start_is_accepted(host, argument):
    done = host.deploy(argument)
    assert done.returncode in (1, 2) and host.docker_calls() == []


def test_a_failed_migration_starts_the_previous_image_again_and_fails(host):
    (host.home / "image.env").write_text(f"METANAVIT_IMAGE={OLD}\n")
    done = host.deploy(DIGEST, FAKE_FAIL_RUN="1")
    assert done.returncode == 1 and "the migrations or the index step failed" in done.stderr
    calls = host.calls()
    (up,) = [i for i, line in enumerate(calls) if " up --detach" in line]
    assert up > _index(calls, " run --rm -T migrate")
    assert calls[up + 1] == f"  image-env: METANAVIT_IMAGE={OLD}"
    assert (host.home / "image.env").read_text() == f"METANAVIT_IMAGE={OLD}\n"
    assert not any("image prune" in line for line in calls)


def test_a_failed_first_deploy_has_nothing_to_go_back_to(host):
    done = host.deploy(DIGEST, FAKE_FAIL_RUN="1")
    assert done.returncode == 1 and "the migrations or the index step failed" in done.stderr
    assert not any(" up " in line for line in host.docker_calls())
    assert not (host.home / "image.env").exists()


def test_a_stack_that_never_turns_healthy_is_rolled_back_to_the_previous_image_and_fails(host):
    (host.home / "image.env").write_text(f"METANAVIT_IMAGE={OLD}\n")
    done = host.deploy(DIGEST, FAKE_FAIL_UP="1")
    assert done.returncode == 1 and "did not become healthy" in done.stderr
    calls = host.calls()
    ups = [i for i, line in enumerate(calls) if " up --detach" in line]
    assert len(ups) == 2
    assert calls[ups[1] + 1] == f"  image-env: METANAVIT_IMAGE={OLD}"  # the second up: the previous image
    assert (host.home / "image.env").read_text() == f"METANAVIT_IMAGE={OLD}\n"
    assert not any("image prune" in line for line in calls)  # the previous image is kept


def test_start_reads_the_secrets_and_starts_the_last_deployed_image(host):
    done = host.deploy("--start")
    assert done.returncode == 1 and "no image deployed yet" in done.stderr  # nothing to start before a first deploy
    (host.home / "image.env").write_text(f"METANAVIT_IMAGE={OLD}\n")
    host.log.unlink()
    done = host.deploy("--start")
    assert done.returncode == 0, done.stderr
    calls = host.calls()
    assert calls[0].startswith("aws ssm get-parameter")
    (up,) = [line for line in calls if line.startswith("docker")]
    assert " up --detach --remove-orphans" in up and "--wait" not in up and " pull " not in up


# ---- opt-in: Compose itself (COMPOSE_BIN="docker compose", or the plugin binary cloud-init installs) ---------------

COMPOSE_BIN = os.environ.get("COMPOSE_BIN", "")


@pytest.mark.skipif(not COMPOSE_BIN, reason="opt-in: set COMPOSE_BIN to `docker compose` or a docker-compose binary")
def test_compose_resolves_the_file_with_the_hosts_three_env_files(tmp_path):
    import shlex

    command = [*shlex.split(COMPOSE_BIN), "--project-directory", str(tmp_path), "-f", str(HOST_FILES / "compose.yml")]
    (tmp_path / "app.env").write_text("".join(f"{name}=x\n" for name in _app_env())
                                      + "API_MEMORY=1024m\nDB_MEMORY=320m\nREDIS_MEMORY=96m\n")
    (tmp_path / "image.env").write_text(f"METANAVIT_IMAGE=ghcr.io/joses2017smjh/metanavt@{DIGEST}\n")
    (tmp_path / "secrets.env").write_text(f"ORIGIN_VERIFY_SECRET={ORIGIN_VALUE}\nANTHROPIC_API_KEY=\nPOSTGRES_PASSWORD=pw\n")
    env_files = [arg for name in ("app.env", "image.env", "secrets.env") for arg in ("--env-file", str(tmp_path / name))]
    environ = {**os.environ, "GOMAXPROCS": os.environ.get("GOMAXPROCS", "1")}
    done = subprocess.run([*command, *env_files, "--profile", "deploy", "config", "--services"], capture_output=True,
                          text=True, env=environ, timeout=120)
    assert done.returncode == 0, done.stderr
    assert set(done.stdout.split()) == {"api", "migrate", "db", "redis"}
    # without the secrets (Parameter Store unread) the stack refuses to start rather than run without the origin check
    refused = subprocess.run([*command, *env_files[:4], "config", "--quiet"], capture_output=True, text=True,
                             env=environ, timeout=120)
    assert refused.returncode != 0 and "ORIGIN_VERIFY_SECRET" in refused.stderr


# ---- the sizing record: scripts/measure_api_memory.py measures what the host loads -----------------------------------

def test_the_memory_measurement_loads_the_models_the_host_serves():
    spec = importlib.util.spec_from_file_location("measure_api_memory", ROOT / "scripts" / "measure_api_memory.py")
    measure = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(measure)  # torch is imported only when it measures
    env = COMPOSE["x-api-environment"]
    for name in ("PUBLIC_DEMO", "ENVIRONMENT", "MODEL_PROVIDER", "EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_DIM",
                 "RERANKER_PRECISION", "RERANKER_MAX_LENGTH", "RERANK_DEPTH", "RERANKER_REQUIRED"):
        assert measure.SETTINGS[name] == env[name], name
    # the cross-encoder serve_reranker = true would load (main.tf), measured on CPU with CUDA hidden
    assert measure.SETTINGS["RERANKER_MODEL"] == "BAAI/bge-reranker-v2-m3" in _locals()["reranker_model"]
    assert measure.SETTINGS["CUDA_VISIBLE_DEVICES"] == ""
