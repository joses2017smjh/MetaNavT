"""The AWS stack under infra/ (Terraform), checked from the files: nothing here calls AWS or needs credentials.

What a reviewer of the public demo's deployment would ask first, each pinned to the code that answers it: only the
deploy role writes, and only from the protected GitHub environments; RDS and Redis are private; the ALB admits only
CloudFront; images run by digest; no secret value is in the code; the task sets every name of the deploy contract in
doc/engineering-reference.md. infra/ is read as text (tests/engine/infra_hcl.py); `terraform validate` itself runs in
.github/workflows/infra.yml, and here only when TERRAFORM_BIN names a terraform binary (opt-in: it needs the providers).
"""

import importlib.util
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INFRA = ROOT / "infra"
_spec = importlib.util.spec_from_file_location("infra_hcl", Path(__file__).with_name("infra_hcl.py"))
hcl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hcl)

BOOTSTRAP = hcl.read_dir(INFRA / "bootstrap")
MODULE = hcl.read_dir(INFRA / "modules" / "demo")
ROOTS = ["bootstrap", "envs/production", "envs/staging"]  # every root module (each has its own state and lock file)
PINNED = re.compile(r"[a-z0-9][a-z0-9._/-]*:[\w.-]+@sha256:[0-9a-f]{64}")  # as tests/eval/test_image_build.py


def _resources(text: str, kind: str) -> dict[str, str]:
    return {labels[1]: body for labels, body in hcl.blocks(text, "resource", kind)}


def _locals(text: str) -> dict[str, str]:
    """Every local value's expression, across all locals blocks."""
    values = {}
    for _, body in hcl.blocks(text, "locals"):
        for name in hcl.object_keys("{" + body + "}"):
            values[name] = hcl.attr(body, name)
    return values


def _actions(policy_body: str) -> set[str]:
    return {action for statement in hcl.nested(policy_body, "statement")
            for action in hcl.strings(hcl.attr(statement, "actions") or "")}


def _contract() -> dict[str, str]:
    """NAME -> value of the deploy contract block, read as tests/api/test_public_demo.py reads it."""
    text = (ROOT / "doc" / "engineering-reference.md").read_text()
    section = text.split("### Public demo mode", 1)[1].split("**Full application locally**", 1)[0]
    block = section.split("```text", 1)[1].split("```", 1)[0]
    return dict(re.findall(r"(?:^|\s)([A-Z][A-Z_]+)=(\S+)", block, flags=re.M))


# ---- only the deploy role writes, and only from the two environments -------------------------------------------

def test_github_reaches_aws_through_two_roles_each_bound_to_this_repository():
    roles = _resources(BOOTSTRAP, "aws_iam_role")
    assert set(roles) == {"plan", "deploy"}
    trusts = {labels[1]: body for labels, body in hcl.blocks(BOOTSTRAP, "data", "aws_iam_policy_document")
              if labels[1].endswith("_trust")}
    assert set(trusts) == {"plan_trust", "deploy_trust"}
    for name, trust in trusts.items():
        (statement,) = hcl.nested(trust, "statement")
        assert hcl.strings(hcl.attr(statement, "actions")) == ["sts:AssumeRoleWithWebIdentity"]
        (principal,) = hcl.nested(statement, "principals")
        assert hcl.attr(principal, "identifiers") == "[aws_iam_openid_connect_provider.github.arn]"
        conditions = {hcl.strings(hcl.attr(c, "variable"))[0]: (hcl.strings(hcl.attr(c, "test"))[0], hcl.attr(c, "values"))
                      for c in hcl.nested(statement, "condition")}
        assert conditions == {"${local.oidc_host}:aud": ("StringEquals", '["sts.amazonaws.com"]'),
                              "${local.oidc_host}:sub": ("StringEquals", f"local.{name.split('_')[0]}_subjects")}
    assert hcl.strings(_locals(BOOTSTRAP)["oidc_host"]) == ["token.actions.githubusercontent.com"]
    # The only GitHub federation in infra/ is here: the environments' roles are ECS task roles.
    assert "token.actions.githubusercontent.com" not in MODULE


def test_the_plan_role_is_for_pull_requests_and_cannot_write():
    assert hcl.strings(_locals(BOOTSTRAP)["plan_subjects"]) == ["repo:${var.github_repository}:pull_request"]
    attachments = _resources(BOOTSTRAP, "aws_iam_role_policy_attachment")
    assert {name: hcl.attr(body, "role") for name, body in attachments.items()} == {"plan_read_only": "aws_iam_role.plan.name"}
    assert hcl.strings(hcl.attr(attachments["plan_read_only"], "policy_arn")) == ["arn:${local.partition}:iam::aws:policy/ReadOnlyAccess"]
    inline = _resources(BOOTSTRAP, "aws_iam_role_policy")
    assert {name: hcl.attr(body, "role") for name, body in inline.items()} == {"deploy": "aws_iam_role.deploy.id"}


def test_the_deploy_role_is_assumable_only_from_the_protected_environments_production_by_default():
    subjects = _locals(BOOTSTRAP)["deploy_subjects"]
    assert subjects == '[for env in var.deploy_environments : "repo:${var.github_repository}:environment:${env}"]'
    variable = hcl.block(BOOTSTRAP, "variable", "deploy_environments")
    assert hcl.attr(variable, "default") == '["production"]'  # staging is opt-in (doc/deploy.md)
    (validation,) = hcl.nested(variable, "validation")
    assert 'contains(["production", "staging"], env)' in hcl.attr(validation, "condition")
    repository = hcl.block(BOOTSTRAP, "variable", "github_repository")
    assert hcl.strings(hcl.attr(repository, "default")) == ["joses2017smjh/MetaNavT"]  # GitHub's spelling: sub is case-sensitive


def test_the_deploy_role_has_only_the_deploy_steps_permissions():
    policy = hcl.block(BOOTSTRAP, "data", "aws_iam_policy_document", "deploy")
    assert _actions(policy) == {
        "ecs:DescribeTaskDefinition", "ecs:RegisterTaskDefinition", "ecs:DescribeServices", "ecs:UpdateService",
        "ecs:RunTask", "ecs:DescribeTasks", "ecs:StopTask", "iam:PassRole", "secretsmanager:DescribeSecret",
        "s3:PutObject", "s3:ListBucket", "cloudfront:CreateInvalidation", "cloudfront:GetInvalidation",
    }
    statements = {hcl.strings(hcl.attr(s, "sid"))[0]: s for s in hcl.nested(policy, "statement")}
    assert all(hcl.attr(s, "effect") is None for s in statements.values())  # Allow only, no Deny to reason about
    # The only "*" resource: the two task definition calls, which take no resource-level permission.
    star = [sid for sid, s in statements.items() if hcl.attr(s, "resources") == '["*"]']
    assert star == ["TaskDefinitions"]
    assert hcl.strings(hcl.attr(statements["RunTheMigrateTask"], "resources")) == ["${local.ecs_arn}:task-definition/${var.project}-*-migrate:*"]
    passed = statements["PassTheTwoTaskRoles"]
    assert hcl.strings(hcl.attr(passed, "resources")) == [
        "arn:${local.partition}:iam::${local.account_id}:role/${var.project}-*-task",
        "arn:${local.partition}:iam::${local.account_id}:role/${var.project}-*-execution"]
    (condition,) = hcl.nested(passed, "condition")
    assert hcl.strings(hcl.attr(condition, "values")) == ["ecs-tasks.amazonaws.com"]
    # Secret metadata only: the values are read by the tasks' execution role, never by the workflow.
    assert "secretsmanager:GetSecretValue" not in BOOTSTRAP
    # The module names what these patterns match: <project>-<environment>-task / -execution / -migrate / -web-<account>.
    roles = _resources(MODULE, "aws_iam_role")
    assert {n: hcl.strings(hcl.attr(b, "name"))[0] for n, b in roles.items()} == {
        "execution": "${local.name}-execution", "task": "${local.name}-task"}
    families = {n: hcl.strings(hcl.attr(b, "family"))[0] for n, b in _resources(MODULE, "aws_ecs_task_definition").items()}
    assert families == {"api": "${local.name}-api", "migrate": "${local.name}-migrate"}
    assert hcl.strings(hcl.attr(_resources(MODULE, "aws_s3_bucket")["web"], "bucket")) == ["${local.name}-web-${local.account_id}"]


# ---- RDS and Redis are not public ----------------------------------------------------------------------------

def test_rds_is_private_encrypted_and_admits_only_the_tasks():
    db = _resources(MODULE, "aws_db_instance")["this"]
    assert hcl.attr(db, "publicly_accessible") == "false"
    assert hcl.attr(db, "storage_encrypted") == "true"
    assert hcl.attr(db, "db_subnet_group_name") == "aws_db_subnet_group.this.name"
    assert hcl.attr(_resources(MODULE, "aws_db_subnet_group")["this"], "subnet_ids") == "aws_subnet.private[*].id"
    assert hcl.attr(db, "vpc_security_group_ids") == "[aws_security_group.db.id]"
    assert hcl.attr(db, "password") is None and hcl.attr(db, "password_wo") == "ephemeral.random_password.db.result"
    production = hcl.read_dir(INFRA / "envs" / "production")
    assert hcl.attr(hcl.block(production, "module", "demo"), "deletion_protection") == "true"


def test_redis_is_private_with_tls_and_admits_only_the_tasks():
    cache = _resources(MODULE, "aws_elasticache_replication_group")["this"]
    assert hcl.attr(cache, "transit_encryption_enabled") == "true"
    assert hcl.attr(cache, "at_rest_encryption_enabled") == "true"
    assert hcl.attr(cache, "subnet_group_name") == "aws_elasticache_subnet_group.this[0].name"
    assert hcl.attr(_resources(MODULE, "aws_elasticache_subnet_group")["this"], "subnet_ids") == "aws_subnet.private[*].id"
    assert hcl.attr(cache, "engine") == "var.cache_engine"
    assert hcl.strings(hcl.attr(hcl.block(MODULE, "variable", "cache_engine"), "default")) == ["valkey"]
    # TLS in transit means the app's URL is rediss:// (the sidecar listens on the task's loopback only).
    assert hcl.strings(_locals(MODULE)["redis_url"]) == [
        "rediss://${aws_elasticache_replication_group.this[0].primary_endpoint_address}:6379/0", "redis://127.0.0.1:6379/0"]
    sidecar = _locals(MODULE)["redis_container"]
    assert '"--bind", "127.0.0.1"' in sidecar


def test_the_private_subnets_have_no_route_out_and_there_is_no_nat_gateway():
    routes = _resources(MODULE, "aws_route")
    assert set(routes) == {"public_internet"}
    assert hcl.attr(routes["public_internet"], "route_table_id") == "aws_route_table.public.id"
    assert hcl.attr(routes["public_internet"], "gateway_id") == "aws_internet_gateway.this.id"
    for table in _resources(MODULE, "aws_route_table").values():
        assert not hcl.nested(table, "route")  # no inline routes
    associations = _resources(MODULE, "aws_route_table_association")
    assert hcl.attr(associations["private"], "route_table_id") == "aws_route_table.private.id"
    assert hcl.attr(associations["private"], "subnet_id") == "aws_subnet.private[count.index].id"
    assert "aws_nat_gateway" not in MODULE and "nat_gateway_id" not in MODULE
    for subnet in _resources(MODULE, "aws_subnet").values():
        assert hcl.attr(subnet, "map_public_ip_on_launch") == "false"


def _ingress(security_group: str) -> list[str]:
    return [body for body in _resources(MODULE, "aws_vpc_security_group_ingress_rule").values()
            if hcl.attr(body, "security_group_id") == f"aws_security_group.{security_group}.id"]


def test_every_security_group_rule_is_its_own_resource():
    for body in _resources(MODULE, "aws_security_group").values():
        assert not hcl.nested(body, "ingress") and not hcl.nested(body, "egress")
    assert "aws_security_group_rule" not in MODULE


def test_the_database_and_the_cache_admit_only_the_tasks():
    (db,) = _ingress("db")
    assert hcl.attr(db, "referenced_security_group_id") == "aws_security_group.tasks.id"
    assert (hcl.attr(db, "from_port"), hcl.attr(db, "to_port")) == ("5432", "5432")
    cache = [body for body in _resources(MODULE, "aws_vpc_security_group_ingress_rule").values()
             if hcl.attr(body, "security_group_id") == "aws_security_group.cache[0].id"]
    assert len(cache) == 1 and hcl.attr(cache[0], "referenced_security_group_id") == "aws_security_group.tasks.id"
    assert (hcl.attr(cache[0], "from_port"), hcl.attr(cache[0], "to_port")) == ("6379", "6379")
    for body in _resources(MODULE, "aws_vpc_security_group_ingress_rule").values():
        if "aws_security_group.alb" not in hcl.attr(body, "security_group_id"):
            assert hcl.attr(body, "cidr_ipv4") is None and hcl.attr(body, "cidr_ipv6") is None


# ---- the ALB admits only CloudFront --------------------------------------------------------------------------

def test_the_alb_admits_only_cloudfronts_origin_facing_addresses():
    (rule,) = _ingress("alb")  # one ingress rule on the ALB's group
    assert hcl.attr(rule, "prefix_list_id") == "data.aws_ec2_managed_prefix_list.cloudfront.id"
    assert hcl.attr(rule, "cidr_ipv4") is None and hcl.attr(rule, "cidr_ipv6") is None
    assert (hcl.attr(rule, "from_port"), hcl.attr(rule, "to_port")) == ("80", "80")
    prefix_list = hcl.block(MODULE, "data", "aws_ec2_managed_prefix_list", "cloudfront")
    assert hcl.strings(hcl.attr(prefix_list, "name")) == ["com.amazonaws.global.cloudfront.origin-facing"]
    # the tasks take the API port from the ALB only, and are not internal-load-balancer-less: they sit behind it
    (tasks,) = _ingress("tasks")
    assert hcl.attr(tasks, "referenced_security_group_id") == "aws_security_group.alb.id"
    assert (hcl.attr(tasks, "from_port"), hcl.attr(tasks, "to_port")) == ("8000", "8000")


def test_the_listener_forwards_only_requests_carrying_cloudfronts_header():
    listeners = _resources(MODULE, "aws_lb_listener")
    assert set(listeners) == {"http"} and hcl.attr(listeners["http"], "port") == "80"
    (default,) = hcl.nested(listeners["http"], "default_action")
    assert hcl.strings(hcl.attr(default, "type")) == ["fixed-response"]
    (fixed,) = hcl.nested(default, "fixed_response")
    assert hcl.strings(hcl.attr(fixed, "status_code")) == ["403"]
    rules = _resources(MODULE, "aws_lb_listener_rule")
    assert set(rules) == {"from_cloudfront"}
    (condition,) = hcl.nested(rules["from_cloudfront"], "condition")
    (header,) = hcl.nested(condition, "http_header")
    assert hcl.attr(header, "http_header_name") == "local.origin_header_name"
    assert hcl.attr(header, "values") == "[random_password.origin_header.result]"
    distribution = _resources(MODULE, "aws_cloudfront_distribution")["this"]
    (api_origin,) = [o for o in hcl.nested(distribution, "origin") if hcl.strings(hcl.attr(o, "origin_id")) == ["api"]]
    (custom_header,) = hcl.nested(api_origin, "custom_header")
    assert (hcl.attr(custom_header, "name"), hcl.attr(custom_header, "value")) == (
        "local.origin_header_name", "random_password.origin_header.result")
    assert hcl.attr(api_origin, "domain_name") == "aws_lb.this.dns_name"


def test_cloudfront_sends_only_the_api_and_health_to_the_alb_uncached():
    assert hcl.strings(_locals(MODULE)["api_paths"]) == ["/api/*", "/health"]
    distribution = _resources(MODULE, "aws_cloudfront_distribution")["this"]
    (dynamic,) = [body for labels, body in hcl.blocks(distribution, "dynamic", "ordered_cache_behavior")]
    (content,) = hcl.nested(dynamic, "content")
    assert hcl.strings(hcl.attr(content, "target_origin_id")) == ["api"]
    assert hcl.attr(content, "cache_policy_id") == "data.aws_cloudfront_cache_policy.disabled.id"
    assert hcl.attr(content, "origin_request_policy_id") == "data.aws_cloudfront_origin_request_policy.all_viewer_except_host.id"
    policies = {labels[1]: hcl.strings(hcl.attr(body, "name"))[0] for labels, body in hcl.blocks(MODULE, "data")
                if labels[0] in ("aws_cloudfront_cache_policy", "aws_cloudfront_origin_request_policy")}
    assert policies == {"disabled": "Managed-CachingDisabled", "optimized": "Managed-CachingOptimized",
                        "all_viewer_except_host": "Managed-AllViewerExceptHostHeader"}
    (default,) = hcl.nested(distribution, "default_cache_behavior")
    assert hcl.strings(hcl.attr(default, "target_origin_id")) == ["web"]
    (web_origin,) = [o for o in hcl.nested(distribution, "origin") if hcl.strings(hcl.attr(o, "origin_id")) == ["web"]]
    assert hcl.attr(web_origin, "origin_access_control_id") == "aws_cloudfront_origin_access_control.web.id"


# ---- images by digest ----------------------------------------------------------------------------------------

def test_every_container_image_is_pinned_by_digest():
    variable = hcl.block(MODULE, "variable", "image_digest")
    assert hcl.attr(variable, "default") is None  # no default: always given
    (validation,) = hcl.nested(variable, "validation")
    assert hcl.attr(validation, "condition") == 'can(regex("^sha256:[0-9a-f]{64}$", var.image_digest))'
    values = _locals(MODULE)
    assert hcl.strings(values["image"]) == ["${var.image_repository}@${var.image_digest}"]
    containers = {name: values[name] for name in ("api_container", "migrate_container", "redis_container", "otel_container")}
    images = {name: hcl.attr(body.strip()[1:-1], "image") for name, body in containers.items()}
    assert images == {"api_container": "local.image", "migrate_container": "local.image",
                      "redis_container": "local.redis_image", "otel_container": "local.adot_image"}
    for sidecar in ("redis_image", "adot_image"):
        (reference,) = hcl.strings(values[sidecar])
        assert PINNED.fullmatch(reference), reference
    assert hcl.strings(hcl.attr(hcl.block(MODULE, "variable", "image_repository"), "default")) == ["ghcr.io/joses2017smjh/metanavt"]
    for env in ("production", "staging"):
        module = hcl.block(hcl.read_dir(INFRA / "envs" / env), "module", "demo")
        assert hcl.attr(module, "image_digest") == "var.image_digest"


# ---- the task's environment is the deploy contract's ---------------------------------------------------------

# The names the task sets beyond the contract: the database, the CPU models of docker-compose.ml.yml and the index
# step's settings. A change here is a deliberate change to what the deployed image runs with.
PLATFORM = {"ENVIRONMENT", "LOG_FORMAT", "PG_CONNECTION_STRING", "PSYCOPG2_CONNECTION_STRING", "PGSSLMODE",
            "MODEL_PROVIDER", "EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_DIM", "RERANKER_MODEL",
            "RERANKER_PRECISION", "RERANKER_MAX_LENGTH", "RERANK_DEPTH", "RERANKER_REQUIRED", "BGE_ALLOW_DOWNLOAD",
            "RETRIEVAL_MODE", "DATA_DIR", "INDEX_ON_START", "INGEST_MAX_REJECTED", "RETRIEVE_K", "RERANK_TOP_N",
            "ENABLE_ROUTER", "RETRIEVE_TIMEOUT_SECONDS"}


def test_the_task_sets_every_name_of_the_deploy_contract_and_only_listed_extras():
    contract = _contract()
    values = _locals(MODULE)
    environment = hcl.object_keys(values["contract_environment"])
    secrets = re.findall(r"\{\s*([A-Z_]+)\s*=", values["contract_secrets"])
    assert sorted(environment + secrets) == sorted(contract)  # the contract, exactly, between the two
    assert secrets == ["ANTHROPIC_API_KEY"]
    assert set(hcl.object_keys(values["platform_environment"])) == PLATFORM
    assert hcl.object_keys(values["platform_secrets"]) == ["PGPASSWORD"]
    assert not set(PLATFORM) & set(contract)
    # Both task definitions get the one merged list, secrets from Secrets Manager by ARN (valueFrom).
    assert values["app_environment"] == ("merge(local.platform_environment, local.xray_environment, local.contract_environment, "
                                         "var.environment_overrides)")
    assert values["app_secrets"] == "merge(local.platform_secrets, local.contract_secrets)"
    assert "valueFrom = local.app_secrets[name]" in values["secrets_list"]
    for container in ("api_container", "migrate_container"):
        body = values[container].strip()[1:-1]
        assert (hcl.attr(body, "environment"), hcl.attr(body, "secrets")) == ("local.environment_list", "local.secrets_list")


def test_the_contracts_fixed_values_are_the_ones_the_task_sets():
    contract, values = _contract(), _locals(MODULE)
    body = values["contract_environment"].strip()[1:-1]
    literal = {name: hcl.strings(hcl.attr(body, name))[0] for name in hcl.object_keys(values["contract_environment"])
               if hcl.attr(body, name).startswith('"')}
    for name in ("PUBLIC_DEMO", "ANSWER_EFFORT", "ANSWER_MAX_TOKENS", "ANSWER_CACHE_TTL_SECONDS", "MAX_BODY_BYTES"):
        assert literal[name] == contract[name], name
    # The answer's time bounds are the AWS task's own: CloudFront gives the origin origin_read_timeout seconds, and an
    # answer is the retrieval budget plus every attempt of the Claude call (the SDK retries a timeout), so the worst
    # case must end before CloudFront gives up, or the visitor gets a 504 instead of the route's own reason.
    platform = values["platform_environment"].strip()[1:-1]
    retrieve = float(hcl.strings(hcl.attr(platform, "RETRIEVE_TIMEOUT_SECONDS"))[0])
    timeout, retries = float(literal["ANSWER_TIMEOUT_SECONDS"]), int(literal["ANSWER_MAX_RETRIES"])
    origin = float(hcl.attr(hcl.block(MODULE, "variable", "origin_read_timeout"), "default"))
    assert retrieve + timeout * (retries + 1) + 5 <= origin, (retrieve, timeout, retries, origin)
    defaults = {name: hcl.attr(hcl.block(MODULE, "variable", name), "default") for name in
                ("answer_model", "answer_daily_budget_usd", "answer_limit_per_minute", "answer_limit_per_day")}
    assert hcl.strings(defaults["answer_model"]) == [contract["ANSWER_MODEL"]]
    assert float(defaults["answer_daily_budget_usd"]) == float(contract["ANSWER_DAILY_BUDGET_USD"])
    assert defaults["answer_limit_per_minute"] == contract["ANSWER_LIMIT_PER_MINUTE"]
    assert defaults["answer_limit_per_day"] == contract["ANSWER_LIMIT_PER_DAY"]
    # Behind CloudFront and the ALB, two proxies append to X-Forwarded-For (the contract's note says so).
    assert literal["TRUSTED_PROXY_HOPS"] == "2"
    override = hcl.block(MODULE, "variable", "environment_overrides")
    (validation,) = hcl.nested(override, "validation")
    assert '["PUBLIC_DEMO", "ANTHROPIC_API_KEY", "PGPASSWORD"]' in hcl.attr(validation, "condition")


def test_the_api_listens_on_8000_and_is_health_checked_on_get_health():
    values = _locals(MODULE)
    assert hcl.attr(values["api_container"].strip()[1:-1], "portMappings") == '[{ containerPort = 8000, protocol = "tcp" }]'
    group = _resources(MODULE, "aws_lb_target_group")["api"]
    assert hcl.attr(group, "port") == "8000"
    (check,) = hcl.nested(group, "health_check")
    assert hcl.strings(hcl.attr(check, "path")) == ["/health"]
    service = _resources(MODULE, "aws_ecs_service")["api"]
    (balancer,) = hcl.nested(service, "load_balancer")
    assert hcl.attr(balancer, "container_port") == "8000"
    (lifecycle,) = hcl.nested(service, "lifecycle")
    assert hcl.attr(lifecycle, "ignore_changes") == "[task_definition]"  # the deploy workflow owns the image


def test_the_migrate_task_runs_the_migrations_then_the_index_step():
    command = hcl.attr(_locals(MODULE)["migrate_container"].strip()[1:-1], "command")
    assert hcl.strings(command) == ["sh", "-c", "python -m app.database.migrate && python -m app.engine.ingest"]


# ---- no secret values in the code ------------------------------------------------------------------------------

SECRET_PATTERNS = {
    "AWS access key id": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "Anthropic API key": re.compile(r"sk-ant-[\w-]{8,}"),
    "GitHub token": re.compile(r"\b(?:ghp|gho|ghs|ghu|github_pat)_[A-Za-z0-9_]{20,}"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "AWS account id": re.compile(r"(?<![\w.:-])\d{12}(?![\w])"),
    "email address": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
}


def _deploy_files() -> list[Path]:
    files = [p for p in INFRA.rglob("*") if p.is_file() and ".terraform" not in p.parts]
    files += [ROOT / ".github" / "workflows" / name for name in ("infra.yml", "deploy.yml")]
    return [p for p in files if p.exists()]


def test_no_secret_value_account_id_or_email_is_in_the_code():
    problems = []
    for path in _deploy_files():
        text = path.read_text()
        for label, pattern in SECRET_PATTERNS.items():
            problems += [f"{path.relative_to(ROOT)}: {label}: {m.group(0)}" for m in pattern.finditer(text)]
    assert problems == []


def test_no_state_plan_or_variable_file_is_in_the_tree():
    found = [p.relative_to(ROOT).as_posix() for p in INFRA.rglob("*") if ".terraform" not in p.parts and p.is_file()
             and (p.suffix in {".tfstate", ".tfplan", ".tfvars"} or p.name.endswith((".tfstate.backup", ".tfvars.json")))]
    assert found == []
    ignored = (INFRA / ".gitignore").read_text().split()
    assert {"**/.terraform/", "*.tfstate", "*.tfvars"} <= set(ignored)


def test_secrets_reach_the_task_from_secrets_manager_and_terraform_never_holds_their_values():
    versions = _resources(MODULE, "aws_secretsmanager_secret_version")
    assert set(versions) == {"db_password"}  # the Anthropic key's value is put in by the owner, outside Terraform
    assert hcl.attr(versions["db_password"], "secret_string") is None
    assert hcl.attr(versions["db_password"], "secret_string_wo") == "ephemeral.random_password.db.result"
    assert hcl.blocks(MODULE, "ephemeral", "random_password", "db")  # generated per apply, kept in neither state nor plan
    assert not _resources(MODULE, "random_password").keys() - {"origin_header"}
    assert set(_resources(MODULE, "aws_secretsmanager_secret")) == {"db_password", "anthropic_api_key"}
    policy = hcl.block(MODULE, "data", "aws_iam_policy_document", "execution")
    secrets = [s for s in hcl.nested(policy, "statement") if "secretsmanager:GetSecretValue" in hcl.strings(hcl.attr(s, "actions"))]
    assert len(secrets) == 1 and hcl.attr(secrets[0], "resources") == (
        "[aws_secretsmanager_secret.db_password.arn, aws_secretsmanager_secret.anthropic_api_key.arn]")
    # the Anthropic key is referenced only once the owner says it is stored (an empty secret stops the task's start)
    assert _locals(MODULE)["contract_secrets"].startswith("var.anthropic_key_stored ?")
    for env in ("production", "staging"):
        assert hcl.attr(hcl.block(hcl.read_dir(INFRA / "envs" / env), "module", "demo"), "anthropic_key_stored") == "false"


# ---- environments: production on, staging opt-in and smaller -----------------------------------------------------

def test_production_and_staging_call_the_one_module_with_their_differences():
    settings = {}
    for env in ("production", "staging"):
        text = hcl.read_dir(INFRA / "envs" / env)
        module = hcl.block(text, "module", "demo")
        assert hcl.strings(hcl.attr(module, "source")) == ["../../modules/demo"]
        assert hcl.strings(hcl.attr(module, "environment")) == [env]
        backend = hcl.block(hcl.block(text, "terraform"), "backend", "s3")
        assert hcl.strings(hcl.attr(backend, "key")) == [f"{env}/terraform.tfstate"]
        assert hcl.attr(backend, "use_lockfile") == "true" and hcl.attr(backend, "bucket") is None  # given at init
        assert hcl.attr(hcl.block(text, "output", "image_digest"), "value") == "var.image_digest"  # the plan job reads it
        settings[env] = {name: hcl.attr(module, name)
                         for name in ("redis_mode", "enable_waf", "deletion_protection", "final_snapshot")}
    assert settings == {  # None: the module's default
        "production": {"redis_mode": '"elasticache"', "enable_waf": "true", "deletion_protection": "true", "final_snapshot": None},
        "staging": {"redis_mode": '"sidecar"', "enable_waf": "false", "deletion_protection": "false", "final_snapshot": "false"},
    }
    assert hcl.attr(hcl.block(MODULE, "variable", "final_snapshot"), "default") == "true"
    # turning deletion protection off to destroy production does not also skip its final snapshot
    db = _resources(MODULE, "aws_db_instance")["this"]
    assert hcl.attr(db, "skip_final_snapshot") == "!var.final_snapshot"


def test_every_root_requires_terraform_1_11_and_commits_its_provider_lock():
    for directory in [*ROOTS, "modules/demo"]:
        terraform = hcl.block(hcl.read_dir(INFRA / directory), "terraform")
        assert hcl.strings(hcl.attr(terraform, "required_version")) == [">= 1.11.0"], directory
    for directory in ROOTS:
        lock = (INFRA / directory / ".terraform.lock.hcl").read_text()
        providers = dict(re.findall(r'provider "registry\.terraform\.io/hashicorp/(\w+)" \{\s*version\s*=\s*"([\d.]+)"', lock))
        expected = {"aws"} if directory == "bootstrap" else {"aws", "random"}
        assert set(providers) == expected, directory
        assert providers["aws"].startswith("6.")
        for block_text in lock.split('provider "')[1:]:
            assert re.search(r'"h1:[A-Za-z0-9+/=]+"', block_text) and len(re.findall(r'"zh:[0-9a-f]{64}"', block_text)) >= 10


# ---- opt-in: terraform itself (TERRAFORM_BIN=/path/to/terraform; providers from TF_PLUGIN_CACHE_DIR or the registry)

TERRAFORM = os.environ.get("TERRAFORM_BIN", "")


@pytest.mark.skipif(not TERRAFORM, reason="opt-in: set TERRAFORM_BIN to a terraform >= 1.11 binary "
                    "(.github/workflows/infra.yml runs fmt and validate on every infra change)")
@pytest.mark.parametrize("directory", ROOTS)
def test_terraform_validates_offline(directory, tmp_path):
    if not shutil.which(TERRAFORM):
        pytest.skip(f"TERRAFORM_BIN={TERRAFORM} is not an executable")
    # No backend, no credentials, and an unreachable AWS endpoint: validate never calls AWS; the guard makes sure.
    env = {key: value for key, value in os.environ.items() if not key.startswith("AWS_")}
    env.update(TF_DATA_DIR=str(tmp_path / ".terraform"), TF_IN_AUTOMATION="1", CHECKPOINT_DISABLE="1",
               AWS_EC2_METADATA_DISABLED="true", AWS_ENDPOINT_URL="http://127.0.0.1:9")
    workdir = INFRA / directory
    init = subprocess.run([TERRAFORM, f"-chdir={workdir}", "init", "-backend=false", "-input=false", "-lockfile=readonly"],
                          env=env, capture_output=True, text=True, timeout=600)
    assert init.returncode == 0, init.stdout + init.stderr
    validate = subprocess.run([TERRAFORM, f"-chdir={workdir}", "validate", "-no-color"], env=env, capture_output=True,
                              text=True, timeout=300)
    assert validate.returncode == 0, validate.stdout + validate.stderr


@pytest.mark.skipif(not TERRAFORM, reason="opt-in: set TERRAFORM_BIN to a terraform binary")
def test_terraform_fmt_is_clean():
    done = subprocess.run([TERRAFORM, "fmt", "-check", "-recursive", "-diff", str(INFRA)], capture_output=True, text=True,
                          env={**os.environ, "CHECKPOINT_DISABLE": "1"}, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
