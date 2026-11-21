# The API as one ECS Fargate service, and a one-off migrate/index task that the deploy workflow runs before it moves
# the service to a new image (scripts/deploy_ecs.py). Both run the same image with the same environment; the image's
# own command only serves, so the migrations and the index step are the one-off task's command.

locals {
  public_origin = local.custom_domain ? "https://${var.domain_name}" : "https://${aws_cloudfront_distribution.this.domain_name}"

  # The deploy contract (doc/engineering-reference.md, "Public demo mode"): every name it lists is set here or, for
  # the API key, in contract_secrets. tests/engine/test_infra_terraform.py compares these names with the contract.
  contract_environment = {
    PUBLIC_DEMO              = "1"
    REDIS_URL                = local.redis_url
    ANSWER_MODEL             = var.answer_model
    ANSWER_EFFORT            = "low"
    ANSWER_MAX_TOKENS        = "4096"
    ANSWER_TIMEOUT_SECONDS   = "25" # with one retry and the retrieval budget below, an answer ends inside CloudFront's
    ANSWER_MAX_RETRIES       = "1"  # origin_read_timeout (60 s): 5 + 25 x 2 plus backoff (tests/engine/test_infra_terraform.py)
    ANSWER_DAILY_BUDGET_USD  = format("%.2f", var.answer_daily_budget_usd)
    ANSWER_CACHE_TTL_SECONDS = "86400"
    ANSWER_LIMIT_PER_MINUTE  = tostring(var.answer_limit_per_minute)
    ANSWER_LIMIT_PER_DAY     = tostring(var.answer_limit_per_day)
    TRUSTED_PROXY_HOPS       = "2" # CloudFront, then the ALB, each append to X-Forwarded-For
    WEB_ORIGIN               = local.public_origin
    MAX_BODY_BYTES           = "8192"
  }

  # Referenced only once a value is stored (var.anthropic_key_stored); without it the app runs with answers off.
  contract_secrets = var.anthropic_key_stored ? { ANTHROPIC_API_KEY = aws_secretsmanager_secret.anthropic_api_key.arn } : {}

  # What the image needs beyond the contract: the database (password from Secrets Manager, TLS required), the CPU
  # models of docker-compose.ml.yml, and the index step's settings (docker-compose.yml).
  platform_environment = {
    ENVIRONMENT                = "prod"
    LOG_FORMAT                 = "json"
    PG_CONNECTION_STRING       = "postgresql://${local.db_user}@${aws_db_instance.this.address}:5432/${local.db_name}"
    PSYCOPG2_CONNECTION_STRING = "dbname=${local.db_name} user=${local.db_user} host=${aws_db_instance.this.address} port=5432"
    PGSSLMODE                  = "require" # libpq (psycopg2) and asyncpg both read it
    RETRIEVE_TIMEOUT_SECONDS   = "5"       # the demo corpus answers in milliseconds; this bounds a hung database
    MODEL_PROVIDER             = "ollama"  # the index step's init_settings() needs a provider; nothing here calls it
    EMBEDDING_PROVIDER         = "huggingface"
    EMBEDDING_MODEL            = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM              = "384"
    RERANKER_MODEL             = "BAAI/bge-reranker-v2-m3"
    RERANKER_PRECISION         = "fp32"
    RERANKER_MAX_LENGTH        = "512"
    RERANK_DEPTH               = "20"
    RERANKER_REQUIRED          = "true"
    BGE_ALLOW_DOWNLOAD         = "1" # the image carries no weights: every task start downloads both models
    RETRIEVAL_MODE             = "sql"
    DATA_DIR                   = "bench/corpus/files"
    INDEX_ON_START             = "false"
    INGEST_MAX_REJECTED        = "0"
    RETRIEVE_K                 = "50"
    RERANK_TOP_N               = "8"
    ENABLE_ROUTER              = "true"
  }

  platform_secrets = {
    PGPASSWORD = aws_secretsmanager_secret.db_password.arn
  }

  xray_environment = var.enable_xray ? {
    OTEL_EXPORTER_OTLP_ENDPOINT = "http://127.0.0.1:4318/v1/traces"
  } : {}

  app_environment = merge(local.platform_environment, local.xray_environment, local.contract_environment, var.environment_overrides)
  app_secrets     = merge(local.platform_secrets, local.contract_secrets)

  environment_list = [for name in sort(keys(local.app_environment)) : { name = name, value = local.app_environment[name] }]
  secrets_list     = [for name in sort(keys(local.app_secrets)) : { name = name, valueFrom = local.app_secrets[name] }]

  log_options = {
    "awslogs-group"  = aws_cloudwatch_log_group.ecs.name
    "awslogs-region" = local.region
  }

  api_container = {
    name         = "api"
    image        = local.image
    essential    = true
    portMappings = [{ containerPort = 8000, protocol = "tcp" }]
    environment  = local.environment_list
    secrets      = local.secrets_list
    dependsOn    = local.sidecar_redis ? [{ containerName = "redis", condition = "HEALTHY" }] : []
    # No container health check: ECS caps its start period at 300 s, shorter than a cold start (two model downloads).
    # The ALB's check of GET /health and the service's grace period decide instead.
    logConfiguration = {
      logDriver = "awslogs"
      options   = merge(local.log_options, { "awslogs-stream-prefix" = "api" })
    }
  }

  redis_container = {
    name      = "redis"
    image     = local.redis_image
    essential = true
    command   = ["redis-server", "--bind", "127.0.0.1", "--save", "", "--appendonly", "no", "--maxmemory", "64mb", "--maxmemory-policy", "volatile-lru"]
    healthCheck = {
      command     = ["CMD", "redis-cli", "ping"]
      interval    = 10
      timeout     = 5
      retries     = 3
      startPeriod = 5
    }
    logConfiguration = {
      logDriver = "awslogs"
      options   = merge(local.log_options, { "awslogs-stream-prefix" = "redis" })
    }
  }

  otel_container = {
    name      = "otel-collector"
    image     = local.adot_image
    essential = false
    command   = ["--config=/etc/ecs/ecs-xray.yaml"]
    logConfiguration = {
      logDriver = "awslogs"
      options   = merge(local.log_options, { "awslogs-stream-prefix" = "otel" })
    }
  }

  api_containers = concat(
    [local.api_container],
    local.sidecar_redis ? [local.redis_container] : [],
    var.enable_xray ? [local.otel_container] : [],
  )

  # The deploy order of docker-compose.yml: the schema migrations, then the index step. The workflow requires exit 0.
  migrate_container = {
    name        = "migrate"
    image       = local.image
    essential   = true
    command     = ["sh", "-c", "python -m app.database.migrate && python -m app.engine.ingest"]
    environment = local.environment_list
    secrets     = local.secrets_list
    logConfiguration = {
      logDriver = "awslogs"
      options   = merge(local.log_options, { "awslogs-stream-prefix" = "migrate" })
    }
  }
}

resource "aws_ecs_cluster" "this" {
  name = local.name

  setting {
    name  = "containerInsights"
    value = "disabled" # its per-metric charges; the alarms below use free metrics
  }
}

resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${local.name}"
  retention_in_days = var.log_retention_days
}

# ---- Roles: the execution role starts the task (logs, the two secrets); the task role is the app's own identity ----

data "aws_iam_policy_document" "ecs_tasks_trust" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${local.name}-execution"
  description        = "Starts the ${local.name} tasks: their log streams and their two secrets"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_trust.json
}

data "aws_iam_policy_document" "execution" {
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.ecs.arn}:*"]
  }

  statement {
    sid       = "TheTwoSecrets"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.db_password.arn, aws_secretsmanager_secret.anthropic_api_key.arn]
  }
}

resource "aws_iam_role_policy" "execution" {
  name   = "start-tasks"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.execution.json
}

# The app calls no AWS API, so its role has no permission unless the X-Ray sidecar runs.
resource "aws_iam_role" "task" {
  name               = "${local.name}-task"
  description        = "The ${local.name} app's identity inside its tasks"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_trust.json
}

data "aws_iam_policy_document" "xray" {
  statement {
    actions   = ["xray:PutTraceSegments", "xray:PutTelemetryRecords", "xray:GetSamplingRules", "xray:GetSamplingTargets"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "task_xray" {
  count = var.enable_xray ? 1 : 0

  name   = "xray"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.xray.json
}

# ---- Task definitions: Terraform registers the first revision of each; the deploy workflow registers later ones ----

resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = tostring(var.task_cpu)
  memory                   = tostring(var.task_memory)
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  container_definitions    = jsonencode(local.api_containers)

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = var.cpu_architecture
  }
}

resource "aws_ecs_task_definition" "migrate" {
  family                   = "${local.name}-migrate"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = tostring(var.task_cpu)
  memory                   = tostring(var.task_memory)
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  container_definitions    = jsonencode([local.migrate_container])

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = var.cpu_architecture
  }
}

resource "aws_ecs_service" "api" {
  name             = "${local.name}-api"
  cluster          = aws_ecs_cluster.this.id
  task_definition  = aws_ecs_task_definition.api.arn
  desired_count    = var.desired_count
  launch_type      = "FARGATE"
  platform_version = "LATEST"

  health_check_grace_period_seconds  = var.health_check_grace_period_seconds
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200
  enable_execute_command             = false

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.tasks.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener_rule.from_cloudfront]

  lifecycle {
    # The deploy workflow moves the service to a revision with a new image digest; a later apply leaves it there
    # (a changed task definition is rolled out by the next deploy, which starts from the newest revision).
    ignore_changes = [task_definition]

    precondition {
      condition     = !local.sidecar_redis || var.desired_count <= 1
      error_message = "redis_mode = \"sidecar\" holds only with one task (each task would have its own Redis): use \"elasticache\" for more."
    }
  }
}
