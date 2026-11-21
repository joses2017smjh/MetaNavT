# ---- Identity ----

variable "project" {
  description = "Prefix of every name (infra/bootstrap's deploy role is scoped to names that start with it)."
  type        = string
  default     = "metanavit"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,20}$", var.project))
    error_message = "project: lowercase letters, digits and hyphens, starting with a letter, at most 21 characters."
  }
}

variable "environment" {
  description = "production or staging; part of every name."
  type        = string

  validation {
    condition     = contains(["production", "staging"], var.environment)
    error_message = "environment: production or staging."
  }
}

# ---- Image ----

variable "image_repository" {
  description = "The image's repository, lowercase, without a tag (CI's image-ml job publishes :main-ml and :sha-<commit>-ml there)."
  type        = string
  default     = "ghcr.io/joses2017smjh/metanavt"

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9._/-]*$", var.image_repository))
    error_message = "image_repository: a lowercase repository with no tag and no digest."
  }
}

variable "image_digest" {
  description = "The image to run, by digest only (sha256:<64 hex>): a tag can move, a digest cannot. The deploy workflow moves the service to later digests."
  type        = string

  validation {
    condition     = can(regex("^sha256:[0-9a-f]{64}$", var.image_digest))
    error_message = "image_digest: sha256:<64 lowercase hex>, not a tag."
  }
}

# ---- Compute ----

variable "cpu_architecture" {
  description = "Fargate CPU architecture. ARM64 is cheaper per hour but needs an arm64 build of the image (CI publishes amd64 only)."
  type        = string
  default     = "X86_64"

  validation {
    condition     = contains(["X86_64", "ARM64"], var.cpu_architecture)
    error_message = "cpu_architecture: X86_64 or ARM64."
  }
}

variable "task_cpu" {
  description = "Fargate CPU units of the API task (1024 = 1 vCPU)."
  type        = number
  default     = 1024
}

variable "task_memory" {
  description = "Fargate memory of the API task in MiB. The fp32 bge-reranker-v2-m3 alone holds about 2.3 GB."
  type        = number
  default     = 4096
}

variable "desired_count" {
  description = "API tasks. More than one needs redis_mode = \"elasticache\": a sidecar Redis is per task."
  type        = number
  default     = 1

  validation {
    condition     = var.desired_count >= 0 && floor(var.desired_count) == var.desired_count
    error_message = "desired_count: a whole number, zero or more."
  }
}

variable "health_check_grace_period_seconds" {
  description = "Seconds ECS ignores failed load balancer health checks after a task starts: every start downloads both models and loads them on CPU."
  type        = number
  default     = 900
}

variable "log_retention_days" {
  description = "CloudWatch Logs retention of the tasks' logs and of the task-stop events."
  type        = number
  default     = 14
}

# ---- Database ----

variable "db_instance_class" {
  description = "RDS instance class."
  type        = string
  default     = "db.t4g.micro"
}

variable "db_engine_version" {
  description = "PostgreSQL major version (RDS picks its current minor; pgvector is created by the app's migrations)."
  type        = string
  default     = "16"
}

variable "db_allocated_storage_gb" {
  description = "gp3 storage in GB."
  type        = number
  default     = 20
}

variable "db_backup_retention_days" {
  description = "Days of automated backups."
  type        = number
  default     = 1
}

variable "deletion_protection" {
  description = "RDS deletion protection. On in production; turn it off (and apply) first to destroy."
  type        = bool
  default     = true
}

variable "final_snapshot" {
  description = "Keep a final RDS snapshot (<project>-<environment>-final) when the database is destroyed. It stays, and is billed as snapshot storage, until deleted by hand."
  type        = bool
  default     = true
}

variable "db_password_version" {
  description = "Bump to set a new random database password: it is written to RDS and to Secrets Manager together and kept in neither the state nor the plan (doc/deploy.md, rotating keys)."
  type        = number
  default     = 1
}

# ---- Redis ----

variable "redis_mode" {
  description = "elasticache: one cache.t4g.micro node with TLS (rediss://), shared by every task. sidecar: Redis inside the task (cheaper), which holds only while there is one task, and starts empty at every task start (the day's budget counter included)."
  type        = string
  default     = "elasticache"

  validation {
    condition     = contains(["elasticache", "sidecar"], var.redis_mode)
    error_message = "redis_mode: elasticache or sidecar."
  }
}

variable "cache_engine" {
  description = "ElastiCache engine: valkey (Redis-compatible, cheaper per hour) or redis."
  type        = string
  default     = "valkey"

  validation {
    condition     = contains(["valkey", "redis"], var.cache_engine)
    error_message = "cache_engine: valkey or redis."
  }
}

variable "cache_engine_version" {
  description = "ElastiCache engine version; null takes the engine's default at creation."
  type        = string
  default     = null
}

variable "cache_node_type" {
  description = "ElastiCache node type."
  type        = string
  default     = "cache.t4g.micro"
}

# ---- The answer route (deploy contract) ----

variable "anthropic_key_stored" {
  description = "Set true once the owner has stored the Anthropic API key in the secret this module creates (doc/deploy.md). An ECS secret with no value stops the task from starting, so until then the task runs without ANTHROPIC_API_KEY: answers off, retrieval on."
  type        = bool
  default     = false
}

variable "answer_model" {
  description = "ANSWER_MODEL: claude-opus-5-5 (the app's default), claude-sonnet-5-5 or claude-haiku-5-5."
  type        = string
  default     = "claude-opus-5-5"
}

variable "answer_daily_budget_usd" {
  description = "ANSWER_DAILY_BUDGET_USD: the app's daily answer budget, reserved in Redis before each call."
  type        = number
  default     = 2
}

variable "answer_limit_per_minute" {
  description = "ANSWER_LIMIT_PER_MINUTE per caller."
  type        = number
  default     = 5
}

variable "answer_limit_per_day" {
  description = "ANSWER_LIMIT_PER_DAY per caller."
  type        = number
  default     = 50
}

variable "environment_overrides" {
  description = "Extra or replaced container environment variables (for example RERANKER_MODEL = \"none\" to run without the reranker). Never a secret, and never PUBLIC_DEMO."
  type        = map(string)
  default     = {}

  validation {
    condition     = length(setintersection(keys(var.environment_overrides), ["PUBLIC_DEMO", "ANTHROPIC_API_KEY", "PGPASSWORD"])) == 0
    error_message = "environment_overrides may not set PUBLIC_DEMO (the public surface stays on) or a secret (ANTHROPIC_API_KEY, PGPASSWORD come from Secrets Manager)."
  }
}

# ---- Edge ----

variable "vpc_cidr" {
  description = "The VPC's address range (a /16: two public and two private /24 subnets are cut from it)."
  type        = string
  default     = "10.40.0.0/16"
}

variable "domain_name" {
  description = "Optional custom domain (for example demo.example.com). Null: the distribution's own *.cloudfront.net name and certificate, no domain needed."
  type        = string
  default     = null
}

variable "route53_zone_id" {
  description = "Route 53 hosted zone of domain_name: the certificate's validation records and the alias records go there. Required with domain_name."
  type        = string
  default     = null
}

variable "price_class" {
  description = "CloudFront price class (PriceClass_100: North America and Europe edges only, the cheapest)."
  type        = string
  default     = "PriceClass_100"
}

variable "origin_read_timeout" {
  description = "Seconds CloudFront waits for the ALB's response. An answer is a CPU rerank plus a Claude call with retries; above 60 needs a CloudFront quota increase."
  type        = number
  default     = 60

  validation {
    condition     = var.origin_read_timeout >= 1 && var.origin_read_timeout <= 180
    error_message = "origin_read_timeout: 1 to 180 seconds."
  }
}

variable "alb_idle_timeout" {
  description = "ALB idle timeout in seconds (longer than origin_read_timeout)."
  type        = number
  default     = 120
}

# ---- WAF (CloudFront scope) ----

variable "enable_waf" {
  description = "A WAF web ACL on the distribution: AWS managed common rules and two per-IP rate rules."
  type        = bool
  default     = true
}

variable "waf_answer_rate_limit" {
  description = "Requests per IP per 5 minutes to /api/answer/ before the WAF blocks that IP."
  type        = number
  default     = 100
}

variable "waf_general_rate_limit" {
  description = "Requests per IP per 5 minutes to every other path before the WAF blocks that IP."
  type        = number
  default     = 1000
}

variable "waf_answer_captcha" {
  description = "A CAPTCHA on /api/answer/. Off by default: a script or the API's own clients cannot solve it, only a browser page that integrates the WAF's CAPTCHA."
  type        = bool
  default     = false
}

# ---- Alarms and tracing ----

variable "alert_email" {
  description = "Optional address subscribed to the alarm topic (it must confirm the subscription by email). Never commit it."
  type        = string
  default     = null
  sensitive   = true
}

variable "alarm_5xx_per_5_minutes" {
  description = "5xx responses (ALB plus API) in 5 minutes that raise the 5xx alarm."
  type        = number
  default     = 10
}

variable "alarm_response_time_p90_seconds" {
  description = "p90 of the API's response time, over three 5-minute periods, that raises the latency alarm."
  type        = number
  default     = 30
}

variable "alarm_rds_cpu_percent" {
  description = "RDS average CPU, over three 5-minute periods, that raises the database CPU alarm."
  type        = number
  default     = 80
}

variable "alarm_rds_free_storage_gb" {
  description = "Free RDS storage below which the storage alarm is raised."
  type        = number
  default     = 2
}

variable "alarm_redis_connections" {
  description = "Open ElastiCache connections above which the Redis alarm is raised."
  type        = number
  default     = 100
}

variable "enable_xray" {
  description = "An ADOT collector sidecar that forwards the app's OpenTelemetry spans to X-Ray. The app exports spans only from an image built with the otel extra (the CPU model image is not), so this is off by default."
  type        = bool
  default     = false
}
