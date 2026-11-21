# ---- Identity ----

variable "project" {
  description = "Prefix of every name, and the instance's Project tag, which the deploy role's Run Command permission requires (infra/bootstrap)."
  type        = string
  default     = "metanavit"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,20}$", var.project))
    error_message = "project: lowercase letters, digits and hyphens, starting with a letter, at most 21 characters."
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
  description = "The image the instance's first boot deploys, by digest only (sha256:<64 hex>). Later deploys go through /opt/metanavit/deploy.sh (the deploy workflow); a new value here changes the user data, which replaces the instance."
  type        = string

  validation {
    condition     = can(regex("^sha256:[0-9a-f]{64}$", var.image_digest))
    error_message = "image_digest: sha256:<64 lowercase hex>, not a tag."
  }
}

# ---- The instance ----

variable "instance_type" {
  description = "EC2 instance type, on demand. A Free plan account may launch only the Free Tier eligible types, for accounts opened on or after 2025-07-15: t3.micro, t3.small, t4g.micro, t4g.small, c7i-flex.large, m7i-flex.large (EC2 User Guide, Track your Free Tier usage, read 2026-11-21), and t8i.small, which the owner's account lists as eligible in us-west-2 (describe-instance-types with the free-tier-eligible filter, 2026-11-21). The t4g types are ARM, and the image is x86_64 only."
  type        = string
  default     = "t3.small"

  validation {
    condition     = contains(["t3.micro", "t3.small", "t4g.micro", "t4g.small", "c7i-flex.large", "m7i-flex.large", "t8i.small"], var.instance_type)
    error_message = "instance_type: a Free Tier eligible type for accounts opened on or after 2025-07-15: t3.micro, t3.small, t4g.micro, t4g.small, c7i-flex.large, m7i-flex.large, or t8i.small (listed eligible in us-west-2)."
  }

  validation {
    condition     = !startswith(var.instance_type, "t4g.")
    error_message = "instance_type: the t4g types are ARM (Graviton); the image (CI publishes linux/amd64 only) and the AMI here are x86_64."
  }
}

variable "serve_reranker" {
  description = "Serve the bge-reranker-v2-m3 cross-encoder (RERANKER_MODEL=BAAI/bge-reranker-v2-m3), or not (RERANKER_MODEL=none: the fused order of BM25 and bge-small, labelled in every response). Measured with the API's own loaders (bench/results/api_memory_cpu.json): the API's peak grows from 842 MiB to 2719 MiB with it, and one query's 20 pairs took 89.92 s on 2 CPU threads, far over the 5-second retrieval budget, so it stays off on every instance type allowed here."
  type        = bool
  default     = false
}

variable "root_volume_gb" {
  description = "Size of the encrypted gp3 root volume in GB: the OS, Docker's images (the model image twice during a deploy), the database, the model cache and the swap file."
  type        = number
  default     = 20

  validation {
    condition     = var.root_volume_gb >= 16 && floor(var.root_volume_gb) == var.root_volume_gb
    error_message = "root_volume_gb: a whole number, 16 or more."
  }
}

variable "swap_mib" {
  description = "Size of the swap file cloud-init makes, MiB: a safety margin under the memory limits below (Docker lets a container swap as much again as its limit)."
  type        = number
  default     = 2048

  validation {
    condition     = var.swap_mib >= 0 && floor(var.swap_mib) == var.swap_mib
    error_message = "swap_mib: a whole number, zero or more."
  }
}

variable "api_memory_mib" {
  description = "Memory limit of the api container, and of the one-off migrate container a deploy runs while the api is stopped, MiB (Compose mem_limit). The API's measured peak without the reranker: 842 MiB (bench/results/api_memory_cpu.json)."
  type        = number
  default     = 1024
}

variable "db_memory_mib" {
  description = "Memory limit of the Postgres container, MiB."
  type        = number
  default     = 320
}

variable "redis_memory_mib" {
  description = "Memory limit of the Redis container, MiB (Redis itself is capped at 64 MB of data)."
  type        = number
  default     = 96
}

variable "host_reserve_mib" {
  description = "Memory left to the kernel, the operating system, Docker, the SSM agent and the page cache, MiB: the three limits above must fit in the instance's memory less this."
  type        = number
  default     = 512
}

# ---- The answer route (deploy contract) ----

variable "answer_model" {
  description = "ANSWER_MODEL: claude-haiku-5-5 (the owner's choice for the live demo), claude-sonnet-5-5 or claude-opus-5-5."
  type        = string
  default     = "claude-haiku-5-5"

  validation {
    condition     = contains(["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"], var.answer_model)
    error_message = "answer_model: claude-haiku-5-5, claude-sonnet-5-5 or claude-opus-5-5."
  }
}

variable "answer_daily_budget_usd" {
  description = "ANSWER_DAILY_BUDGET_USD: the app's daily answer budget, reserved in Redis before each call (the owner's choice: about 70 USD of Claude credit over three months)."
  type        = number
  default     = 0.75

  validation {
    condition     = var.answer_daily_budget_usd > 0
    error_message = "answer_daily_budget_usd: more than zero."
  }
}

# ---- Network and edge ----

variable "vpc_cidr" {
  description = "The VPC's address range (one public /24 subnet is cut from it). Production's ECS stack uses 10.40.0.0/16 and staging 10.41.0.0/16."
  type        = string
  default     = "10.42.0.0/16"
}

variable "price_class" {
  description = "CloudFront price class (PriceClass_100: North America and Europe edges only)."
  type        = string
  default     = "PriceClass_100"
}

variable "origin_read_timeout" {
  description = "Seconds CloudFront waits for the host's response. Above 60 needs a CloudFront quota increase; the app's answer settings end an answer inside 60 (compose.yml)."
  type        = number
  default     = 60

  validation {
    condition     = var.origin_read_timeout >= 1 && var.origin_read_timeout <= 180
    error_message = "origin_read_timeout: 1 to 180 seconds."
  }
}

# ---- Alarm ----

variable "alert_email" {
  description = "Optional address for an email when the instance's status checks fail (it must confirm the SNS subscription). Null: no alarm and no topic. Never commit it."
  type        = string
  default     = null
  sensitive   = true
}
