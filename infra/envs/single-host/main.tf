# The live demo on one host sized to the AWS Free plan (doc/deploy.md, "The live demo: one host on the AWS Free plan"):
# what .github/workflows/deploy-single-host.yml deploys to when the repository variable DEPLOY_TARGET is single-host.
# Settings live here, in code, so a pull request shows every change; only the first image digest and the alarm email
# come from the command line. infra/envs/production (ECS) is the other target, never applied beside this one.
terraform {
  required_version = ">= 1.11.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.73"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.9"
    }
  }

  # State in infra/bootstrap's bucket, with S3's own lockfile (no DynamoDB table). The bucket name holds the account
  # id, so it is given at init: terraform init -backend-config="bucket=<bootstrap output state_bucket>".
  backend "s3" {
    key          = "single-host/terraform.tfstate"
    region       = "us-west-2"
    encrypt      = true
    use_lockfile = true
  }
}

variable "region" {
  description = "Region of the host (the backend block above names the state bucket's region)."
  type        = string
  default     = "us-west-2"
}

variable "image_digest" {
  description = "sha256:<64 hex> of the CPU model image the instance's first boot deploys (doc/deploy.md). A later apply passes the last apply's digest back (terraform output image_digest): a new one replaces the instance."
  type        = string
}

variable "alert_email" {
  description = "Optional address for the status-check alarm. Never commit it."
  type        = string
  default     = null
  sensitive   = true
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project     = "metanavit"
      Environment = "production"
      Target      = "single-host"
      ManagedBy   = "terraform"
    }
  }
}

module "single_host" {
  source = "../../modules/single-host"

  image_digest = var.image_digest
  alert_email  = var.alert_email

  # The owner's decision (2026-11-21): the cheapest Free Tier eligible type that holds the stack, with the reranker
  # off (its fp32 weights do not fit in 2 GiB beside Postgres): the fused order is served, labelled as such, and the
  # README's benchmark tables show what the reranker adds. doc/deploy.md has the monthly cost and the runway.
  instance_type  = "t3.small"
  serve_reranker = false

  # Claude Haiku under a daily budget of 0.75 USD: the owner's three-month allowance of Claude credit.
  answer_model            = "claude-haiku-5-5"
  answer_daily_budget_usd = 0.75
}

output "public_url" {
  value = module.single_host.public_url
}

output "instance_id" {
  value = module.single_host.instance_id
}

output "cloudfront_distribution_id" {
  value = module.single_host.cloudfront_distribution_id
}

output "anthropic_key_parameter" {
  value = module.single_host.anthropic_key_parameter
}

output "github_environment_variables" {
  value = module.single_host.github_environment_variables
}

# The digest of the last apply (the state's): a later plan or apply passes it back, so it shows only the code's
# changes and does not replace the instance.
output "image_digest" {
  value = var.image_digest
}
