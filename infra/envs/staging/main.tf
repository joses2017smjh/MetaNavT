# The staging environment: OPT-IN (doc/deploy.md, staging mode). A second always-on stack about doubles the monthly
# bill, so by default nothing applies this root and the deploy workflow goes straight to production. To test a deploy
# here: apply it, set the repository variable DEPLOY_STAGING=true (and add "staging" to the bootstrap's
# deploy_environments), deploy, then `terraform destroy` it and unset the variable. Smaller than production: Redis as
# a sidecar instead of ElastiCache, no WAF, no deletion protection and no final snapshot.
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

  backend "s3" {
    key          = "staging/terraform.tfstate"
    region       = "us-west-2"
    encrypt      = true
    use_lockfile = true
  }
}

variable "region" {
  description = "Region of the environment (the backend block above names the state bucket's region)."
  type        = string
  default     = "us-west-2"
}

variable "image_digest" {
  description = "sha256:<64 hex> of the CPU model image to run."
  type        = string
}

variable "alert_email" {
  description = "Optional address for the alarm topic. Never commit it."
  type        = string
  default     = null
  sensitive   = true
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project     = "metanavit"
      Environment = "staging"
      ManagedBy   = "terraform"
    }
  }
}

provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"

  default_tags {
    tags = {
      Project     = "metanavit"
      Environment = "staging"
      ManagedBy   = "terraform"
    }
  }
}

module "demo" {
  source = "../../modules/demo"

  providers = {
    aws           = aws
    aws.us_east_1 = aws.us_east_1
  }

  environment  = "staging"
  image_digest = var.image_digest
  alert_email  = var.alert_email

  redis_mode          = "sidecar"
  enable_waf          = false
  deletion_protection = false
  final_snapshot      = false
  vpc_cidr            = "10.41.0.0/16"

  anthropic_key_stored = false
}

output "public_url" {
  value = module.demo.public_url
}

output "anthropic_secret_name" {
  value = module.demo.anthropic_secret_name
}

output "github_environment_variables" {
  value = module.demo.github_environment_variables
}

# The digest of the last apply (the state's): a later plan passes it back, so it shows only the code's changes.
output "image_digest" {
  value = var.image_digest
}
