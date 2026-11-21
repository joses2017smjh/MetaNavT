# The production environment of the public demo (doc/deploy.md). Settings live here, in code, so a pull request
# shows every change; only the image digest and the alarm email come from the command line.
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
    key          = "production/terraform.tfstate"
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
  description = "sha256:<64 hex> of the CPU model image to run (doc/deploy.md: the digest the service runs now, or a new one)."
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
      Environment = "production"
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
      Environment = "production"
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

  environment  = "production"
  image_digest = var.image_digest
  alert_email  = var.alert_email

  redis_mode          = "elasticache"
  enable_waf          = true
  deletion_protection = true
  vpc_cidr            = "10.40.0.0/16"

  # Set to true, in a pull request, once the key is in the secret (doc/deploy.md, one-time steps).
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
