# infra/bootstrap: what the environments need before their first apply, applied once by the owner with an
# administrator's credentials (doc/deploy.md): the state bucket, GitHub's OIDC provider, the plan and deploy roles,
# and the account's monthly budget. Its own state stays local, with the owner: it holds no secret.
terraform {
  # 1.10 for the S3 backend's lockfile (the environments' use_lockfile); 1.11 for the write-only arguments that keep
  # the environments' database password out of their state. One floor for every root under infra/.
  required_version = ">= 1.11.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.73"
    }
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = var.project
      Stack     = "bootstrap"
      ManagedBy = "terraform"
    }
  }
}
