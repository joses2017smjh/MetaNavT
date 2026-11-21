# infra/modules/single-host: the live demo on one EC2 instance sized to the AWS Free plan (doc/deploy.md, "The live
# demo: one host on the AWS Free plan"). CloudFront in front of one on-demand instance in one public subnet (no NAT),
# with an Elastic IP as CloudFront's origin, access through Systems Manager only, and deploy/single-host/compose.yml
# running the CPU model image by digest beside Postgres with pgvector and Redis; an optional status-check alarm.
# Called by infra/envs/single-host. The ECS stack (infra/modules/demo) stays the second, production-grade target.
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
}
