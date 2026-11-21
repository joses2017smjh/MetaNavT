# infra/modules/demo: one environment of the public demo (PUBLIC_DEMO=1, doc/engineering-reference.md "Public demo
# mode"): CloudFront (+ WAF) in front of an ALB that admits only CloudFront, one ECS Fargate service running the CPU
# model image by digest, RDS PostgreSQL with pgvector, Redis (ElastiCache, or a sidecar), alarms. Called by
# infra/envs/production and infra/envs/staging; doc/deploy.md is the runbook.
terraform {
  required_version = ">= 1.11.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.73"
      # CloudFront's WAF web ACL and certificate live in us-east-1, whatever the region of the rest.
      configuration_aliases = [aws.us_east_1]
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.9"
    }
  }
}
