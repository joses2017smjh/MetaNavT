data "aws_caller_identity" "current" {}

data "aws_partition" "current" {}

data "aws_region" "current" {}

# An availability zone that offers the instance type (not every zone offers every type).
data "aws_ec2_instance_type_offerings" "zones" {
  location_type = "availability-zone"

  filter {
    name   = "instance-type"
    values = [var.instance_type]
  }
}

locals {
  name       = "${var.project}-single-host"
  account_id = data.aws_caller_identity.current.account_id
  partition  = data.aws_partition.current.partition
  region     = data.aws_region.current.region
  zone       = sort(data.aws_ec2_instance_type_offerings.zones.locations)[0]

  # The instance's own tags. The deploy role may run commands only on an instance carrying both (infra/bootstrap), and
  # the deploy workflow finds it by Name and Target.
  instance_tags = {
    Name    = local.name
    Project = var.project
    Target  = "single-host"
  }

  # Memory of each type the variable allows, MiB (the t4g types are refused there: ARM).
  instance_memory_mib = {
    "t3.micro"       = 1024
    "t3.small"       = 2048
    "c7i-flex.large" = 4096
    "m7i-flex.large" = 8192
    "t8i.small"      = 2048
  }

  # The two SecureString parameters the host reads at every start and deploy (deploy/single-host/deploy.sh): CloudFront's
  # header value, which Terraform makes (host.tf), and the Anthropic key, which the owner stores (doc/deploy.md) and
  # which may never exist: then answers are off.
  origin_verify_parameter     = "/${var.project}/single-host/origin-verify"
  anthropic_key_parameter     = "/${var.project}/single-host/anthropic-api-key"
  anthropic_key_parameter_arn = "arn:${local.partition}:ssm:${local.region}:${local.account_id}:parameter${local.anthropic_key_parameter}"

  # CloudFront adds this header to every request it sends to the host; the app refuses a request without it
  # (ORIGIN_VERIFY_SECRET, app/api/public.py), since every CloudFront distribution shares the prefix list the security
  # group admits.
  origin_header_name = "X-Origin-Verify"

  # none is the app's "not configured" (app/engine/retriever.py load_reranker): the fused order is served, labelled.
  reranker_model = var.serve_reranker ? "BAAI/bge-reranker-v2-m3" : "none"

  public_url = "https://${aws_cloudfront_distribution.this.domain_name}"
}
