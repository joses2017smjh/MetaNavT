data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

data "aws_availability_zones" "available" {
  state = "available"

  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

locals {
  name       = "${var.project}-${var.environment}"
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.region
  azs        = slice(data.aws_availability_zones.available.names, 0, 2)

  elasticache   = var.redis_mode == "elasticache"
  sidecar_redis = var.redis_mode == "sidecar"
  custom_domain = var.domain_name != null

  # The image runs by digest only (variables.tf validates the digest's form).
  image = "${var.image_repository}@${var.image_digest}"

  # Sidecar images, pinned by digest as every image in this repository is (the tag in front, its release beside):
  # redis:7.2-alpine = 7.2.17, the same index digest docker-compose.prod.yml pins, from ECR Public's mirror of the
  # Docker official image (no Docker Hub pull limit from AWS); aws-otel-collector v0.51.0 (2026-10-29). Looked up
  # 2026-11-20.
  redis_image = "public.ecr.aws/docker/library/redis:7.2-alpine@sha256:2f66de68111e64c3927859f158f8e666066dae2b97d9f82770a6f37a5aa1c335"
  adot_image  = "public.ecr.aws/aws-observability/aws-otel-collector:v0.51.0@sha256:955e240b27d7fd3c7995fd937db4b963975948eb553f656d8c37a564e5b5e07d"

  db_name = "metanavit"
  db_user = "metanavit"

  # CloudFront adds this header to every request it sends to the ALB; the listener forwards only requests that carry
  # its value (alb.tf). The security group already admits only CloudFront's addresses, so the header is what keeps
  # out another account's CloudFront distribution.
  origin_header_name = "X-Origin-Verify"
}

check "custom_domain_has_a_zone" {
  assert {
    condition     = !local.custom_domain || var.route53_zone_id != null
    error_message = "domain_name needs route53_zone_id: the certificate is validated and the alias records are created in that hosted zone."
  }
}
