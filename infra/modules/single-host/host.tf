# The instance, its role and the header secret. On demand only (no spot: the Free plan's offer is per instance type,
# and spot is not part of it); Amazon Linux 2023 x86_64; IMDSv2 only with a hop limit of 1, so a container (one hop
# further) gets no instance credentials; an encrypted gp3 root volume; no key pair. cloud-init (cloud-init.yaml.tftpl)
# installs Docker and the Compose plugin, makes the swap file, writes /opt/metanavit and the systemd unit, and runs the
# first deploy.

# ---- CloudFront's header value: in the state (CloudFront's origin carries it too) and in Parameter Store ----

resource "random_password" "origin_verify" {
  length  = 40
  special = false
}

resource "aws_ssm_parameter" "origin_verify" {
  name        = local.origin_verify_parameter
  description = "X-Origin-Verify, the header CloudFront adds to every request it sends to ${local.name} (ORIGIN_VERIFY_SECRET)"
  type        = "SecureString" # with the AWS managed key aws/ssm
  value       = random_password.origin_verify.result
}

# ---- The instance's role: Systems Manager, and the two SecureString parameters ----

data "aws_iam_policy_document" "ec2_trust" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "host" {
  name               = "${local.name}-host"
  description        = "The ${local.name} instance: Systems Manager (Run Command, Session Manager) and its two parameters"
  assume_role_policy = data.aws_iam_policy_document.ec2_trust.json
}

# Run Command (the deploy workflow) and Session Manager (a shell for the owner, instead of SSH). This managed policy
# also allows ssm:GetParameter on every parameter of the account; the statement below names the two the host reads.
resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.host.name
  policy_arn = "arn:${local.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# SecureStrings under the AWS managed key aws/ssm, whose key policy lets Parameter Store decrypt them for any principal
# of the account allowed to read the parameter: no KMS permission is needed here.
data "aws_iam_policy_document" "parameters" {
  statement {
    sid       = "TheTwoSecureStrings"
    actions   = ["ssm:GetParameter"]
    resources = [aws_ssm_parameter.origin_verify.arn, local.anthropic_key_parameter_arn]
  }
}

resource "aws_iam_role_policy" "parameters" {
  name   = "read-parameters"
  role   = aws_iam_role.host.id
  policy = data.aws_iam_policy_document.parameters.json
}

resource "aws_iam_instance_profile" "host" {
  name = "${local.name}-host"
  role = aws_iam_role.host.name
}

# ---- What cloud-init writes and runs ----

# The newest Amazon Linux 2023 AMI (x86_64), from AWS's public parameter. The instance ignores a newer one later; a
# replaced instance starts from the newest.
data "aws_ssm_parameter" "al2023" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

locals {
  # Docker Compose v5.6.1 (2026-10-21): the sha256 of its docker-compose-linux-x86_64, from the release's checksums.txt
  # and its .sha256 file (both read 2026-11-21, and equal to a download's). cloud-init stops if a download differs.
  compose_version = "v5.6.1"
  compose_sha256  = "47737f4236e6d3ccea7291995203f8f018058783b4c3a9469119811a0b5db8c0"

  # /opt/metanavit/app.env: the host's settings, not secret. compose.yml interpolates the last seven; deploy.sh reads
  # the first four.
  app_env = {
    IMAGE_REPOSITORY        = var.image_repository
    AWS_REGION              = local.region
    ORIGIN_VERIFY_PARAMETER = aws_ssm_parameter.origin_verify.name
    ANTHROPIC_KEY_PARAMETER = local.anthropic_key_parameter
    WEB_ORIGIN              = local.public_url
    ANSWER_MODEL            = var.answer_model
    ANSWER_DAILY_BUDGET_USD = format("%.2f", var.answer_daily_budget_usd)
    RERANKER_MODEL          = local.reranker_model
    API_MEMORY              = "${var.api_memory_mib}m"
    DB_MEMORY               = "${var.db_memory_mib}m"
    REDIS_MEMORY            = "${var.redis_memory_mib}m"
  }
  app_env_file = join("", concat(["# Written by Terraform (infra/modules/single-host, local.app_env). Not secret.\n"],
  [for name in sort(keys(local.app_env)) : "${name}=${local.app_env[name]}\n"]))

  # The three host files are the repository's own (deploy/single-host), embedded as they are committed.
  host_files = "${path.module}/../../../deploy/single-host"
  cloud_init = templatefile("${path.module}/cloud-init.yaml.tftpl", {
    compose         = filebase64("${local.host_files}/compose.yml")
    deploy          = filebase64("${local.host_files}/deploy.sh")
    unit            = filebase64("${local.host_files}/metanavit.service")
    app_env         = base64encode(local.app_env_file)
    swap_mib        = var.swap_mib
    compose_version = local.compose_version
    compose_sha256  = local.compose_sha256
    image_digest    = var.image_digest
  })
}

# ---- The instance ----

resource "aws_instance" "host" {
  ami                  = data.aws_ssm_parameter.al2023.insecure_value
  instance_type        = var.instance_type
  iam_instance_profile = aws_iam_instance_profile.host.name
  monitoring           = false # basic monitoring: the status checks the alarm reads are free; detailed monitoring is not

  # gzip: cloud-init reads compressed user data, and the embedded files would not fit the 16 KB limit as text
  user_data_base64            = base64gzip(local.cloud_init)
  user_data_replace_on_change = true # cloud-init runs once per instance: a new host file means a new instance

  primary_network_interface {
    network_interface_id = aws_network_interface.host.id
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
    instance_metadata_tags      = "disabled"
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_gb
    encrypted             = true
    delete_on_termination = true
  }

  # Burstable types: standard mode, so CPU above the baseline is never billed as surplus credits (unlimited mode bills
  # them). c7i-flex and m7i-flex are not burstable and take no credit specification.
  dynamic "credit_specification" {
    for_each = startswith(var.instance_type, "t") ? [1] : []

    content {
      cpu_credits = "standard"
    }
  }

  tags = local.instance_tags

  # the first boot reads both parameters and registers with Systems Manager: the role must hold its policies by then
  depends_on = [aws_iam_role_policy.parameters, aws_iam_role_policy_attachment.ssm_core]

  lifecycle {
    # A newer AMI does not replace a running host; a host replaced for another reason boots the newest.
    ignore_changes = [ami]

    precondition {
      condition     = var.api_memory_mib + var.db_memory_mib + var.redis_memory_mib + var.host_reserve_mib <= local.instance_memory_mib[var.instance_type]
      error_message = "The memory limits (api_memory_mib + db_memory_mib + redis_memory_mib) and host_reserve_mib exceed the instance type's memory."
    }

    # A deploy stops the api before its one-off migrate container starts (deploy.sh): the two never run side by side.
    precondition {
      condition     = !var.serve_reranker || (local.instance_memory_mib[var.instance_type] >= 4096 && var.api_memory_mib >= 3072)
      error_message = "serve_reranker = true needs an instance type with 4 GiB or more (c7i-flex.large or m7i-flex.large) and api_memory_mib of 3072 or more: the API's measured peak with the reranker is 2719 MiB (bench/results/api_memory_cpu.json)."
    }
  }
}
