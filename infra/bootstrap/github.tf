# GitHub Actions reaches AWS through OIDC: a job asks GitHub for a short-lived token and exchanges it for one of the
# two roles below. No AWS key is stored in GitHub. A role is assumable only by tokens whose sub claim names this
# repository and, for the deploy role, one of the GitHub environments the owner protects (doc/deploy.md).

locals {
  oidc_host = "token.actions.githubusercontent.com"

  # The sub claim GitHub puts in a token: "<prefix>:pull_request" for a pull request's jobs, "<prefix>:environment:<name>"
  # for a job that names an environment. <prefix> is repo:<owner>/<name>, or repo:<owner>@<owner id>/<name>@<id> for a
  # repository with immutable subject claims, as this one has (var.github_subject_prefix).
  subject_prefix  = coalesce(var.github_subject_prefix, "repo:${var.github_repository}")
  plan_subjects   = ["${local.subject_prefix}:pull_request"]
  deploy_subjects = [for env in var.deploy_environments : "${local.subject_prefix}:environment:${env}"]

  # Every environment's resources are named <project>-<environment>... (infra/modules/demo).
  ecs_arn    = "arn:${local.partition}:ecs:${var.region}:${local.account_id}"
  name_glob  = "${var.project}-*"
  web_bucket = "arn:${local.partition}:s3:::${var.project}-*-web-${local.account_id}"
}

resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://${local.oidc_host}"
  client_id_list = ["sts.amazonaws.com"]
}

# ---- Plan role: read-only, for pull requests (terraform plan -lock=false; .github/workflows/infra.yml) ----

data "aws_iam_policy_document" "plan_trust" {
  statement {
    sid     = "PullRequestsOfThisRepository"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "${local.oidc_host}:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "${local.oidc_host}:sub"
      values   = local.plan_subjects
    }
  }
}

resource "aws_iam_role" "plan" {
  name                 = "${var.project}-github-plan"
  description          = "terraform plan from pull requests of ${var.github_repository}: read-only"
  assume_role_policy   = data.aws_iam_policy_document.plan_trust.json
  max_session_duration = 3600
}

# Read-only across the account, which includes reading the state bucket. Nothing that writes: the plan job runs
# terraform plan with -lock=false, so it does not even write a lock file.
resource "aws_iam_role_policy_attachment" "plan_read_only" {
  role       = aws_iam_role.plan.name
  policy_arn = "arn:${local.partition}:iam::aws:policy/ReadOnlyAccess"
}

# ---- Deploy role: the only role that writes, assumable only from the protected GitHub environments ----

data "aws_iam_policy_document" "deploy_trust" {
  statement {
    sid     = "ProtectedEnvironmentsOfThisRepository"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "${local.oidc_host}:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "${local.oidc_host}:sub"
      values   = local.deploy_subjects
    }
  }
}

resource "aws_iam_role" "deploy" {
  name                 = "${var.project}-github-deploy"
  description          = "The deploy workflow of ${var.github_repository} (.github/workflows/deploy.yml), from its protected environments"
  assume_role_policy   = data.aws_iam_policy_document.deploy_trust.json
  max_session_duration = 3600
}

data "aws_iam_policy_document" "deploy" {
  # ECS takes no resource-level permission for these two: a revision is registered from the current one with only the
  # image changed (scripts/deploy_ecs.py), and only the two roles below may be passed to it.
  statement {
    sid       = "TaskDefinitions"
    actions   = ["ecs:DescribeTaskDefinition", "ecs:RegisterTaskDefinition"]
    resources = ["*"]
  }

  statement {
    sid       = "UpdateTheServices"
    actions   = ["ecs:DescribeServices", "ecs:UpdateService"]
    resources = ["${local.ecs_arn}:service/${local.name_glob}/*"]
  }

  # Only the migrate/index task family may be run by hand, and only in this project's clusters.
  statement {
    sid       = "RunTheMigrateTask"
    actions   = ["ecs:RunTask"]
    resources = ["${local.ecs_arn}:task-definition/${var.project}-*-migrate:*"]

    condition {
      test     = "ArnLike"
      variable = "ecs:cluster"
      values   = ["${local.ecs_arn}:cluster/${local.name_glob}"]
    }
  }

  statement {
    sid       = "FollowTheMigrateTask"
    actions   = ["ecs:DescribeTasks", "ecs:StopTask"]
    resources = ["${local.ecs_arn}:task/${local.name_glob}/*"]
  }

  statement {
    sid     = "PassTheTwoTaskRoles"
    actions = ["iam:PassRole"]
    resources = [
      "arn:${local.partition}:iam::${local.account_id}:role/${var.project}-*-task",
      "arn:${local.partition}:iam::${local.account_id}:role/${var.project}-*-execution",
    ]

    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["ecs-tasks.amazonaws.com"]
    }
  }

  # Metadata only (is a value stored?): the values are read by the tasks' execution role at start, never by the
  # workflow, so it gets no secretsmanager:GetSecretValue.
  statement {
    sid       = "AppSecretsMetadata"
    actions   = ["secretsmanager:DescribeSecret"]
    resources = ["arn:${local.partition}:secretsmanager:${var.region}:${local.account_id}:secret:${local.name_glob}"]
  }

  # The web client's export goes to the web bucket; nothing is deleted (old hashed assets stay for open pages).
  statement {
    sid       = "UploadTheWebClient"
    actions   = ["s3:PutObject"]
    resources = ["${local.web_bucket}/*"]
  }

  statement {
    sid       = "ListTheWebBucket"
    actions   = ["s3:ListBucket"]
    resources = [local.web_bucket]
  }

  # Distribution ids are random, so the account's distributions; an invalidation only empties CloudFront's cache.
  statement {
    sid       = "InvalidateTheCache"
    actions   = ["cloudfront:CreateInvalidation", "cloudfront:GetInvalidation"]
    resources = ["arn:${local.partition}:cloudfront::${local.account_id}:distribution/*"]
  }

  # ---- The single host (infra/envs/single-host; .github/workflows/deploy-single-host.yml) ----

  # Run Command on the instance carrying both tags (infra/modules/single-host sets them), and with one document only:
  # SendCommand is authorized against the instance and against the document. ssm:resourceTag, not aws:ResourceTag:
  # with a global condition key, Systems Manager's guide asks for an aws:ViaAWSService statement too.
  statement {
    sid       = "RunCommandOnTheSingleHost"
    actions   = ["ssm:SendCommand"]
    resources = ["arn:${local.partition}:ec2:${var.region}:${local.account_id}:instance/*"]

    condition {
      test     = "StringEquals"
      variable = "ssm:resourceTag/Project"
      values   = [var.project]
    }

    condition {
      test     = "StringEquals"
      variable = "ssm:resourceTag/Target"
      values   = ["single-host"]
    }
  }

  # An AWS-owned document: its ARN has no account id.
  statement {
    sid       = "OnlyTheShellScriptDocument"
    actions   = ["ssm:SendCommand"]
    resources = ["arn:${local.partition}:ssm:${var.region}::document/AWS-RunShellScript"]
  }

  # Neither the instance lookup nor following a command's invocation takes a resource-level permission (the Service
  # Authorization Reference lists no resource type for them): the workflow finds the instance by its tags and reads
  # its own command's output.
  statement {
    sid       = "FindTheHostAndFollowTheCommand"
    actions   = ["ec2:DescribeInstances", "ssm:GetCommandInvocation", "ssm:ListCommandInvocations"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "deploy" {
  name   = "deploy"
  role   = aws_iam_role.deploy.id
  policy = data.aws_iam_policy_document.deploy.json
}
