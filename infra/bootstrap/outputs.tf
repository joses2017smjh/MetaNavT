output "state_bucket" {
  description = "The environments' backend bucket: terraform init -backend-config=\"bucket=<this>\"."
  value       = aws_s3_bucket.state.bucket
}

output "plan_role_arn" {
  description = "Repository variable AWS_PLAN_ROLE_ARN."
  value       = aws_iam_role.plan.arn
}

output "deploy_role_arn" {
  description = "Repository variable AWS_DEPLOY_ROLE_ARN."
  value       = aws_iam_role.deploy.arn
}

output "github_repository_variables" {
  description = "The repository variables the deploy and infra workflows read (doc/deploy.md sets them with gh variable set)."
  value = {
    AWS_REGION          = var.region
    AWS_DEPLOY_ROLE_ARN = aws_iam_role.deploy.arn
    AWS_PLAN_ROLE_ARN   = aws_iam_role.plan.arn
    TF_STATE_BUCKET     = aws_s3_bucket.state.bucket
  }
}
