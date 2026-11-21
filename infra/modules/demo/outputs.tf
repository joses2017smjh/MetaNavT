output "public_url" {
  description = "The demo's address (CloudFront's, or the custom domain)."
  value       = local.public_origin
}

output "cloudfront_distribution_id" {
  description = "The distribution's id (the deploy workflow invalidates its cache)."
  value       = aws_cloudfront_distribution.this.id
}

output "web_bucket" {
  description = "The web client's bucket."
  value       = aws_s3_bucket.web.bucket
}

output "ecs_cluster" {
  description = "The ECS cluster."
  value       = aws_ecs_cluster.this.name
}

output "ecs_service" {
  description = "The API service."
  value       = aws_ecs_service.api.name
}

output "anthropic_secret_name" {
  description = "Where the owner stores the Anthropic API key (doc/deploy.md)."
  value       = aws_secretsmanager_secret.anthropic_api_key.name
}

output "alarm_topic_arn" {
  description = "The alarms' SNS topic."
  value       = aws_sns_topic.alarms.arn
}

# The deploy workflow's environment variables: doc/deploy.md sets them on the GitHub environment with gh variable set.
# tests/engine/test_deploy_workflow.py checks that these are exactly the vars the deploy job reads.
output "github_environment_variables" {
  description = "Variables of this environment's GitHub environment, for .github/workflows/deploy.yml."
  value = {
    ECS_CLUSTER                = aws_ecs_cluster.this.name
    ECS_SERVICE                = aws_ecs_service.api.name
    API_TASK_FAMILY            = aws_ecs_task_definition.api.family
    MIGRATE_TASK_FAMILY        = aws_ecs_task_definition.migrate.family
    WEB_BUCKET                 = aws_s3_bucket.web.bucket
    CLOUDFRONT_DISTRIBUTION_ID = aws_cloudfront_distribution.this.id
    PUBLIC_URL                 = local.public_origin
  }
}
