output "public_url" {
  description = "The live demo's address (CloudFront's)."
  value       = local.public_url
}

output "instance_id" {
  description = "The instance (aws ssm start-session --target <this> for a shell; no SSH)."
  value       = aws_instance.host.id
}

output "cloudfront_distribution_id" {
  description = "The distribution's id."
  value       = aws_cloudfront_distribution.this.id
}

output "anthropic_key_parameter" {
  description = "Where the owner stores the Anthropic API key, as a SecureString (doc/deploy.md); until then answers are off."
  value       = local.anthropic_key_parameter
}

# The single-host deploy workflow's environment variables: doc/deploy.md sets them on the GitHub environment
# production with gh variable set. tests/engine/test_single_host.py checks that these are exactly the vars it reads.
output "github_environment_variables" {
  description = "Variables of the GitHub environment production, for .github/workflows/deploy-single-host.yml."
  value = {
    PUBLIC_URL       = local.public_url
    SINGLE_HOST_NAME = local.instance_tags.Name
    SERVE_RERANKER   = tostring(var.serve_reranker)
  }
}
