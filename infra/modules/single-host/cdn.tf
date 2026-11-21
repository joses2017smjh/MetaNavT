# CloudFront in front of the host, on its default *.cloudfront.net certificate: viewers reach it over HTTPS only, and it
# reaches the host over HTTP on port 80 at the Elastic IP's public DNS name, adding X-Origin-Verify to every request.
# /api/* and /health are never cached and carry every viewer header but Host; the web client's hashed assets
# (/_next/static/*) are cached; every other path (the pages) is not. No WAF: it is not free.

# ---- AWS managed policies, looked up by name ----

data "aws_cloudfront_cache_policy" "disabled" {
  name = "Managed-CachingDisabled"
}

data "aws_cloudfront_cache_policy" "optimized" {
  name = "Managed-CachingOptimized"
}

data "aws_cloudfront_origin_request_policy" "all_viewer_except_host" {
  name = "Managed-AllViewerExceptHostHeader"
}

data "aws_cloudfront_response_headers_policy" "security" {
  name = "Managed-SecurityHeadersPolicy"
}

# ---- The distribution ----

locals {
  api_paths = ["/api/*", "/health"]
}

resource "aws_cloudfront_distribution" "this" {
  enabled         = true
  comment         = local.name
  is_ipv6_enabled = true
  http_version    = "http2and3"
  price_class     = var.price_class

  origin {
    origin_id   = "host"
    domain_name = aws_eip.host.public_dns

    custom_origin_config {
      http_port                = 80
      https_port               = 443
      origin_protocol_policy   = "http-only"
      origin_ssl_protocols     = ["TLSv1.2"]
      origin_read_timeout      = var.origin_read_timeout
      origin_keepalive_timeout = 5 # shorter than the API's own keep-alive (compose.yml), so CloudFront closes first
    }

    custom_header {
      name  = local.origin_header_name
      value = random_password.origin_verify.result
    }
  }

  # The pages: the web client's export, which the api serves at / (ENVIRONMENT=prod), never cached.
  default_cache_behavior {
    target_origin_id           = "host"
    viewer_protocol_policy     = "redirect-to-https"
    allowed_methods            = ["GET", "HEAD"]
    cached_methods             = ["GET", "HEAD"]
    compress                   = true
    cache_policy_id            = data.aws_cloudfront_cache_policy.disabled.id
    response_headers_policy_id = data.aws_cloudfront_response_headers_policy.security.id
  }

  # The API: HTTPS only (a POST redirected from HTTP would lose its body), never cached, every viewer header but Host.
  dynamic "ordered_cache_behavior" {
    for_each = local.api_paths

    content {
      path_pattern               = ordered_cache_behavior.value
      target_origin_id           = "host"
      viewer_protocol_policy     = "https-only"
      allowed_methods            = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
      cached_methods             = ["GET", "HEAD"]
      compress                   = true
      cache_policy_id            = data.aws_cloudfront_cache_policy.disabled.id
      origin_request_policy_id   = data.aws_cloudfront_origin_request_policy.all_viewer_except_host.id
      response_headers_policy_id = data.aws_cloudfront_response_headers_policy.security.id
    }
  }

  # The web client's hashed assets: their names change with their content, so they are cached.
  ordered_cache_behavior {
    path_pattern               = "/_next/static/*"
    target_origin_id           = "host"
    viewer_protocol_policy     = "redirect-to-https"
    allowed_methods            = ["GET", "HEAD"]
    cached_methods             = ["GET", "HEAD"]
    compress                   = true
    cache_policy_id            = data.aws_cloudfront_cache_policy.optimized.id
    response_headers_policy_id = data.aws_cloudfront_response_headers_policy.security.id
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}
