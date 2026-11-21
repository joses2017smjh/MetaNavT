# A WAF web ACL on the distribution (CloudFront scope, so it lives in us-east-1): AWS's managed common rules, then a
# per-IP rate limit on the answer route and another on everything else, and an optional CAPTCHA on the answer route.
# The app's own per-caller quotas and daily budget (Redis) still apply behind it.

locals {
  answer_path_prefix = "/api/answer"
}

resource "aws_wafv2_web_acl" "this" {
  count    = var.enable_waf ? 1 : 0
  provider = aws.us_east_1

  name        = local.name
  description = "${local.name}: managed common rules and per-IP rate limits"
  scope       = "CLOUDFRONT"

  default_action {
    allow {}
  }

  rule {
    name     = "aws-common-rules"
    priority = 0

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        vendor_name = "AWS"
        name        = "AWSManagedRulesCommonRuleSet"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${local.name}-aws-common-rules"
      sampled_requests_enabled   = true
    }
  }

  dynamic "rule" {
    for_each = var.waf_answer_captcha ? [1] : []

    content {
      name     = "answer-captcha"
      priority = 1

      action {
        captcha {}
      }

      statement {
        byte_match_statement {
          search_string         = local.answer_path_prefix
          positional_constraint = "STARTS_WITH"

          field_to_match {
            uri_path {}
          }

          text_transformation {
            priority = 0
            type     = "NONE"
          }
        }
      }

      visibility_config {
        cloudwatch_metrics_enabled = true
        metric_name                = "${local.name}-answer-captcha"
        sampled_requests_enabled   = true
      }
    }
  }

  rule {
    name     = "answer-rate-per-ip"
    priority = 2

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit                 = var.waf_answer_rate_limit
        evaluation_window_sec = 300
        aggregate_key_type    = "IP"

        scope_down_statement {
          byte_match_statement {
            search_string         = local.answer_path_prefix
            positional_constraint = "STARTS_WITH"

            field_to_match {
              uri_path {}
            }

            text_transformation {
              priority = 0
              type     = "NONE"
            }
          }
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${local.name}-answer-rate-per-ip"
      sampled_requests_enabled   = true
    }
  }

  rule {
    name     = "general-rate-per-ip"
    priority = 3

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit                 = var.waf_general_rate_limit
        evaluation_window_sec = 300
        aggregate_key_type    = "IP"

        scope_down_statement {
          not_statement {
            statement {
              byte_match_statement {
                search_string         = local.answer_path_prefix
                positional_constraint = "STARTS_WITH"

                field_to_match {
                  uri_path {}
                }

                text_transformation {
                  priority = 0
                  type     = "NONE"
                }
              }
            }
          }
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${local.name}-general-rate-per-ip"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = local.name
    sampled_requests_enabled   = true
  }
}
