# Alarms to an SNS topic (an email subscription when alert_email is set): ALB and API 5xx, the API's response time,
# API task stops that no deployment asked for, RDS CPU and free storage, and the Redis connection count. Only free
# metrics: no Container Insights. The account's monthly budget is in infra/bootstrap.

resource "aws_sns_topic" "alarms" {
  name = "${local.name}-alarms"
}

resource "aws_sns_topic_subscription" "alarm_email" {
  count = var.alert_email == null || var.alert_email == "" ? 0 : 1

  topic_arn = aws_sns_topic.alarms.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

locals {
  alarm_actions = [aws_sns_topic.alarms.arn]
}

resource "aws_cloudwatch_metric_alarm" "alb_5xx" {
  alarm_name          = "${local.name}-5xx"
  alarm_description   = "5xx responses from the ALB (no healthy task, timeouts) plus the API's own, per 5 minutes"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  threshold           = var.alarm_5xx_per_5_minutes
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions

  metric_query {
    id          = "total"
    expression  = "FILL(elb, 0) + FILL(target, 0)"
    label       = "5xx responses"
    return_data = true
  }

  metric_query {
    id = "elb"

    metric {
      namespace   = "AWS/ApplicationELB"
      metric_name = "HTTPCode_ELB_5XX_Count"
      period      = 300
      stat        = "Sum"
      dimensions  = { LoadBalancer = aws_lb.this.arn_suffix }
    }
  }

  metric_query {
    id = "target"

    metric {
      namespace   = "AWS/ApplicationELB"
      metric_name = "HTTPCode_Target_5XX_Count"
      period      = 300
      stat        = "Sum"
      dimensions  = { LoadBalancer = aws_lb.this.arn_suffix, TargetGroup = aws_lb_target_group.api.arn_suffix }
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "response_time" {
  alarm_name          = "${local.name}-response-time"
  alarm_description   = "p90 of the API's response time (an answer is a CPU rerank plus a Claude call)"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "TargetResponseTime"
  dimensions          = { LoadBalancer = aws_lb.this.arn_suffix, TargetGroup = aws_lb_target_group.api.arn_suffix }
  extended_statistic  = "p90"
  period              = 300
  evaluation_periods  = 3
  comparison_operator = "GreaterThanThreshold"
  threshold           = var.alarm_response_time_p90_seconds
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions
}

# ---- API task stops (a crash, a failed health check, a task that cannot start), not deployment replacements ----

resource "aws_cloudwatch_event_rule" "task_stopped" {
  name        = "${local.name}-task-stopped"
  description = "An API task stopped for a reason other than a deployment or a scale-in"

  event_pattern = jsonencode({
    source        = ["aws.ecs"]
    "detail-type" = ["ECS Task State Change"]
    detail = {
      clusterArn    = [aws_ecs_cluster.this.arn]
      group         = ["service:${aws_ecs_service.api.name}"]
      lastStatus    = ["STOPPED"]
      stoppedReason = [{ "anything-but" = { prefix = "Scaling activity initiated by" } }]
    }
  })
}

# Each stop event is kept (with its stoppedReason) in a log group, the rule's target.
resource "aws_cloudwatch_log_group" "task_stopped" {
  name              = "/aws/events/${local.name}-task-stopped"
  retention_in_days = var.log_retention_days
}

data "aws_iam_policy_document" "events_to_logs" {
  statement {
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.task_stopped.arn}:*"]

    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com", "delivery.logs.amazonaws.com"]
    }
  }
}

resource "aws_cloudwatch_log_resource_policy" "events_to_logs" {
  policy_name     = "${local.name}-task-stopped-events"
  policy_document = data.aws_iam_policy_document.events_to_logs.json
}

resource "aws_cloudwatch_event_target" "task_stopped" {
  rule = aws_cloudwatch_event_rule.task_stopped.name
  arn  = aws_cloudwatch_log_group.task_stopped.arn

  depends_on = [aws_cloudwatch_log_resource_policy.events_to_logs]
}

resource "aws_cloudwatch_metric_alarm" "task_stopped" {
  alarm_name          = "${local.name}-task-restarts"
  alarm_description   = "An API task stopped unasked (log group /aws/events/${local.name}-task-stopped has the reason)"
  namespace           = "AWS/Events"
  metric_name         = "TriggeredRules"
  dimensions          = { RuleName = aws_cloudwatch_event_rule.task_stopped.name }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  threshold           = 1
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
}

# ---- Database and Redis ----

resource "aws_cloudwatch_metric_alarm" "rds_cpu" {
  alarm_name          = "${local.name}-db-cpu"
  alarm_description   = "RDS average CPU"
  namespace           = "AWS/RDS"
  metric_name         = "CPUUtilization"
  dimensions          = { DBInstanceIdentifier = aws_db_instance.this.identifier }
  statistic           = "Average"
  period              = 300
  evaluation_periods  = 3
  comparison_operator = "GreaterThanThreshold"
  threshold           = var.alarm_rds_cpu_percent
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions
}

resource "aws_cloudwatch_metric_alarm" "rds_storage" {
  alarm_name          = "${local.name}-db-free-storage"
  alarm_description   = "RDS free storage"
  namespace           = "AWS/RDS"
  metric_name         = "FreeStorageSpace"
  dimensions          = { DBInstanceIdentifier = aws_db_instance.this.identifier }
  statistic           = "Minimum"
  period              = 300
  evaluation_periods  = 1
  comparison_operator = "LessThanThreshold"
  threshold           = var.alarm_rds_free_storage_gb * 1024 * 1024 * 1024
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions
}

resource "aws_cloudwatch_metric_alarm" "redis_connections" {
  count = local.elasticache ? 1 : 0

  alarm_name          = "${local.name}-redis-connections"
  alarm_description   = "Open connections to the ElastiCache node"
  namespace           = "AWS/ElastiCache"
  metric_name         = "CurrConnections"
  dimensions          = { CacheClusterId = tolist(aws_elasticache_replication_group.this[0].member_clusters)[0] }
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 2
  comparison_operator = "GreaterThanThreshold"
  threshold           = var.alarm_redis_connections
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions
}
