# Optional (alert_email set): an email when the instance or the hardware under it fails an EC2 status check. One
# standard-resolution alarm on a free EC2 metric, within CloudWatch's free alarms; the topic exists only with it.

locals {
  alarm = var.alert_email != null && var.alert_email != ""
}

resource "aws_sns_topic" "alarms" {
  count = local.alarm ? 1 : 0

  name = "${local.name}-alarms"
}

resource "aws_sns_topic_subscription" "alarm_email" {
  count = local.alarm ? 1 : 0

  topic_arn = aws_sns_topic.alarms[0].arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_cloudwatch_metric_alarm" "status_check" {
  count = local.alarm ? 1 : 0

  alarm_name          = "${local.name}-status-check"
  alarm_description   = "The ${local.name} instance or its hardware failed an EC2 status check"
  namespace           = "AWS/EC2"
  metric_name         = "StatusCheckFailed"
  dimensions          = { InstanceId = aws_instance.host.id }
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 2
  comparison_operator = "GreaterThanOrEqualToThreshold"
  threshold           = 1
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms[0].arn]
  ok_actions          = [aws_sns_topic.alarms[0].arn]
}
