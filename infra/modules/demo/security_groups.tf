# One rule per resource, so each admitted flow is one block. A group created here starts with no rule at all: the
# provider removes the allow-all egress rule AWS adds to a new group.

# CloudFront's origin-facing addresses, as AWS publishes and maintains them.
data "aws_ec2_managed_prefix_list" "cloudfront" {
  name = "com.amazonaws.global.cloudfront.origin-facing"
}

# ---- ALB: HTTP from CloudFront only; out to the tasks' port only ----

resource "aws_security_group" "alb" {
  name        = "${local.name}-alb"
  description = "ALB: HTTP from CloudFront origin-facing addresses only"
  vpc_id      = aws_vpc.this.id
}

resource "aws_vpc_security_group_ingress_rule" "alb_http_from_cloudfront" {
  security_group_id = aws_security_group.alb.id
  description       = "HTTP from CloudFront origin-facing addresses"
  ip_protocol       = "tcp"
  from_port         = 80
  to_port           = 80
  prefix_list_id    = data.aws_ec2_managed_prefix_list.cloudfront.id
}

resource "aws_vpc_security_group_egress_rule" "alb_to_tasks" {
  security_group_id            = aws_security_group.alb.id
  description                  = "The API port of the tasks"
  ip_protocol                  = "tcp"
  from_port                    = 8000
  to_port                      = 8000
  referenced_security_group_id = aws_security_group.tasks.id
}

# ---- Tasks: the API port from the ALB only; out to HTTPS, Postgres and Redis ----

resource "aws_security_group" "tasks" {
  name        = "${local.name}-tasks"
  description = "API tasks: port 8000 from the ALB only"
  vpc_id      = aws_vpc.this.id
}

resource "aws_vpc_security_group_ingress_rule" "tasks_api_from_alb" {
  security_group_id            = aws_security_group.tasks.id
  description                  = "The API port from the ALB"
  ip_protocol                  = "tcp"
  from_port                    = 8000
  to_port                      = 8000
  referenced_security_group_id = aws_security_group.alb.id
}

resource "aws_vpc_security_group_egress_rule" "tasks_https" {
  security_group_id = aws_security_group.tasks.id
  description       = "HTTPS: image registry, model downloads, Anthropic API, Secrets Manager, CloudWatch Logs"
  ip_protocol       = "tcp"
  from_port         = 443
  to_port           = 443
  cidr_ipv4         = "0.0.0.0/0"
}

resource "aws_vpc_security_group_egress_rule" "tasks_to_db" {
  security_group_id            = aws_security_group.tasks.id
  description                  = "PostgreSQL"
  ip_protocol                  = "tcp"
  from_port                    = 5432
  to_port                      = 5432
  referenced_security_group_id = aws_security_group.db.id
}

resource "aws_vpc_security_group_egress_rule" "tasks_to_cache" {
  count = local.elasticache ? 1 : 0

  security_group_id            = aws_security_group.tasks.id
  description                  = "Redis (ElastiCache, TLS)"
  ip_protocol                  = "tcp"
  from_port                    = 6379
  to_port                      = 6379
  referenced_security_group_id = aws_security_group.cache[0].id
}

# ---- RDS and ElastiCache: from the tasks only ----

resource "aws_security_group" "db" {
  name        = "${local.name}-db"
  description = "RDS PostgreSQL: from the API tasks only"
  vpc_id      = aws_vpc.this.id
}

resource "aws_vpc_security_group_ingress_rule" "db_from_tasks" {
  security_group_id            = aws_security_group.db.id
  description                  = "PostgreSQL from the API tasks"
  ip_protocol                  = "tcp"
  from_port                    = 5432
  to_port                      = 5432
  referenced_security_group_id = aws_security_group.tasks.id
}

resource "aws_security_group" "cache" {
  count = local.elasticache ? 1 : 0

  name        = "${local.name}-cache"
  description = "ElastiCache: from the API tasks only"
  vpc_id      = aws_vpc.this.id
}

resource "aws_vpc_security_group_ingress_rule" "cache_from_tasks" {
  count = local.elasticache ? 1 : 0

  security_group_id            = aws_security_group.cache[0].id
  description                  = "Redis from the API tasks"
  ip_protocol                  = "tcp"
  from_port                    = 6379
  to_port                      = 6379
  referenced_security_group_id = aws_security_group.tasks.id
}
