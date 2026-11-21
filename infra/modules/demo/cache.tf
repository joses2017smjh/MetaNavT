# Redis for the answer route's shared quotas, daily budget and answer cache (app/api/limits.py). With
# redis_mode = "elasticache": one node in the private subnets with TLS in transit, so REDIS_URL is rediss://. With
# "sidecar", Redis runs inside the API task instead (ecs.tf) and none of this is created.

resource "aws_elasticache_subnet_group" "this" {
  count = local.elasticache ? 1 : 0

  name        = local.name
  description = "${local.name}: private subnets"
  subnet_ids  = aws_subnet.private[*].id
}

resource "aws_elasticache_replication_group" "this" {
  count = local.elasticache ? 1 : 0

  replication_group_id = local.name
  description          = "${local.name}: answer quotas, daily budget and answer cache"
  engine               = var.cache_engine
  engine_version       = var.cache_engine_version
  node_type            = var.cache_node_type
  num_cache_clusters   = 1
  port                 = 6379

  subnet_group_name  = aws_elasticache_subnet_group.this[0].name
  security_group_ids = [aws_security_group.cache[0].id]

  transit_encryption_enabled = true
  at_rest_encryption_enabled = true
  automatic_failover_enabled = false
  multi_az_enabled           = false
  snapshot_retention_limit   = 0
  apply_immediately          = true
}

locals {
  redis_url = local.elasticache ? "rediss://${aws_elasticache_replication_group.this[0].primary_endpoint_address}:6379/0" : "redis://127.0.0.1:6379/0"
}
