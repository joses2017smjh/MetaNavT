# RDS PostgreSQL in the private subnets, encrypted, never publicly accessible. The app's migrations create the vector
# extension (pgvector) and the tables; the API runs no DDL.

resource "aws_db_subnet_group" "this" {
  name        = local.name
  description = "${local.name}: private subnets"
  subnet_ids  = aws_subnet.private[*].id
}

# ---- The two secrets the task reads at start (its execution role, ecs.tf) ----

# The database password: random, generated during the apply that sets it, written to RDS and to this secret through
# write-only arguments, so it is in neither the state nor a plan. Bump var.db_password_version to set a new one.
ephemeral "random_password" "db" {
  length  = 40
  special = false
}

resource "aws_secretsmanager_secret" "db_password" {
  name                    = "${local.name}/db-password"
  description             = "PGPASSWORD of the ${local.name} API and migrate tasks"
  recovery_window_in_days = var.environment == "production" ? 7 : 0
}

resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id                = aws_secretsmanager_secret.db_password.id
  secret_string_wo         = ephemeral.random_password.db.result
  secret_string_wo_version = var.db_password_version
}

# ANTHROPIC_API_KEY: created empty; the owner stores the key (doc/deploy.md), then sets anthropic_key_stored.
resource "aws_secretsmanager_secret" "anthropic_api_key" {
  name                    = "${local.name}/anthropic-api-key"
  description             = "ANTHROPIC_API_KEY of the ${local.name} API (a key with a spend limit set in the Anthropic Console)"
  recovery_window_in_days = var.environment == "production" ? 7 : 0
}

# ---- The database ----

resource "aws_db_instance" "this" {
  identifier     = local.name
  engine         = "postgres"
  engine_version = var.db_engine_version
  instance_class = var.db_instance_class

  allocated_storage = var.db_allocated_storage_gb
  storage_type      = "gp3"
  storage_encrypted = true

  db_name             = local.db_name
  username            = local.db_user
  password_wo         = ephemeral.random_password.db.result
  password_wo_version = var.db_password_version

  db_subnet_group_name   = aws_db_subnet_group.this.name
  vpc_security_group_ids = [aws_security_group.db.id]
  publicly_accessible    = false
  multi_az               = false

  backup_retention_period    = var.db_backup_retention_days
  copy_tags_to_snapshot      = true
  auto_minor_version_upgrade = true

  deletion_protection       = var.deletion_protection
  skip_final_snapshot       = !var.final_snapshot
  final_snapshot_identifier = var.final_snapshot ? "${local.name}-final" : null
  delete_automated_backups  = true
}
