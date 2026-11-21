variable "region" {
  description = "Region of the state bucket and of every environment (the environments' backend blocks name it too)."
  type        = string
  default     = "us-west-2"
}

variable "project" {
  description = "Prefix of every name; the deploy role's permissions are scoped to resources whose names start with it."
  type        = string
  default     = "metanavit"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,20}$", var.project))
    error_message = "project: lowercase letters, digits and hyphens, starting with a letter, at most 21 characters."
  }
}

variable "github_repository" {
  description = "owner/name of the GitHub repository whose workflows may assume the roles, exactly as GitHub spells it (the token's sub claim is compared case-sensitively)."
  type        = string
  default     = "joses2017smjh/MetaNavT"

  validation {
    condition     = can(regex("^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", var.github_repository))
    error_message = "github_repository: owner/name."
  }
}

variable "deploy_environments" {
  description = "GitHub environments whose jobs may assume the deploy role. Production only by default; add \"staging\" when the staging environment is used (doc/deploy.md, staging mode)."
  type        = list(string)
  default     = ["production"]

  validation {
    condition     = length(var.deploy_environments) > 0 && alltrue([for env in var.deploy_environments : contains(["production", "staging"], env)])
    error_message = "deploy_environments: production and/or staging."
  }
}

variable "budget_email" {
  description = "Address the monthly budget's alerts go to. Pass it on the command line or in a local tfvars file: never commit it."
  type        = string
  sensitive   = true

  validation {
    condition     = can(regex("^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", var.budget_email))
    error_message = "budget_email: an email address."
  }
}

variable "monthly_budget_usd" {
  description = "The account's monthly cost budget in USD (gross: Free Tier credits are not subtracted)."
  type        = number
  default     = 150

  validation {
    condition     = var.monthly_budget_usd > 0
    error_message = "monthly_budget_usd: more than zero."
  }
}

variable "budget_alert_percentages" {
  description = "Actual-spend thresholds, in percent of the monthly budget, that send an email."
  type        = list(number)
  default     = [50, 80, 100]
}

variable "state_noncurrent_version_days" {
  description = "Days an overwritten state file is kept (the bucket is versioned, so a bad apply can be rolled back)."
  type        = number
  default     = 90
}
