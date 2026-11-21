# tflint for every module under infra/ (`tflint --init` once, then `tflint --recursive` from infra/; CI's infra.yml
# runs the same with the pinned tflint). The bundled Terraform rules use their recommended preset; the AWS rules check
# instance types, engines and other values against the provider's lists, offline (no deep check: no AWS calls).
config {
  # also lint the local module through the environments' calls, with the values they pass
  call_module_type = "local"
}

plugin "terraform" {
  enabled = true
  preset  = "recommended"
}

plugin "aws" {
  enabled = true
  version = "0.50.0"
  source  = "github.com/terraform-linters/tflint-ruleset-aws"
}
