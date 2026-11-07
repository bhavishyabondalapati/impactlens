# SSM Parameter Store = AWS's place for config and secrets. A SecureString is encrypted with KMS
# (the AWS-managed aws/ssm key, free) and only readable by roles that are allowed to.
#
# Terraform creates the parameter with a PLACEHOLDER and ignores later changes to the value, so
# the real API key is never in this repo or in terraform.tfstate. After `make deploy`, set it
# once with `make set-key` (asks for the key without echoing it).
resource "aws_ssm_parameter" "anthropic_api_key" {
  name        = "/${var.project}/anthropic_api_key"
  description = "Anthropic API key for the Report Lambda. Set manually, not by Terraform."
  type        = "SecureString"
  value       = "REPLACE_ME"

  lifecycle {
    ignore_changes = [value]
  }
}
