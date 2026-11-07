#!/usr/bin/env bash
# Stores the Anthropic API key in SSM Parameter Store (SecureString). The key is typed at a
# hidden prompt: it is never written to a file, to Terraform state or to the shell history.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=$(terraform -chdir=infra output -raw anthropic_key_parameter)
read -rsp "Anthropic API key (input hidden): " KEY
echo
if [[ "$KEY" != sk-ant-* ]]; then
  echo "that does not look like an Anthropic key (sk-ant-...)"
  exit 1
fi
aws ssm put-parameter --name "$NAME" --type SecureString --overwrite --value "$KEY" >/dev/null
unset KEY
echo "stored in $NAME"
