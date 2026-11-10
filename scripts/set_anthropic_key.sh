#!/usr/bin/env bash
# Sets ANTHROPIC_API_KEY in a .env file without the key ever appearing on screen, in the shell
# history, in a command line (`ps`), or in this script's output.
#
#   scripts/set_anthropic_key.sh path/to/.env
#
# - The key is typed at a hidden prompt (read -rs).
# - It reaches awk through an environment variable, not as an argument.
# - The existing ANTHROPIC_API_KEY line (with or without `export`) is replaced; if there is
#   none, one is added. Every other line is kept as it is.
# - The new file is written next to the old one with owner-only permissions (600), then moved
#   into place in one step, so a crash never leaves a half-written .env.
# - The only output is "updated <path>".
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 path/to/.env" >&2
  exit 2
fi
ENV_FILE=$1
DIR=$(dirname "$ENV_FILE")
[[ -d "$DIR" ]] || { echo "no such folder: $DIR" >&2; exit 1; }

read -rsp "Anthropic API key for $ENV_FILE (input hidden): " NEW_KEY
echo >&2
if [[ "$NEW_KEY" != sk-ant-* || "$NEW_KEY" =~ [[:space:]] ]]; then
  unset NEW_KEY
  echo "that does not look like an Anthropic key (should start with sk-ant-); nothing changed" >&2
  exit 1
fi

umask 077                                   # temp file readable by you only
TMP=$(mktemp "$DIR/.env.tmp.XXXXXX")
trap 'rm -f "$TMP"' EXIT                    # removed on any error

[[ -f "$ENV_FILE" ]] || : > "$ENV_FILE"     # create an empty .env if there is none
NEW_KEY="$NEW_KEY" awk '
  /^[[:space:]]*(export[[:space:]]+)?ANTHROPIC_API_KEY[[:space:]]*=/ {
    if (!done) print "ANTHROPIC_API_KEY=" ENVIRON["NEW_KEY"]
    done = 1
    next
  }
  { print }
  END { if (!done) print "ANTHROPIC_API_KEY=" ENVIRON["NEW_KEY"] }
' "$ENV_FILE" > "$TMP"
unset NEW_KEY

chmod 600 "$TMP"
mv "$TMP" "$ENV_FILE"
trap - EXIT
echo "updated $ENV_FILE"
