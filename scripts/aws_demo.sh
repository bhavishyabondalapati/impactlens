#!/usr/bin/env bash
# Uploads the Valencia request to requests/, waits for the Step Functions run, downloads the
# reports to build/aws_reports/<run_id>/.   (AWS credentials must be in the environment)
set -euo pipefail
cd "$(dirname "$0")/.."

BUCKET=$(terraform -chdir=infra output -raw bucket_name)
SM=$(terraform -chdir=infra output -raw state_machine_arn)
RUN_ID="valencia-$(date +%Y%m%d-%H%M%S)"
REQ="build/requests/$RUN_ID.json"
mkdir -p build/requests

# The AOI and date windows of the local run; "event" sets the event name and date in the reports.
uv run python scripts/make_request.py "$REQ" --event
aws s3 cp "$REQ" "s3://$BUCKET/requests/$RUN_ID.json" --only-show-errors
echo "uploaded requests/$RUN_ID.json; waiting for the run (about 3-5 minutes) ..."

EXEC="${SM/:stateMachine:/:execution:}:$RUN_ID"
STATUS=PENDING
for _ in $(seq 1 120); do
  STATUS=$(aws stepfunctions describe-execution --execution-arn "$EXEC" \
    --query status --output text 2>/dev/null || echo PENDING)
  case "$STATUS" in
    SUCCEEDED | FAILED | TIMED_OUT | ABORTED) break ;;
  esac
  sleep 10
done
echo "Step Functions: $STATUS"
if [ "$STATUS" != "SUCCEEDED" ]; then
  echo "check the execution in the console and the DLQ: $(terraform -chdir=infra output -raw dlq_url)"
  exit 1
fi

OUT="build/aws_reports/$RUN_ID"
aws s3 cp "s3://$BUCKET/reports/$RUN_ID/" "$OUT/" --recursive --exclude "work/*" --only-show-errors
uv run python -c "
import json, sys
r = json.load(open(sys.argv[1]))
u = r['usage']
print(f\"status: {r['status']} | drafts per report: {r['attempts_per_report']} | \"
      f\"{u['calls']} Claude calls, \${u['cost_usd']:.4f} of \${u['budget_usd']:.2f} cap\")
" "$OUT/run.json"
echo "reports: $OUT/"
