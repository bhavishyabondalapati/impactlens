#!/usr/bin/env bash
# Runs the AWS state machine locally inside the Lambda image (no PyTorch), against a folder that
# stands in for the bucket (build/aws_local/). Real Earth Search fetch, ONNX inference, Overpass
# (cached after the first run), Shapely rollup and ONNX retrieval; a fake LLM, so no API cost.
set -euo pipefail
cd "$(dirname "$0")/.."

STORE=build/aws_local
uv run python scripts/make_request.py "$STORE/request.json"
docker run --rm --platform linux/arm64 -v "$PWD":/repo:ro -v "$PWD/$STORE":/store -w /repo \
  -e PYTHONPATH=/var/task -e MPLCONFIGDIR=/tmp/mpl --entrypoint python impactlens-lambda:latest \
  -m impactlens.cli aws-local --store /store --request /store/request.json --run-id local-demo \
  --seed --fake-llm --template /repo/infra/statemachine/pipeline.asl.json.tftpl
