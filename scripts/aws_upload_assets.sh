#!/usr/bin/env bash
# Uploads the trained ONNX models, the SOP FAISS index + chunks and the ONNX embedder to the
# PRIVATE bucket (models/ and index/). The bucket blocks all public access. The SOP chunks are
# text from the public documents listed in data/raw/sops/sources.md, kept private for
# retrieval and not redistributed.
set -euo pipefail
cd "$(dirname "$0")/.."

BUCKET=$(terraform -chdir=infra output -raw bucket_name)
FLOODS=$(ls -d build/models_*/floods_r2 | tail -1)

put() {  # put <local file> <key>
  if [ ! -f "$1" ]; then
    echo "missing $1 (see docs/AWS.md, 'Before deploying')"
    exit 1
  fi
  aws s3 cp "$1" "s3://$BUCKET/$2" --only-show-errors
  echo "  $1 -> s3://$BUCKET/$2"
}

put "$FLOODS/floods.onnx" models/floods/floods.onnx
put "$FLOODS/metrics.json" models/floods/metrics.json
put build/models/oscd/oscd.onnx models/oscd/oscd.onnx
put build/models/oscd/metrics.json models/oscd/metrics.json
put data/processed/rag/sops.faiss index/sops.faiss
put data/processed/rag/chunks.jsonl index/chunks.jsonl
put build/embedder/minilm.onnx index/embedder/minilm.onnx
put build/embedder/tokenizer.json index/embedder/tokenizer.json
