#!/usr/bin/env bash
# Pushes the local Lambda image to ECR and writes its digest-pinned URI to build/image_uri.txt.
# Usage: scripts/aws_push_image.sh <local image name>   (AWS credentials in the environment)
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="$1"
REPO=$(terraform -chdir=infra output -raw ecr_repository_url)
REGISTRY="${REPO%%/*}"
TAG="$(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty)"

aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY"
docker tag "$IMAGE:latest" "$REPO:$TAG"
docker push "$REPO:$TAG" | tail -1
DIGEST=$(aws ecr describe-images --repository-name "${REPO#*/}" --image-ids imageTag="$TAG" \
  --query 'imageDetails[0].imageDigest' --output text)
mkdir -p build
echo "$REPO@$DIGEST" > build/image_uri.txt
echo "image: $(cat build/image_uri.txt)"
