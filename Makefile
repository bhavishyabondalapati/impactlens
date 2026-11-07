# Common commands. `make kaggle-run` asks before every push / upload / GPU run.

.PHONY: test lint kaggle-run

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

kaggle-run:  ## push code, upload data if changed, run the notebook on a Kaggle GPU, fetch results
	scripts/kaggle_run.sh

# --------------------------------------------------------------------------- AWS (Phase 6)
# Same pattern as LexIntake: Terraform (infra/) + these targets. Profile `sky`, us-east-2 only.
# Run `aws login --profile sky` first. Details: docs/AWS.md.
#
#   make deploy    build + push the Lambda image, create everything, upload models + index
#   make set-key   store the Anthropic key in SSM (asks for it; never in files or state)
#   make demo      upload the Valencia request, wait, download the reports
#   make destroy   delete everything Terraform created
#   make local     the whole AWS pipeline on the laptop, inside the Lambda image (no AWS)
PROFILE ?= sky
REGION  := us-east-2
TF       = terraform -chdir=infra
IMAGE    = impactlens-lambda
# Terraform can't read `aws login` credentials directly, so export them for each command.
AWS_ENV  = eval "$$(aws configure export-credentials --profile $(PROFILE) --format env)" && \
           export AWS_REGION=$(REGION) AWS_DEFAULT_REGION=$(REGION)
IMAGE_URI = $$(cat build/image_uri.txt 2>/dev/null)

.PHONY: init fmt validate image ecr push apply assets deploy plan destroy set-key demo local

init:
	$(TF) init -input=false

fmt:
	$(TF) fmt

validate: init
	$(TF) validate

# The Lambda image (arm64, no PyTorch). Built locally; nothing leaves the laptop.
image:
	docker build --platform linux/arm64 -f infra/docker/Dockerfile -t $(IMAGE):latest .

# Deploy step 1: the ECR repository must exist before an image can be pushed to it.
ecr: init
	$(AWS_ENV) && $(TF) apply -target=aws_ecr_repository.app -target=aws_ecr_lifecycle_policy.app

# Deploy step 2: push the image. Its digest pins exactly this build (build/image_uri.txt).
push: image
	$(AWS_ENV) && scripts/aws_push_image.sh $(IMAGE)

# Deploy step 3: everything else. Terraform shows the plan and asks you to type "yes".
apply: init
	$(AWS_ENV) && $(TF) apply -var image_uri=$(IMAGE_URI)

# Deploy step 4: models, FAISS index and ONNX embedder -> the private bucket.
assets:
	$(AWS_ENV) && scripts/aws_upload_assets.sh

deploy: ecr push apply assets
	@echo "Deployed. Next: make set-key (once), then make demo."

# Shows what would change. Creates nothing.
plan: init
	$(AWS_ENV) && $(TF) plan -var image_uri=$(IMAGE_URI)

# Deletes everything Terraform created, including the bucket's files and the images.
destroy: init
	$(AWS_ENV) && $(TF) destroy -var image_uri=$(IMAGE_URI)

set-key:
	$(AWS_ENV) && scripts/aws_set_key.sh

# Upload the Valencia request and wait for the reports (about 3-5 minutes).
demo:
	$(AWS_ENV) && scripts/aws_demo.sh

# The whole AWS state machine on the laptop, inside the Lambda image, against a folder
# (build/aws_local/). No AWS account and no Anthropic calls (fake LLM).
local: image
	scripts/aws_local.sh
