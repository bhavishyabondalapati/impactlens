# One image for all ImpactLens Lambdas (arm64). Each function picks its handler with its own
# CMD in infra/lambda.tf, e.g. impactlens.aws.handlers.detect.
#
# Build from the repo root:  docker build --platform linux/arm64 -f infra/docker/Dockerfile .
# (`make image` does this.) The Lambda base image ships the runtime interface and boto3.
FROM public.ecr.aws/lambda/python:3.12

# The rasterio wheel bundles GDAL but links the system XML parser, which the minimal base lacks.
RUN dnf install -y expat && dnf clean all

COPY infra/docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Only the package source; no data, models, or secrets (see .dockerignore). Models and the
# FAISS index are read from the private S3 bucket at run time.
COPY src/impactlens ${LAMBDA_TASK_ROOT}/impactlens

CMD ["impactlens.aws.handlers.start"]
