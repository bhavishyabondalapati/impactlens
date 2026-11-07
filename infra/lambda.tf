# Lambda = run code on demand, pay per millisecond, $0 when idle. Eight functions, one container
# image (ecr.tf), arm64 (Graviton: ~20% cheaper per second than x86, and the Mac builds it
# natively). Each function runs one handler from src/impactlens/aws/handlers.py.
#
# Memory also sets the CPU share (1,769 MB = 1 vCPU). This account allows at most 3,008 MB per
# function; var.lambda_memory_mb (variables.tf) enforces that before anything is created.

locals {
  # The seven workflow steps. name = timeout (s), /tmp size (MB); memory is var.lambda_memory_mb
  functions = {
    fetch       = { timeout = 600, tmp = 2048 }
    detect      = { timeout = 900, tmp = 2048 }
    geo_analyze = { timeout = 600, tmp = 2048 }
    retrieve    = { timeout = 120, tmp = 1024 }
    report      = { timeout = 600, tmp = 512 }
    validate    = { timeout = 60, tmp = 512 }
    finish      = { timeout = 60, tmp = 512 }
  }
  # every function (the 7 steps + the starter) gets its own IAM role (iam.tf)
  roles = toset(concat(keys(local.functions), ["start"]))
}

# Keep logs 14 days so they don't pile up.
resource "aws_cloudwatch_log_group" "lambda" {
  for_each          = local.functions
  name              = "/aws/lambda/${var.project}-${replace(each.key, "_", "-")}"
  retention_in_days = 14
}

resource "aws_lambda_function" "step" {
  for_each      = local.functions
  function_name = "${var.project}-${replace(each.key, "_", "-")}"
  role          = aws_iam_role.lambda[each.key].arn
  package_type  = "Image"
  image_uri     = var.image_uri
  architectures = ["arm64"]
  memory_size   = var.lambda_memory_mb[each.key]
  timeout       = each.value.timeout

  ephemeral_storage {
    size = each.value.tmp
  }

  image_config {
    command = ["impactlens.aws.handlers.${each.key}"]
  }

  environment {
    variables = merge(
      { BUCKET = aws_s3_bucket.data.bucket, MPLCONFIGDIR = "/tmp/mpl" },
      each.key == "report" ? { API_KEY_PARAMETER = aws_ssm_parameter.anthropic_api_key.name } : {},
      contains(["report", "finish"], each.key) ? {
        MODEL      = var.anthropic_model
        BUDGET_USD = tostring(var.budget_usd)
      } : {},
    )
  }

  lifecycle {
    precondition {
      condition     = can(regex("@sha256:[0-9a-f]{64}$", var.image_uri))
      error_message = "image_uri must be an image pinned by digest; run `make deploy`, which builds and pushes it."
    }
  }

  depends_on = [aws_cloudwatch_log_group.lambda]
}

# The start function needs the state machine ARN. It is set here, after both exist, because
# the state machine also needs the function ARNs (Terraform would see a cycle otherwise).
resource "aws_lambda_function" "start" {
  function_name = "${var.project}-starter"
  role          = aws_iam_role.lambda["start"].arn
  package_type  = "Image"
  image_uri     = var.image_uri
  architectures = ["arm64"]
  memory_size   = var.lambda_memory_mb["start"]
  timeout       = 30

  image_config {
    command = ["impactlens.aws.handlers.start"]
  }

  environment {
    variables = {
      BUCKET            = aws_s3_bucket.data.bucket
      STATE_MACHINE_ARN = aws_sfn_state_machine.pipeline.arn
    }
  }

  lifecycle {
    precondition {
      condition     = can(regex("@sha256:[0-9a-f]{64}$", var.image_uri))
      error_message = "image_uri must be an image pinned by digest; run `make deploy`."
    }
  }

  depends_on = [aws_cloudwatch_log_group.starter]
}

resource "aws_cloudwatch_log_group" "starter" {
  name              = "/aws/lambda/${var.project}-starter"
  retention_in_days = 14
}

# The requests queue triggers the start function, one message at a time, at most 2 at once.
resource "aws_lambda_event_source_mapping" "requests" {
  event_source_arn = aws_sqs_queue.requests.arn
  function_name    = aws_lambda_function.start.arn
  batch_size       = 1

  scaling_config {
    maximum_concurrency = 2
  }
}
