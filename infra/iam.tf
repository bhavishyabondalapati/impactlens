# IAM = who may do what. Every Lambda gets its own role (least privilege): it can touch only the
# S3 prefixes, queue, parameter and log group that its step actually needs.

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  for_each           = local.roles
  name               = "${var.project}-${replace(each.key, "_", "-")}-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

locals {
  log_group_arns = merge(
    { for k, g in aws_cloudwatch_log_group.lambda : k => g.arn },
    { start = aws_cloudwatch_log_group.starter.arn },
  )
  bucket = aws_s3_bucket.data.arn

  # step -> S3 prefixes it may read and write
  s3_access = {
    fetch       = { read = ["requests/"], write = ["layers/"] }
    detect      = { read = ["layers/", "models/"], write = ["layers/"] }
    geo_analyze = { read = ["requests/", "layers/", "osm/"], write = ["layers/", "osm/"] }
    retrieve    = { read = ["index/"], write = ["reports/"] }
    report      = { read = ["layers/", "reports/"], write = ["reports/"] }
    validate    = { read = ["layers/", "reports/"], write = ["reports/"] }
    finish      = { read = ["layers/", "reports/"], write = ["reports/"] }
  }
}

# Writing logs: only to the function's own log group.
data "aws_iam_policy_document" "logs" {
  for_each = local.roles
  statement {
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${local.log_group_arns[each.key]}:*"]
  }
}

resource "aws_iam_role_policy" "logs" {
  for_each = local.roles
  name     = "logs"
  role     = aws_iam_role.lambda[each.key].id
  policy   = data.aws_iam_policy_document.logs[each.key].json
}

data "aws_iam_policy_document" "s3" {
  for_each = local.s3_access
  statement {
    actions   = ["s3:GetObject"]
    resources = [for p in each.value.read : "${local.bucket}/${p}*"]
  }
  statement {
    actions   = ["s3:PutObject"]
    resources = [for p in each.value.write : "${local.bucket}/${p}*"]
  }
  # Without ListBucket, S3 answers "403" instead of "404" for a missing key. GeoAnalyze needs
  # the 404 to tell "no OSM cache for this AOI yet" from a real error.
  dynamic "statement" {
    for_each = each.key == "geo_analyze" ? [1] : []
    content {
      actions   = ["s3:ListBucket"]
      resources = [local.bucket]
      condition {
        test     = "StringLike"
        variable = "s3:prefix"
        values   = ["osm/*"]
      }
    }
  }
}

resource "aws_iam_role_policy" "s3" {
  for_each = local.s3_access
  name     = "s3"
  role     = aws_iam_role.lambda[each.key].id
  policy   = data.aws_iam_policy_document.s3[each.key].json
}

# report: read the Anthropic key (decryption with the AWS-managed aws/ssm key is allowed for
# account principals that may read the parameter).
resource "aws_iam_role_policy" "report_key" {
  name = "anthropic-key"
  role = aws_iam_role.lambda["report"].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "ssm:GetParameter"
      Resource = aws_ssm_parameter.anthropic_api_key.arn
    }]
  })
}

# start: read the requests queue, start the workflow.
resource "aws_iam_role_policy" "start" {
  name = "start"
  role = aws_iam_role.lambda["start"].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes"]
        Resource = aws_sqs_queue.requests.arn
      },
      {
        Effect   = "Allow"
        Action   = "states:StartExecution"
        Resource = aws_sfn_state_machine.pipeline.arn
      },
    ]
  })
}

# --- Step Functions: invoke the seven step Lambdas, send failures to the DLQ ---

data "aws_iam_policy_document" "sfn_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["states.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sfn" {
  name               = "${var.project}-state-machine"
  assume_role_policy = data.aws_iam_policy_document.sfn_assume.json
}

resource "aws_iam_role_policy" "sfn" {
  name = "pipeline"
  role = aws_iam_role.sfn.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "lambda:InvokeFunction"
        Resource = [for f in aws_lambda_function.step : f.arn]
      },
      {
        Effect   = "Allow"
        Action   = "sqs:SendMessage"
        Resource = aws_sqs_queue.dlq.arn
      },
    ]
  })
}
