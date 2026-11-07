# Step Functions = a workflow engine. The state machine (statemachine/pipeline.asl.json.tftpl)
# calls the step Lambdas in order, retries AWS hiccups with backoff, loops Report <-> Validate
# at most var.max_retries times, and sends any failed run to the DLQ. "Standard" workflows keep
# a full execution history you can click through in the console.
resource "aws_sfn_state_machine" "pipeline" {
  name     = "${var.project}-pipeline"
  role_arn = aws_iam_role.sfn.arn
  type     = "STANDARD"

  definition = templatefile("${path.module}/statemachine/pipeline.asl.json.tftpl", {
    fetch_arn       = aws_lambda_function.step["fetch"].arn
    detect_arn      = aws_lambda_function.step["detect"].arn
    geo_analyze_arn = aws_lambda_function.step["geo_analyze"].arn
    retrieve_arn    = aws_lambda_function.step["retrieve"].arn
    report_arn      = aws_lambda_function.step["report"].arn
    validate_arn    = aws_lambda_function.step["validate"].arn
    finish_arn      = aws_lambda_function.step["finish"].arn
    dlq_url         = aws_sqs_queue.dlq.url
    max_retries     = var.max_retries
  })
}
