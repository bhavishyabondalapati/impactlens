# SQS = message queues.
#
# requests: S3 drops a message here for every new requests/*.json. The start Lambda reads it
#           and starts one Step Functions run. A message that fails 3 times moves to the DLQ.
# dlq:      dead-letter queue. Also receives the run id + error when a workflow step fails
#           after its retries (statemachine/pipeline.asl.json.tftpl), so nothing fails silently.

resource "aws_sqs_queue" "dlq" {
  name                      = "${var.project}-dlq"
  message_retention_seconds = 1209600 # 14 days, the maximum
  sqs_managed_sse_enabled   = true
}

resource "aws_sqs_queue" "requests" {
  name                       = "${var.project}-requests"
  visibility_timeout_seconds = 180    # >= 6 x the start Lambda's 30 s timeout (AWS guidance)
  message_retention_seconds  = 345600 # 4 days
  sqs_managed_sse_enabled    = true

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.dlq.arn
    maxReceiveCount     = 3
  })
}

# Only our bucket may send to the requests queue.
resource "aws_sqs_queue_policy" "requests" {
  queue_url = aws_sqs_queue.requests.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "AllowBucketNotifications"
      Effect    = "Allow"
      Principal = { Service = "s3.amazonaws.com" }
      Action    = "sqs:SendMessage"
      Resource  = aws_sqs_queue.requests.arn
      Condition = {
        ArnEquals    = { "aws:SourceArn" = aws_s3_bucket.data.arn }
        StringEquals = { "aws:SourceAccount" = data.aws_caller_identity.current.account_id }
      }
    }]
  })
}
