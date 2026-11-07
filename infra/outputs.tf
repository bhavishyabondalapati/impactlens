output "bucket_name" {
  value = aws_s3_bucket.data.bucket
}

output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

output "state_machine_arn" {
  value = aws_sfn_state_machine.pipeline.arn
}

output "dlq_url" {
  value = aws_sqs_queue.dlq.url
}

output "anthropic_key_parameter" {
  value = aws_ssm_parameter.anthropic_api_key.name
}
