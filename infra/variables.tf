variable "region" {
  description = "AWS region for everything."
  type        = string
  default     = "us-east-2"

  validation {
    condition     = var.region == "us-east-2"
    error_message = "This project deploys to us-east-2 only."
  }
}

variable "project" {
  description = "Name prefix for resources."
  type        = string
  default     = "impactlens"
}

variable "image_uri" {
  description = "The Lambda container image, pinned by digest (repo@sha256:...). `make deploy` builds, pushes and passes it; it is only empty while the ECR repository is being created."
  type        = string
  default     = ""
}

variable "anthropic_model" {
  description = "Claude model used by the Report step."
  type        = string
  default     = "claude-haiku-4-5"
}

variable "budget_usd" {
  description = "Hard cost cap for the Claude calls of one run, in USD. The Report step stops before a call could pass it."
  type        = number
  default     = 0.10
}

variable "max_retries" {
  description = "Validation retries after the first draft before a run is flagged for human review."
  type        = number
  default     = 2
}

variable "lambda_memory_mb" {
  description = "Memory per Lambda in MB (it also sets the CPU share). This account's limit is 3,008 MB per function."
  type        = map(number)
  default = {
    start       = 256
    fetch       = 2048
    detect      = 3008 # the ONNX models: as much CPU as the account allows
    geo_analyze = 3008
    retrieve    = 2048
    report      = 512
    validate    = 512
    finish      = 512
  }

  validation {
    condition     = alltrue([for m in values(var.lambda_memory_mb) : m >= 128 && m <= 3008])
    error_message = "Each Lambda's memory must be 128-3008 MB: this account's limit is 3,008 MB (a higher value fails at deploy)."
  }

  validation {
    condition     = length(setsubtract(["start", "fetch", "detect", "geo_analyze", "retrieve", "report", "validate", "finish"], keys(var.lambda_memory_mb))) == 0
    error_message = "lambda_memory_mb needs an entry for every function."
  }
}
