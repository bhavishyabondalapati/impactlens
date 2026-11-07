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
