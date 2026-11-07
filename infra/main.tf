# Provider setup. State is local (terraform.tfstate, gitignored) to keep the project simple.
# Region is us-east-2 only: us-east-1 is blocked for this account.

terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.region

  # Every resource gets these tags, so you can find (and cost-track) everything in the console.
  default_tags {
    tags = {
      Project   = var.project
      ManagedBy = "terraform"
    }
  }
}

# Who am I? Used to make the S3 bucket name globally unique.
data "aws_caller_identity" "current" {}

locals {
  bucket_name = "${var.project}-${data.aws_caller_identity.current.account_id}"
}
