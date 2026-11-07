# S3 = object storage. One private bucket holds everything (layout in src/impactlens/aws/store.py):
#   requests/  models/  index/  osm/  layers/  reports/

resource "aws_s3_bucket" "data" {
  bucket = local.bucket_name

  # Lets `make destroy` delete the bucket even when it still holds files.
  force_destroy = true
}

# Nothing in this bucket is ever public.
resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# No ACLs at all: access is decided only by IAM policies.
resource "aws_s3_bucket_ownership_controls" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Refuse plain-HTTP requests.
resource "aws_s3_bucket_policy" "data" {
  bucket = aws_s3_bucket.data.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyInsecureTransport"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [aws_s3_bucket.data.arn, "${aws_s3_bucket.data.arn}/*"]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })
  depends_on = [aws_s3_bucket_public_access_block.data]
}

# Intermediate rasters are big and only needed while a run is going; requests are small but
# pointless after a while. Reports, models, index and the OSM cache are kept.
resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "expire-layers"
    status = "Enabled"
    filter {
      prefix = "layers/"
    }
    expiration {
      days = 30
    }
  }

  rule {
    id     = "expire-requests"
    status = "Enabled"
    filter {
      prefix = "requests/"
    }
    expiration {
      days = 90
    }
  }
}

# A new requests/*.json -> a message on the requests queue (sqs.tf) -> the start Lambda.
resource "aws_s3_bucket_notification" "requests" {
  bucket = aws_s3_bucket.data.id

  queue {
    queue_arn     = aws_sqs_queue.requests.arn
    events        = ["s3:ObjectCreated:*"]
    filter_prefix = "requests/"
    filter_suffix = ".json"
  }

  depends_on = [aws_sqs_queue_policy.requests]
}
