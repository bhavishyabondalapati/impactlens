# ECR = a private registry for container images. All Lambdas run one image (no PyTorch:
# onnxruntime, an ONNX MiniLM embedder, faiss-cpu, Shapely, rasterio); each sets its own CMD.
resource "aws_ecr_repository" "app" {
  name                 = var.project
  image_tag_mutability = "MUTABLE"
  force_delete         = true # `make destroy` removes the images too

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Keep the 3 newest images; older ones would only cost storage.
resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep the last 3 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 3
      }
      action = { type = "expire" }
    }]
  })
}
