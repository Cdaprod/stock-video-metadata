group "default" {
  targets = ["gpu"]
}

target "cpu" {
  context    = "."
  dockerfile = "Dockerfile"
  tags       = ["cdaprod/enrich-api-dev:cpu"]
}

target "gpu" {
  context    = "."
  dockerfile = "Dockerfile.gpu"
  tags       = ["cdaprod/enrich-api-dev:gpu"]
}
