group "default" {
  targets = ["cpu"]
}

target "cpu" {
  context    = "."
  dockerfile = "Dockerfile"
  tags       = ["cdaprod/stock-metadata:cpu"]
}

target "gpu" {
  context    = "."
  dockerfile = "Dockerfile.gpu"
  tags       = ["cdaprod/stock-metadata:gpu"]
}
