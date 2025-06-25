variable "TAG" {
  default = "dev"
}

variable "VERSION" {
  default = "0.1.0"
}

variable "PLATFORM" {
  default = "linux/amd64"
}

variable "PROFILE" {
  default = "gpu"
}

variable "DOCKERFILE" {
  default = "Dockerfile.gpu"
}

variable "DOCKERFILE_CPU" {
  default = "Dockerfile"
}

variable "IMAGE_BASE" {
  default = "cdaprod/enrich-api-dev"
}

variable "GIT_COMMIT" {
  default = "unknown"
}

variable "BUILD_DATE" {
  default = "2025-06-25"
}

# ----------- CPU TARGET -------------
target "cpu" {
  context    = "."
  dockerfile = "${DOCKERFILE_CPU}"
  tags       = ["${IMAGE_BASE}:cpu", "${IMAGE_BASE}:cpu-${TAG}"]
  args = {
    VERSION = "${VERSION}"
    PROFILE = "cpu"
    BUILD_DATE = "${BUILD_DATE}"
    GIT_COMMIT = "${GIT_COMMIT}"
  }
  labels = {
    "com.cdaprod.project"           = "Blackbox Video Enrichment API"
    "com.cdaprod.version"           = "${VERSION}"
    "com.cdaprod.env"               = "development"
    "org.opencontainers.image.source" = "https://github.com/Cdaprod/stock-video-metadata"
    "org.opencontainers.image.revision" = "${GIT_COMMIT}"
    "org.opencontainers.image.created"  = "${BUILD_DATE}"
    "org.opencontainers.image.description" = "FastAPI app for video enrichment (CPU, ARM64/AMD64)"
  }
  platforms = ["linux/amd64", "linux/arm64"]
  # BuildKit SBOM example--comment out if not needed
  # output = [ "type=image,name=${IMAGE_BASE}:cpu-${TAG},push=false" ]
}

# ----------- GPU TARGET -------------
target "gpu" {
  context    = "."
  dockerfile = "${DOCKERFILE}"
  tags       = ["${IMAGE_BASE}:gpu", "${IMAGE_BASE}:gpu-${TAG}"]
  args = {
    VERSION = "${VERSION}"
    PROFILE = "gpu"
    BUILD_DATE = "${BUILD_DATE}"
    GIT_COMMIT = "${GIT_COMMIT}"
  }
  labels = {
    "com.cdaprod.project"           = "Blackbox Video Enrichment API"
    "com.cdaprod.version"           = "${VERSION}"
    "com.cdaprod.env"               = "development"
    "org.opencontainers.image.source" = "https://github.com/Cdaprod/stock-video-metadata"
    "org.opencontainers.image.revision" = "${GIT_COMMIT}"
    "org.opencontainers.image.created"  = "${BUILD_DATE}"
    "org.opencontainers.image.description" = "FastAPI app for video enrichment (GPU, CUDA)"
  }
  platforms = ["linux/amd64"]
  # output = [ "type=image,name=${IMAGE_BASE}:gpu-${TAG},push=false" ]
}

# ----------- SBOM Target (Optional) -------------
target "sbom" {
  inherits = ["gpu"]
  output = ["type=sbom,name=sbom/spdx.json"]
  args = {
    SBOM = "true"
  }
  tags = [] # Not producing a docker image, just the SBOM file
}

# ----------- GROUPS ---------------
group "default" {
  targets = ["gpu"]
}

group "all" {
  targets = ["cpu", "gpu"]
}

group "sbom" {
  targets = ["sbom"]
}