######## Docker Compose Bake File ########
## For GPU: `docker buildx bake auto --set PROFILE=gpu`
## For CPU: `docker buildx bake auto --set PROFILE=cpu --set PLATFORM=linux/arm64`
## Or Override in `.env`
##########################################

# ---------- VARIABLES ----------
variable "TAG"         { default = "dev" }
variable "VERSION"     { default = "0.1.0" }
variable "PROFILE"     { default = "gpu" } # set to "cpu" or "gpu"
variable "PLATFORM"    { default = "linux/amd64" }
variable "IMAGE_BASE"  { default = "cdaprod/enrich-api-dev" }
variable "GIT_COMMIT"  { default = "unknown" }
variable "BUILD_DATE"  { default = "2025-06-25" }

# ---------- DOCKERFILE SELECTION ----------
variable "DOCKERFILE" {
  default = "${PROFILE == "gpu" ? "Dockerfile.gpu" : "Dockerfile"}"
}

# ---------- COMMON TARGET ----------
target "_common" {
  context    = "."
  dockerfile = "${DOCKERFILE}"
  args = {
    VERSION     = "${VERSION}"
    PROFILE     = "${PROFILE}"
    BUILD_DATE  = "${BUILD_DATE}"
    GIT_COMMIT  = "${GIT_COMMIT}"
  }
  labels = {
    "com.cdaprod.project"             = "Blackbox Video Enrichment API"
    "com.cdaprod.version"             = "${VERSION}"
    "com.cdaprod.env"                 = "development"
    "org.opencontainers.image.source" = "https://github.com/Cdaprod/stock-video-metadata"
    "org.opencontainers.image.revision" = "${GIT_COMMIT}"
    "org.opencontainers.image.created"  = "${BUILD_DATE}"
    "org.opencontainers.image.description" = "${PROFILE == "gpu" ? "FastAPI app for video enrichment (GPU, CUDA)" : "FastAPI app for video enrichment (CPU, ARM64/AMD64)"}"
  }
}

# ---------- CPU TARGET ----------
target "cpu" {
  inherits   = ["_common"]
  dockerfile = "Dockerfile"
  args = { PROFILE = "cpu" }
  tags = [
    "${IMAGE_BASE}:cpu",
    "${IMAGE_BASE}:cpu-${TAG}"
  ]
  platforms = ["linux/amd64", "linux/arm64"]
}

# ---------- GPU TARGET ----------
target "gpu" {
  inherits   = ["_common"]
  dockerfile = "Dockerfile.gpu"
  args = { PROFILE = "gpu" }
  tags = [
    "${IMAGE_BASE}:gpu",
    "${IMAGE_BASE}:gpu-${TAG}"
  ]
  platforms = ["linux/amd64"]
}

# ---------- AUTO TARGET (based on PROFILE) ----------
target "auto" {
  inherits = ["_common"]
  tags = [
    "${IMAGE_BASE}:${PROFILE}",
    "${IMAGE_BASE}:${PROFILE}-${TAG}",
    "${IMAGE_BASE}:latest"
  ]
  platforms = ["${PLATFORM}"]
}

# ---------- SBOM TARGET ----------
target "sbom" {
  inherits = ["auto"]
  output = ["type=sbom,name=sbom/spdx.json"]
  args = { SBOM = "true" }
  tags = [] # no image produced
}

# ---------- GROUPS ----------
group "default"  { targets = ["auto"] }
group "all"      { targets = ["cpu", "gpu"] }
group "sbom"     { targets = ["sbom"] }