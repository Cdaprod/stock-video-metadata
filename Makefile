# -------- User-tweakable defaults --------------------------------------------
TAG        ?= dev
VERSION    ?= 0.1.0
IMAGE_BASE ?= cdaprod/enrich-api-dev

# -------- Host / GPU autodetection -------------------------------------------
UNAME_S   := $(shell uname -s 2>/dev/null || echo "Unknown")
ARCH      := $(shell uname -m)
PLATFORM  := $(if $(findstring arm,$(ARCH)),linux/arm64,linux/amd64)
HAS_GPU   := $(shell command -v nvidia-smi >/dev/null 2>&1 && echo true || echo false)

PROFILE     := $(if $(filter true,$(HAS_GPU)),gpu,cpu)
DOCKERFILE  := $(if $(filter gpu,$(PROFILE)),Dockerfile.gpu,Dockerfile)

# -------- Files --------------------------------------------------------------
ENV_FILE      := .env.autogen
COMPOSE_FILE  := docker-compose.makefile.yaml
BAKE_FILE     := docker-bake.hcl      # (your refactored HCL from before)

# -------- Helper: write .env.autogen so Compose can read the same facts ------
$(ENV_FILE):
	@echo "# Auto-generated – do not commit"                >  $(ENV_FILE)
	@echo "TAG=$(TAG)"                                     >> $(ENV_FILE)
	@echo "VERSION=$(VERSION)"                             >> $(ENV_FILE)
	@echo "IMAGE_BASE=$(IMAGE_BASE)"                       >> $(ENV_FILE)
	@echo "PROFILE=$(PROFILE)"                             >> $(ENV_FILE)
	@echo "DOCKERFILE=$(DOCKERFILE)"                       >> $(ENV_FILE)
	@echo "PLATFORM=$(PLATFORM)"                           >> $(ENV_FILE)
	@echo "HAS_GPU=$(HAS_GPU)"                             >> $(ENV_FILE)

# -------- Commands -----------------------------------------------------------
.PHONY: help build up down logs sbom push clean

help:
	@echo "Targets: build | up | down | logs | sbom | push | clean"

# Build image(s) with Bake (cache + SBOM ready)
build: $(ENV_FILE)
	docker buildx bake auto                       \
		--file $(BAKE_FILE)                       \
		--set auto.platform=$(PLATFORM)           \
		--set auto.args.PROFILE=$(PROFILE)        \
		--set auto.args.VERSION=$(VERSION)        \
		--set auto.tags=$(IMAGE_BASE):$(PROFILE)-$(TAG)

# Bring up stack (Compose v2 plugin required)
up: $(ENV_FILE)
	@echo ">> launching stack with PROFILE=$(PROFILE), PLATFORM=$(PLATFORM)"
	docker compose --env-file $(ENV_FILE) -f $(COMPOSE_FILE) up --build -d

# Convenience wrappers --------------------------------------------------------
down:
	docker compose --env-file $(ENV_FILE) -f $(COMPOSE_FILE) down

logs:
	docker compose --env-file $(ENV_FILE) -f $(COMPOSE_FILE) logs -f

sbom: $(ENV_FILE)
	docker buildx bake sbom --file $(BAKE_FILE) \
		--set sbom.args.PROFILE=$(PROFILE) --set sbom.platform=$(PLATFORM)

push: build
	docker buildx bake auto --file $(BAKE_FILE) \
		--set auto.output=type=registry

clean:
	@rm -f $(ENV_FILE)
	docker image prune -f