# --- Stage 1: Python Base (CPU, ARM) ---
FROM python:3.11-slim as base

LABEL maintainer="cdaprod.dev" \
      org.opencontainers.image.source="https://github.com/Cdaprod/stock-video-metadata" \
      org.opencontainers.image.description="FastAPI app for video enrichment API (CPU, ARM)" \
      org.opencontainers.image.version="0.1.0"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget unzip ffmpeg libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY ./scripts /workspace/scripts
COPY requirements.txt /workspace/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]