# --- Stage 1: CUDA + Python Base ---
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

LABEL maintainer="cdaprod.dev"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev python3-pip \
    git curl wget unzip ffmpeg libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN pip install --upgrade pip wheel setuptools

WORKDIR /workspace

# Copy code and requirements
# COPY ./app /workspace/app
COPY ./scripts /workspace/scripts
COPY requirements-core.txt /workspace/

# Install production dependencies
RUN pip install --no-cache-dir -r requirements-core.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]