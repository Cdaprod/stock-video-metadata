# Stage 1: Build base with CUDA and PyTorch
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Metadata
LABEL maintainer="cdaprod.dev"
ENV DEBIAN_FRONTEND=noninteractive

# Install core dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 python3.9-venv python3.9-dev python3-pip \
    git curl wget unzip ffmpeg libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up Python aliases
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip and wheel
RUN pip install --upgrade pip wheel setuptools

# Stage 2: Install Python dependencies
FROM base as builder

WORKDIR /app

# Download torch + torchvision with CUDA 11.8
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.2.2+cu118 torchvision==0.17.2+cu118

# Install notebook dependencies
COPY notebook_requirements.txt .
RUN pip install --upgrade --force-reinstall -r notebook_requirements.txt

# Clone and install OpenAI CLIP (from Git)
RUN git clone https://github.com/openai/CLIP.git && \
    pip install ./CLIP && rm -rf CLIP

# Stage 3: Runtime container with minimal layers
FROM base as runtime

WORKDIR /workspace

# Copy environment from builder
COPY --from=builder /usr/local/lib/python3.9 /usr/local/lib/python3.9
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/include /usr/local/include
COPY --from=builder /usr/local/lib/python3.9/dist-packages /usr/local/lib/python3.9/dist-packages

# Mount volumes for notebooks and data
VOLUME ["/workspace/notebooks", "/workspace/videos", "/workspace/outputs"]

# Optional: expose for Jupyter
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]