FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir torbi

COPY train_requirements.txt .
RUN pip install --no-cache-dir -r train_requirements.txt

COPY . .

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

