# Use PyTorch official image with CUDA 12.8 and cuDNN 9
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY train_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r train_requirements.txt

# Copy project files
COPY . .

# Set Python environment variables for GPU memory management
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Default command to run training
#CMD ["python", "train.py"]
