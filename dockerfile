# Use NVIDIA CUDA base image with Python for GPU support
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Chicago
# Disable Flash Attention to avoid Triton compilation issues
ENV USE_FLASH_ATTENTION=0
ENV DISABLE_FLASH_ATTENTION=1
ENV USE_FLASH_ATTENTION_2=0

# Install Python 3.9 and pip from deadsnakes PPA
RUN apt-get update && apt-get install -y \
    tzdata \
    software-properties-common \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    build-essential \
    libffi-dev \
    libssl-dev \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.9
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.9 get-pip.py && \
    rm get-pip.py

# Create symlinks for python and pip
RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3.9 /usr/bin/pip && \
    ln -sf /usr/local/bin/pip3.9 /usr/bin/pip3

# Set the working directory inside the container
WORKDIR /app

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA 12.1 support (compatible with CUDA 13.0 drivers)
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Upgrade Triton (even though Flash Attention is disabled, some model code may still reference it)
# This ensures compatibility if any Triton code is executed
RUN pip install --no-cache-dir --upgrade "triton>=2.2.0" || echo "Triton upgrade failed, continuing..."

# Uninstall flash-attn if it exists to force models to use standard attention
RUN pip uninstall -y flash-attn flash-attention 2>/dev/null || echo "flash-attn not installed, skipping uninstall"

# Copy requirements file into the container
COPY requirements.txt requirements.txt

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container (optional)
# COPY . .

# Specify the default command
# For interactive mode: docker run ... aspect-gpu bash
# For detached mode: docker run -d ... aspect-gpu (will keep running)
CMD ["bash", "-c", "tail -f /dev/null"]
