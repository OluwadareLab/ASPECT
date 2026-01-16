FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Chicago
ENV USE_FLASH_ATTENTION=0
ENV DISABLE_FLASH_ATTENTION=1
ENV USE_FLASH_ATTENTION_2=0

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

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.9 get-pip.py && \
    rm get-pip.py

RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3.9 /usr/bin/pip && \
    ln -sf /usr/local/bin/pip3.9 /usr/bin/pip3

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121


RUN pip install --no-cache-dir --upgrade "triton>=2.2.0" || echo "Triton upgrade failed, continuing..."
RUN pip uninstall -y flash-attn flash-attention 2>/dev/null || echo "flash-attn not installed, skipping uninstall"

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash", "-c", "tail -f /dev/null"]
