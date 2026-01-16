# ASPECT
ASPECT: Alternative SPlicing Events Classification with Transformer
ASPECT is a sequence-based framework for alternative splicing event classification built on DNABERT-2 with Byte Pair Encoding (BPE) tokenization. The model is designed to learn discriminative splicing signals from fixed-length genomic sequences and supports both binary event-pair classification and hierarchical multi-class inference.


## Docker Setup

### 1. Build the Docker Image


```bash
# Navigate to the project root directory
cd /path/to/aspect

# Build the Docker image
docker build -t aspect-gpu -f dockerfile .
```

This will:
- Use NVIDIA CUDA 12.4.0 base image
- Install Python 3.9 and required system packages
- Install PyTorch 2.1.0 with CUDA 12.1 support
- Install all Python dependencies from `requirements.txt.`
- Disable Flash Attention to avoid Triton compilation issues



