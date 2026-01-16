# ASPECT
ASPECT: Alternative SPlicing Events Classification with Transformer


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



