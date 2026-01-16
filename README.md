# ASPECT: Alternative SPlicing Events Classification with Transformer

ASPECT is a sequence-based framework for alternative splicing event classification built on DNABERT-2 with Byte Pair Encoding (BPE) tokenization. The model is designed to learn discriminative splicing signals from fixed-length genomic sequences(1,024 bp) and supports both binary event-pair classification and hierarchical multi-class inference.

**Authors:**
* Sahil Thapa
* Miguelangel Tamargo
*  Prof. Oluwatosin Oluwadare

___________________
#### OluwadareLab, University of North Texas, Denton
___________________

## Folder Structure


### Step 1: Clone Repository

```bash
git clone https://github.com/OluwadareLab/ASPECT.git
cd ASPECT
```

### Step 2: Download Data and Models

Download the Zenodo archive from the link below:

[https://doi.org/10.5281/zenodo.15538290](https://doi.org/10.5281/zenodo.15538290)

### Step 3: Place Files

* Extract the `ASPECT.zip` archive.
* Replace the `data/` folder in the repo with the extracted `data/` folder.
* Replace the `models/` folder in the repo with the extracted `models/` folder.

### Step 4: Build Docker Image

```bash
docker build -t aspect-gpu .
```
This will:
- Use NVIDIA CUDA 12.4.0 base image
- Install Python 3.9 and required system packages
- Install PyTorch 2.1.0 with CUDA 12.1 support
- Install all Python dependencies from `requirements.txt.`
- Disable Flash Attention to avoid Triton compilation issues



