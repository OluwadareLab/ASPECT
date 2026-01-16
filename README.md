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
```
SpliceRead/
+-- data/                 # Placeholder folder to be replaced with the downloaded dataset
+-- models/               # Placeholder folder to be replaced with pre-trained models
+-- output/               # Stores generated synthetic sequences and visualization outputs
+-- code/                 # All training, generation, evaluation
¦   +-- AA_final_two_class_model/         # binary-class classification code
¦   +-- three_class_model_training/  # multi-class classification code
+-- Pipeline            # contain 3class to 2class hierarchical pipeline
+--Dockerfile           # Containerized environment for reproducibility 
+-- README.md            # Project documentation
```

---


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

### Step 5: Binary-Class Model Training
```bash
./run_training_docker.sh start -d <dataset> -g <gpu_id> -t <num_trials>
```

**Parameters:**
- `-d`: Dataset name (`cassette_vs_alt_three`, `cassette_vs_alt_five`, `alt_three_vs_alt_five`, `constitutive_vs_cassette`, `constitutive_vs_alt_three`, `constitutive_vs_alt_five`)
- `-g`: GPU ID (e.g., `0` or `1`)
- `-t`: Number of Optuna trials (e.g., `20`)

**Configuration:**
- **Data Path**: Edit `DATASET_PATH` on line 187 in `run_training_docker.sh` (default: `/app/data_preprocessing/balanced_binary_datasets/${DATASET}`)
- **Output Directory**: Edit `RESULTS_DIR` on line 26 in `run_training_docker.sh` (default: `/app/binary_model_training/result_mn`)

**Example:**
```bash
./run_training_docker.sh start -d cassette_vs_alt_three -g 0 -t 20
```

**View logs:** `./run_training_docker.sh logs`  
**Check status:** `./run_training_docker.sh status`  
**Stop training:** `./run_training_docker.sh stop`



### Step 6: Three-Class Model Training

  **Start training:** `./run_training_docker.sh start`  
  **Configure data path:** Edit line **163** in `run_training_docker.sh` → set `DATASET_PATH`  
  **Configure output directory:** Edit line **27** in `run_training_docker.sh` → set `RESULTS_DIR`  
  **GPU selection:** `./run_training_docker.sh start -g 1` (default: GPU 0)  
  **Number of trials:** `./run_training_docker.sh start -t 30` (default: 20)  
  **Combine options:** `./run_training_docker.sh start -g 1 -t 30`  
  **Monitor:** `./run_training_docker.sh logs` | **Status:** `./run_training_docker.sh status` | **Stop:** `./run_training_docker.sh stop`  
  **Results location:** `RESULTS_DIR/DB2_{dataset_name}/` (contains `best_model/`, `model_output/`, `logs/`, and evaluation files)

  #### Testing the Three-Class Model

  **From project root** (where `three_class_model_training/` exists):
```bash
docker run --gpus device=0 --rm \
    -v $(pwd):/app \
    -w /app/three_class_model_training \
    aspect-gpu \
    python3 test_model.py \
        --model_path <path_to_model>/best_model \
        --test_data_path <path_to_test.csv> \
        --output_dir ./test_results
```
**Example** (from project root):
```bash
docker run --gpus device=0 --rm \
    -v $(pwd):/app \
    -w /app/three_class_model_training \
    aspect-gpu \
    python3 test_model.py \
        --model_path result_sample_2/DB2_balanced_three_class_from_multiclass/best_model \
        --test_data_path ../data_preprocessing/balanced_three_class_datasets/cassette_alt_three_alt_five/test.csv \
        --output_dir ./test_results
```


