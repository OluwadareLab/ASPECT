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

[https://doi.org/10.5281/zenodo.15538290](https://doi.org/10.5281/zenodo.1553tyty82)

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


#### Testing Three-Class Model
**From project root** (where `binary_model_training/` exists):
```bash
docker run --rm --gpus device=<gpu_id> \
  -v $(pwd):/app \
  -w /app/binary_model_training \
  aspect-gpu \
  python3 test_model.py \
    --model_path <model_path> \
    --test_data_path <test_data_path> \
    --output_dir <output_dir> \
    --batch_size <batch_size>
```
**Required Parameters:**
- `--model_path`: Path to trained model (e.g., `result_13/DB2_alt_three_vs_alt_five/best_model`)
- `--test_data_path`: Path to test CSV (e.g., `../data_preprocessing/balanced_binary_datasets/alt_three_vs_alt_five/test.csv`)

**Optional Parameters:**
- `--output_dir`: Output directory (default: `./test_results`)
- `--batch_size`: Batch size (default: `32`)

**Example:**
```bash
docker run --rm --gpus device=0 \
  -v $(pwd):/app \
  -w /app/binary_model_training \
  aspect-gpu \
  python3 test_model.py \
    --model_path result_13/DB2_alt_three_vs_alt_five/best_model \
    --test_data_path ../data_preprocessing/balanced_binary_datasets/alt_three_vs_alt_five/test.csv \
    --output_dir ./test_results
```

**Output:** Generates confusion matrix, ROC curve, classification report, and metrics JSON.



### Step 6: Three-Class Model Training

  **Start training:** `./run_training_docker.sh start`  
  **Configure data path:** Edit line **163** in `run_training_docker.sh` → set `DATASET_PATH`  
  **Configure output directory:** Edit line **27** in `run_training_docker.sh` → set `RESULTS_DIR`  
  **GPU selection:** `./run_training_docker.sh start -g 1` (default: GPU 0)  
  **Number of trials:** `./run_training_docker.sh start -t 30` (default: 20)  
  **Combine options:** `./run_training_docker.sh start -g 1 -t 30`  
  **Monitor:** `./run_training_docker.sh logs` | **Status:** `./run_training_docker.sh status` | **Stop:** `./run_training_docker.sh stop`  
  **Results location:** `RESULTS_DIR/DB2_{dataset_name}/` (contains `best_model/`, `model_output/`, `logs/`, and evaluation files)

  #### Testing Three-Class Model

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

### Step 7: Hierarchical ASPECT Pipeline
A cascaded classification pipeline for alternative splicing event prediction (cassette, alt_three, alt_five).

```bash
# Run full pipeline with custom data
python run_all_tests.py /path/to/your/data.csv
```

#### Configuration

##### A. Three-Class Model Path
**File**: `three_class_test.py` (line ~105)
```python
model_path = "../three_class_model_training/result_11/DB2_balanced_three_class_from_multiclass/best_model"
```

##### B. Binary Model Paths
**File**: `binary_class_test.py` (lines ~100-104)
```python
binary_model_overrides = {
    tuple(sorted(["cassette", "alt_three"])): "../binary_model_training/result_8/DB2_cassette_vs_alt_three/best_model",
    tuple(sorted(["cassette", "alt_five"])): "../binary_model_training/result_8/DB2_cassette_vs_alt_five/best_model",
    tuple(sorted(["alt_three", "alt_five"])): "../binary_model_training/result_13/DB2_alt_three_vs_alt_five/best_model",
}
```
##### Output

Results saved in: `./test_result_{dataset_name}_{timestamp}/`
- `result_three_class/predictions_with_probabilities.csv` - Three-class predictions
- `result_binary_class/predictions_with_probabilities.csv` - Final hierarchical predictions

##### Visualization

```bash
python plot_cascaded_results.py --input-dir ./test_result_{dataset_name}_{timestamp}
```

Generates `event_counts_side_by_side.png` comparing three-class vs hierarchical pipeline performance.


