# ASPECT: Alternative Splicing Event Classification with Transformers for Cancer Transcriptomics
ASPECT is a sequence-based framework for alternative splicing event classification built on DNABERT-2 with Byte Pair Encoding (BPE) tokenization. The model is designed to learn discriminative splicing signals from fixed-length genomic sequences(1,024 bp) and supports both binary event-pair classification and hierarchical multi-class inference.

**Authors:**
* Sahil Thapa
* Miguelangel Tamargo
*  Prof. Oluwatosin Oluwadare

[See ASPECT Wiki for Full Documentation on Installation and Usage](https://github.com/OluwadareLab/ASPECT/wiki)

___________________
#### OluwadareLab, University of North Texas, Denton
___________________
## Data and Model Availability
All datasets generated and analyzed during this study, as well as the trained models, are publicly available on Zenodo at:
[https://doi.org/10.5281/zenodo.18283327](https://doi.org/10.5281/zenodo.18283327)
## Folder Structure
```
SpliceRead/
+-- data/                 # Placeholder folder to be replaced with the downloaded dataset
├── cancer_derived_data/          
├── cancer_models/  
+-- models/               # Placeholder folder to be replaced with pre-trained models
+-- output/               # Stores generated synthetic sequences and visualization outputs
+-- code/                 # All training, generation, evaluation
¦   +-- AA_final_two_class_model/         # binary-class classification code
¦   +-- Hierarchical ASPECT Pipeline/
    ¦   ├── data_preparation/
    │   └── build_cascade_datasets.py
        ├── training/
        │   ├── train_classifier.py
        │   └── train_all_cascade_models.sh
        ├── inference/
        │   ├── hierarchical_cascade.py
        │   └── evaluate_predictions.py
        ├── examples/
        │   └── example_input.csv

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

[https://doi.org/10.5281/zenodo.18283327](https://doi.org/10.5281/zenodo.18283327)

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



### Step 5: Hierarchical ASPECT Pipeline
A cascaded classification pipeline for alternative splicing event prediction (cassette, alt_three, alt_five, Mutually Exclusive exons and retained introns).

## Folder Structure

```text
Hierarchical_ASPECT_Pipeline/
├── cancer_derived_data/          
├── cancer_models/               
├── data_preparation/
│   └── build_cascade_datasets.py
├── training/
│   ├── train_classifier.py
│   └── train_all_cascade_models.sh
├── inference/
│   ├── hierarchical_cascade.py
│   └── evaluate_predictions.py
├── examples/
│   └── example_input.csv
├── Dockerfile
├── requirements.txt
├── LICENSE
└── README.md
```

After placing the Zenodo folders:

```text
cancer_derived_data/
├── multi_class_datasets/{train,val,test}.csv
├── four_class_datasets/<A_vs_B_vs_C_vs_D>/{train,val,test}.csv
├── three_class_datasets/<A_vs_B_vs_C>/{train,val,test}.csv
└── binary_datsets/<A_vs_B>/{train,val,test}.csv

cancer_models/
├── multi_class_datasets/DB2_multi_class_datasets/best_model/
├── four_class_datasets/DB2_<labels>/best_model/
├── three_class_datasets/DB2_<labels>/best_model/
└── binary_datsets/DB2_<labels>/best_model/
```

### Step 6: Build Cascade Datasets (optional)

Required only if you are rebuilding splits from processed event CSVs.
Zenodo users can skip this step.

```bash
docker run --rm \
  -v $(pwd):/app \
  -v /path/to/processed_event_csvs:/raw:ro \
  -w /app \
  aspect-gpu \
  python data_preparation/build_cascade_datasets.py \
    --processed-dir /raw \
    --output-dir cancer_derived_data \
    --seed 42
```

---

### Step 7: Train Hierarchical Cascade Models

Trains the full model set for all five events:

1. root **5-class** model
2. all **4-class** subset models
3. all **3-class** subset models
4. all **2-class (binary)** subset models

```bash
docker run --rm --gpus all \
  -v $(pwd):/app \
  -w /app \
  -e DATA_DIR=/app/cancer_derived_data \
  -e MODELS_DIR=/app/cancer_models \
  -e USE_OPTUNA=True \
  -e OPTUNA_TRIALS=15 \
  -e USE_CLASS_WEIGHTS=True \
  -e OPTUNA_TARGET_METRIC=auto \
  -e MODEL_MAX_LENGTH=256 \
  -e TRAIN_BS=32 \
  -e EVAL_BS=32 \
  -e USE_WANDB=False \
  aspect-gpu \
  bash training/train_all_cascade_models.sh
```

**Example (GPU 0, 20 Optuna trials):**

```bash
docker run --rm --gpus '"device=0"' \
  -v $(pwd):/app \
  -w /app \
  -e DATA_DIR=/app/cancer_derived_data \
  -e MODELS_DIR=/app/cancer_models \
  -e OPTUNA_TRIALS=20 \
  -e USE_CLASS_WEIGHTS=True \
  -e USE_WANDB=False \
  aspect-gpu \
  bash training/train_all_cascade_models.sh
```

**Train one model only:**

```bash
docker run --rm --gpus all \
  -v $(pwd):/app \
  -w /app \
  -e RESULTS_DIR=/app/cancer_models/multi_class_datasets \
  aspect-gpu \
  python training/train_classifier.py \
    --data_path /app/cancer_derived_data/multi_class_datasets \
    --use_optuna True \
    --optuna_trials 15 \
    --use_class_weights True \
    --model_max_length 256 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --use_wandb False
```

Checkpoints are written under `cancer_models/`. Existing `best_model/` directories are skipped.

---

### Step 7: Hierarchical Inference

Input CSV columns:

| Column | Required | Description |
|--------|----------|-------------|
| `sequence` | yes | DNA sequence |
| `splice_type` | no | Ground truth (`AA/AD/ES/ME/RI`) for metrics |

**Internal test split (using Zenodo data + models):**

```bash
docker run --rm --gpus all \
  -v $(pwd):/app \
  -w /app \
  aspect-gpu \
  python inference/hierarchical_cascade.py \
    --input-csv /app/cancer_derived_data/multi_class_datasets/test.csv \
    --models-root /app/cancer_models \
    --training-runs-retrain none \
    --out-dir /app/results/internal_test \
    --write-metrics \
    --device cuda
```

**Holdout / custom CSV:**

```bash
docker run --rm --gpus all \
  -v $(pwd):/app \
  -w /app \
  aspect-gpu \
  python inference/hierarchical_cascade.py \
    --input-csv /app/cancer_derived_data/holdout_BRCA.csv \
    --models-root /app/cancer_models \
    --training-runs-retrain none \
    --out-dir /app/results/holdout_BRCA \
    --write-metrics \
    --device cuda
```

**Default thresholds:**

| Stage | Probability (`τ`) | Normalized entropy (`η`) |
|-------|-------------------|--------------------------|
| 5-class | 0.90 | 0.10 |
| 4-class | 0.90 | 0.18 |
| 3-class | 0.86 | 0.16 |
| 2-class | 0.78 | 0.14 |

Shared margin (`δ`): **0.65**

**Output:**

```text
results/<run>/
├── hierarchical_predictions.csv
├── hierarchical_predictions.jsonl
├── summary.json
├── metrics_summary.json
├── classification_report_stage5.csv
├── classification_report_final.csv
├── confusion_matrix_stage5.png
└── confusion_matrix_final.png
```

---

### Step 8: Evaluate Predictions

```bash
docker run --rm \
  -v $(pwd):/app \
  -w /app \
  aspect-gpu \
  python inference/evaluate_predictions.py \
    --run-dir /app/results/internal_test
```

---



## Citation

If you use ASPECT in your research, please cite our repository:

**Zenodo DOI**: [https://doi.org/10.5281/zenodo.18283327](https://doi.org/10.5281/zenodo.18283327)

---

## Contact

For questions, please contact [Sahil Thapa](mailto:sahilthapa@my.unt.edu) or [Prof. Oluwatosin Oluwadare](mailto:Oluwatosin.Oluwadare@unt.edu)

---

## License

MIT License. See `LICENSE` file for details.


