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
+-- models/               # Placeholder folder to be replaced with pre-trained models
+-- output/               # Stores generated synthetic sequences and visualization outputs
+-- code/                 # All training, generation, evaluation
¦   +-- AA_final_two_class_model/         # binary-class classification code
¦   +-- Hierarchical ASPECT Pipeline/ 
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
A cascaded classification pipeline for alternative splicing event prediction (cassette, alt_three, alt_five).

#### Docker
```bash
docker run --rm --gpus all -v $(pwd):/app -w /app/three_class_pipeline aspect-gpu python run_all_tests.py /path/to/your/data.csv
```
#### Local
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

## Citation

If you use ASPECT in your research, please cite our repository:

**Zenodo DOI**: [https://doi.org/10.5281/zenodo.18283327](https://doi.org/10.5281/zenodo.18283327)

---

## Contact

For questions, please contact [Sahil Thapa](mailto:sahilthapa@my.unt.edu) or [Prof. Oluwatosin Oluwadare](mailto:Oluwatosin.Oluwadare@unt.edu)

---

## License

MIT License. See `LICENSE` file for details.


