# Hierarchical ASPECT

Reproducibility code for training and evaluating the cancer splice-event
classification cascade:

**5-class → 4-class → 3-class → 2-class**

The five event labels are `AA`, `AD`, `ES`, `ME`, and `RI`. Every uncertain
prediction is passed to a classifier over a smaller candidate-label subset.


## Repository layout

```text
.
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
└── .gitignore
```

Model weights and patient-derived sequence data are deliberately excluded.

## Installation

Python 3.9–3.10 and an NVIDIA GPU are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional GPU container:

```bash
docker build -t hierarchical-aspect:latest .
docker run --rm --gpus all -v "$PWD:/app" -w /app hierarchical-aspect:latest bash
```

The models use `zhihan1996/DNABERT-2-117M` from Hugging Face. The first run
therefore requires network access or an existing Hugging Face cache.

## 1. Prepare the cascade datasets

Place processed cancer event files in a directory. Every input filename must
match `*_all_events_1024.csv` and contain:

```text
splice_type,sequence
```

Build globally deduplicated, stratified train/validation/test splits:

```bash
python data_preparation/build_cascade_datasets.py \
  --processed-dir /path/to/processed_event_csvs \
  --output-dir datasets \
  --seed 42
```

This creates:

```text
datasets/
├── multi_class_datasets/       # one 5-class dataset
├── four_class_datasets/        # every 4-label subset
├── three_class_datasets/       # every 3-label subset
└── binary_datsets/             # every 2-label subset
```

`binary_datsets` retains the spelling used by the trained-checkpoint layout.
The same global split is used before deriving subsets, preventing exact
sequence overlap between train, validation, and test splits.

To exclude sequences found in a separate holdout cohort:

```bash
python data_preparation/build_cascade_datasets.py \
  --processed-dir /path/to/training_cohorts \
  --output-dir datasets \
  --exclude-sequences-file /path/to/holdout_sequences.csv
```

## 2. Train all cascade classifiers

The launcher trains the root 5-class model followed by all 4-class, 3-class,
and binary subset models:

```bash
DATA_FINAL_DIR="$PWD/datasets" \
OPTUNA_TRIALS=15 \
USE_CLASS_WEIGHTS=True \
bash training/train_all_cascade_models.sh
```

Useful environment variables:

- `USE_OPTUNA=True|False`
- `OPTUNA_TRIALS=15`
- `USE_CLASS_WEIGHTS=True|False`
- `OPTUNA_TARGET_METRIC=auto|weighted|macro|blend`
- `MODEL_MAX_LENGTH=256`
- `TRAIN_BS=32`
- `EVAL_BS=32`
- `USE_WANDB=False`
- `WANDB_PROJECT=hierarchical-aspect`
- `WANDB_ENTITY=<optional account/team>`

Checkpoints are written under:

```text
datasets/training_runs/
├── multi_class_datasets/DB2_multi_class_datasets/best_model/
├── four_class_datasets/DB2_<labels>/best_model/
├── three_class_datasets/DB2_<labels>/best_model/
└── binary_datsets/DB2_<labels>/best_model/
```

## 3. Run hierarchical inference

The input CSV requires `sequence`; `splice_type` is optional and enables
evaluation metrics.

```bash
python inference/hierarchical_cascade.py \
  --input-csv examples/example_input.csv \
  --final-root datasets \
  --training-runs-retrain none \
  --out-dir results/example \
  --write-metrics \
  --device cuda
```

Default routing thresholds:

- 5-class probability: `0.90`
- 4-class probability: `0.90`
- 3-class probability: `0.86`
- 2-class probability: `0.78`
- margin: `0.65`
- normalized entropy (5/4/3/2): `0.10 / 0.18 / 0.16 / 0.14`

All thresholds can be overridden through the command-line options shown by:

```bash
python inference/hierarchical_cascade.py --help
```

Primary outputs:

- `hierarchical_predictions.csv`
- `hierarchical_predictions.jsonl`
- `summary.json`
- `metrics_summary.json` when `--write-metrics` is enabled

## 4. Evaluate saved predictions

```bash
python inference/evaluate_predictions.py \
  --run-dir results/example
```

This writes stage-5 and final-cascade classification reports and confusion
matrices.

## Reproducibility and publication notes

- The data split seed is 42 by default.
- Training data and model checkpoints are intentionally not committed.
- Do not publish controlled-access or patient-identifiable data.
- Add a paper citation and an institution-approved `LICENSE` before making
  the repository public.

## What this package includes

| Component | Status |
|-----------|--------|
| Build 5/4/3/2 stratified datasets | Included |
| Train all cascade classifiers | Included |
| Hierarchical 5→4→3→2 inference | Included |
| Classification metrics / confusion matrices | Included |
| Dockerfile + requirements | Included |
| Model checkpoints (`best_model/`) | Excluded (large; host separately if needed) |
| Processed TCGA / SpliceSeq CSVs | Excluded (data-access / privacy) |
| Threshold-sweep ablation scripts | Excluded (optional paper tooling) |
| Holdout-cohort plotting / table scripts | Excluded (paper figures, not core cascade) |

## Before public GitHub release

1. Add `LICENSE` (ask coauthors / institution which license to use).
2. Add the paper citation / DOI once available.
3. Confirm that no controlled-access sequences or TCGA-derived private tables
   will be uploaded.
4. Optionally host trained `best_model/` directories on Zenodo / Hugging Face /
   institutional storage and link them from this README.
5. Initialize git in this folder only (not the full research workspace).

This package is **code-clean for GitHub** (no secrets, no host paths, no
checkpoints, ~135 KB). It is **not publication-complete** until a license and
citation are added, and until data/model availability is documented.

