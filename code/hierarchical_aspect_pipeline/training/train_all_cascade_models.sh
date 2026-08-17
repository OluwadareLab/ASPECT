#!/usr/bin/env bash
set -euo pipefail

# Trains every model required by the 5→4→3→2 cascade, sequentially.
# DATA_FINAL_DIR must contain the dataset tree created by
# data_preparation/build_cascade_datasets.py.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_FINAL_DIR="${DATA_FINAL_DIR:-${PROJECT_ROOT}/datasets}"
CODE_FILE="${CODE_FILE:-${SCRIPT_DIR}/train_classifier.py}"

RESULTS_BASE="${DATA_FINAL_DIR}/training_runs"
LOGS_BASE="${DATA_FINAL_DIR}/training_logs"
mkdir -p "${RESULTS_BASE}" "${LOGS_BASE}"

# Tunable defaults (override via env vars when needed)
USE_OPTUNA="${USE_OPTUNA:-True}"
OPTUNA_TRIALS="${OPTUNA_TRIALS:-15}"
USE_CLASS_WEIGHTS="${USE_CLASS_WEIGHTS:-True}"
OPTUNA_TARGET_METRIC="${OPTUNA_TARGET_METRIC:-auto}"   # auto|weighted|macro|blend
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-256}"
TRAIN_BS="${TRAIN_BS:-32}"
EVAL_BS="${EVAL_BS:-32}"
USE_WANDB="${USE_WANDB:-False}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${CODE_FILE}" ]]; then
  echo "ERROR: Training script not found: ${CODE_FILE}" >&2
  exit 1
fi
if [[ ! -d "${DATA_FINAL_DIR}/multi_class_datasets" ]]; then
  echo "ERROR: Dataset tree not found: ${DATA_FINAL_DIR}" >&2
  echo "Build it first with data_preparation/build_cascade_datasets.py." >&2
  exit 1
fi

run_dataset() {
  local group="$1"
  local dataset_dir="$2"
  local dataset_name
  dataset_name="$(basename "${dataset_dir}")"

  local result_root="${RESULTS_BASE}/${group}"
  local run_dir="${result_root}/DB2_${dataset_name}"
  local best_model_dir="${run_dir}/best_model"
  local group_log_dir="${LOGS_BASE}/${group}"
  local log_file="${group_log_dir}/${dataset_name}.log"

  mkdir -p "${result_root}" "${group_log_dir}"

  if [[ -d "${best_model_dir}" ]]; then
    echo "[SKIP] ${group}/${dataset_name} (best_model already exists)"
    return 0
  fi

  echo "[RUN] ${group}/${dataset_name}"
  echo "      dataset=${dataset_dir}"
  echo "      results=${run_dir}"
  echo "      log=${log_file}"

  RESULTS_DIR="${result_root}" \
  "${PYTHON_BIN}" "${CODE_FILE}" \
    --data_path "${dataset_dir}" \
    --use_wandb "${USE_WANDB}" \
    --use_optuna "${USE_OPTUNA}" \
    --optuna_trials "${OPTUNA_TRIALS}" \
    --use_class_weights "${USE_CLASS_WEIGHTS}" \
    --optuna_target_metric "${OPTUNA_TARGET_METRIC}" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --per_device_train_batch_size "${TRAIN_BS}" \
    --per_device_eval_batch_size "${EVAL_BS}" \
    2>&1 | tee "${log_file}"
}

# 1) Root 5-class model
if [[ -d "${DATA_FINAL_DIR}/multi_class_datasets" ]]; then
  run_dataset "multi_class_datasets" "${DATA_FINAL_DIR}/multi_class_datasets"
fi

# 2) All 4-class, 3-class, and binary subset models
for group in four_class_datasets three_class_datasets binary_datsets; do
  group_dir="${DATA_FINAL_DIR}/${group}"
  if [[ ! -d "${group_dir}" ]]; then
    echo "[WARN] Missing group folder: ${group_dir}"
    continue
  fi
  while IFS= read -r -d '' ds; do
    run_dataset "${group}" "${ds}"
  done < <(find "${group_dir}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
done

echo "[DONE] All dataset training jobs finished."
