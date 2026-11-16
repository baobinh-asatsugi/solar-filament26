#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/magfilo}"
PYTHON="${PYTHON:-python}"
CONDA_SETUP="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-finetune-sd}"
mkdir -p "${ARTIFACT_ROOT}/logs"
SEEDS=(3020 3021 3022 3023)
for INDEX in "${!SEEDS[@]}"; do
  SEED="${SEEDS[$INDEX]}"; GPU="${INDEX}"; SESSION="m2lite_gpu${GPU}"
  QUEUE=""; PENDING=0
  RESULT="${ARTIFACT_ROOT}/results/magfilo_v1_m2lite_s${SEED}_val.json"
  if [[ -s "${RESULT}" ]]; then
    echo "Skip complete: M2-Lite seed ${SEED} (${RESULT})"
  else
    QUEUE+=" bash scripts/run_with_gpu_limit.sh ${GPU} ${ARTIFACT_ROOT}/logs/m2lite_s${SEED}.log bash scripts/train_one_seed.sh ${SEED} ${GPU};"
    PENDING=$((PENDING + 1))
  fi
  if [[ "${GPU}" == "0" ]]; then
    RESULT_3024="${ARTIFACT_ROOT}/results/magfilo_v1_m2lite_s3024_val.json"
    if [[ -s "${RESULT_3024}" ]]; then
      echo "Skip complete: M2-Lite seed 3024 (${RESULT_3024})"
    else
      QUEUE+=" bash scripts/run_with_gpu_limit.sh 0 ${ARTIFACT_ROOT}/logs/m2lite_s3024.log bash scripts/train_one_seed.sh 3024 0;"
      PENDING=$((PENDING + 1))
    fi
  fi
  tmux has-session -t "${SESSION}" 2>/dev/null && { echo "Exists: ${SESSION}"; continue; }
  (( PENDING > 0 )) || { echo "No pending M2-Lite runs on GPU ${GPU}"; continue; }
  COMMAND="source ${CONDA_SETUP}; conda activate ${CONDA_ENV}; cd ${PROJECT_ROOT}; export ARTIFACT_ROOT=${ARTIFACT_ROOT}; export DATA_ROOT=${DATA_ROOT}; export PYTHONPATH=${PROJECT_ROOT}/src:${PROJECT_ROOT}; export PYTHON=${PYTHON}; set -euo pipefail;${QUEUE}"
  tmux new-session -d -s "${SESSION}" "bash -lc '${COMMAND}'"
  echo "Started ${SESSION}: ${PENDING} pending M2-Lite runs on GPU ${GPU}"
done
tmux ls
