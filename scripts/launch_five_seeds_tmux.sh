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
  tmux has-session -t "${SESSION}" 2>/dev/null && { echo "Exists: ${SESSION}"; continue; }
  EXTRA=""
  if [[ "${GPU}" == "0" ]]; then
    EXTRA="; bash scripts/run_with_gpu_limit.sh 0 ${ARTIFACT_ROOT}/logs/m2lite_s3024.log bash scripts/train_one_seed.sh 3024 0"
  fi
  COMMAND="source ${CONDA_SETUP}; conda activate ${CONDA_ENV}; cd ${PROJECT_ROOT}; export ARTIFACT_ROOT=${ARTIFACT_ROOT}; export DATA_ROOT=${DATA_ROOT}; export PYTHONPATH=${PROJECT_ROOT}/src:${PROJECT_ROOT}; export PYTHON=${PYTHON}; set -euo pipefail; bash scripts/run_with_gpu_limit.sh ${GPU} ${ARTIFACT_ROOT}/logs/m2lite_s${SEED}.log bash scripts/train_one_seed.sh ${SEED} ${GPU}${EXTRA}"
  tmux new-session -d -s "${SESSION}" "bash -lc '${COMMAND}'"
  echo "Started ${SESSION} on GPU ${GPU}"
done
tmux ls
