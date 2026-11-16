#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/magfilo}"
PYTHON="${PYTHON:-python}"; CONDA_SETUP="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"; mkdir -p "${ARTIFACT_ROOT}/logs"
CONDA_ENV="${CONDA_ENV:-finetune-sd}"
for GPU in 0 1 2 3; do
  SEED=$((3020 + GPU)); SESSION="unetpp_ws_gpu${GPU}"; QUEUE=""; PENDING=0
  RESULT="${ARTIFACT_ROOT}/results/baseline_magfilo_v1_unetpp_watershed_s${SEED}/test_best.json"
  if [[ -s "${RESULT}" ]]; then
    echo "Skip complete: unetpp_watershed seed ${SEED} (${RESULT})"
  else
    QUEUE+=" bash scripts/run_with_gpu_limit.sh ${GPU} ${ARTIFACT_ROOT}/logs/unetpp_watershed_s${SEED}.log bash scripts/run_unetpp_watershed_seed.sh ${SEED} ${GPU};"
    PENDING=$((PENDING + 1))
  fi
  if [[ "${GPU}" == 0 ]]; then
    RESULT_3024="${ARTIFACT_ROOT}/results/baseline_magfilo_v1_unetpp_watershed_s3024/test_best.json"
    if [[ -s "${RESULT_3024}" ]]; then
      echo "Skip complete: unetpp_watershed seed 3024 (${RESULT_3024})"
    else
      QUEUE+=" bash scripts/run_with_gpu_limit.sh 0 ${ARTIFACT_ROOT}/logs/unetpp_watershed_s3024.log bash scripts/run_unetpp_watershed_seed.sh 3024 0;"
      PENDING=$((PENDING + 1))
    fi
  fi
  tmux has-session -t "${SESSION}" 2>/dev/null && { echo "Exists: ${SESSION}"; continue; }
  (( PENDING > 0 )) || { echo "No pending U-Net++ watershed runs on GPU ${GPU}"; continue; }
  COMMAND="source ${CONDA_SETUP}; conda activate ${CONDA_ENV}; cd ${PROJECT_ROOT}; export ARTIFACT_ROOT=${ARTIFACT_ROOT}; export DATA_ROOT=${DATA_ROOT}; export PYTHON=${PYTHON}; set -euo pipefail;${QUEUE}"
  tmux new-session -d -s "${SESSION}" "bash -lc '${COMMAND}'"; echo "Started ${SESSION}"
done
