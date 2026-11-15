#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-python}"; CONDA_SETUP="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"; mkdir -p "${ARTIFACT_ROOT}/logs"
for GPU in 0 1 2 3; do
  SEED=$((3020 + GPU)); SESSION="unetpp_ws_gpu${GPU}"; EXTRA=""
  [[ "${GPU}" == 0 ]] && EXTRA="; bash scripts/run_unetpp_watershed_seed.sh 3024 0 2>&1 | tee ${ARTIFACT_ROOT}/logs/unetpp_watershed_s3024.log"
  tmux has-session -t "${SESSION}" 2>/dev/null && { echo "Exists: ${SESSION}"; continue; }
  COMMAND="source ${CONDA_SETUP}; conda activate finetune-sd; cd ${PROJECT_ROOT}; export ARTIFACT_ROOT=${ARTIFACT_ROOT}; export PYTHON=${PYTHON}; set -euo pipefail; bash scripts/run_unetpp_watershed_seed.sh ${SEED} ${GPU} 2>&1 | tee ${ARTIFACT_ROOT}/logs/unetpp_watershed_s${SEED}.log${EXTRA}"
  tmux new-session -d -s "${SESSION}" "bash -lc '${COMMAND}'"; echo "Started ${SESSION}"
done
