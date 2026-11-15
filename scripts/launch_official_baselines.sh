#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/magfilo}"
PYTHON="${PYTHON:-python}"; CONDA_SETUP="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"
CONDA_ENV="${BASELINE_CONDA_ENV:-mmdet-baselines}"; mkdir -p "${ARTIFACT_ROOT}/logs"
MODELS=(mask_rcnn cascade_mask_rcnn htc solov2 condinst queryinst mask2former rtmdet_ins)
SEEDS=(3020 3021 3022 3023 3024)
for GPU in 0 1 2 3; do
  SESSION="baseline_gpu${GPU}"; QUEUE=""
  INDEX=0
  for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      if (( INDEX % 4 == GPU )); then
        QUEUE+=" bash scripts/run_with_gpu_limit.sh ${GPU} ${ARTIFACT_ROOT}/logs/${MODEL}_s${SEED}.log bash scripts/run_mmdet_seed.sh ${MODEL} ${SEED} ${GPU};"
      fi
      INDEX=$((INDEX + 1))
    done
  done
  tmux has-session -t "${SESSION}" 2>/dev/null && { echo "Exists: ${SESSION}"; continue; }
  COMMAND="source ${CONDA_SETUP}; conda activate ${CONDA_ENV}; cd ${PROJECT_ROOT}; export ARTIFACT_ROOT=${ARTIFACT_ROOT}; export DATA_ROOT=${DATA_ROOT}; export PYTHON=${PYTHON}; export PYTHONPATH=${PROJECT_ROOT}/src:${PROJECT_ROOT}; set -euo pipefail;${QUEUE}"
  tmux new-session -d -s "${SESSION}" "bash -lc '${COMMAND}'"
  echo "Started ${SESSION}: 10 sequential runs on GPU ${GPU}"
done
