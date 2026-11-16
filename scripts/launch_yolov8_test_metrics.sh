#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/magfilo}"
PYTHON="${PYTHON:-python}"
CONDA_SETUP="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-finetune-sd}"
mkdir -p "${ARTIFACT_ROOT}/logs"

for GPU in 0 1 2 3; do
  SEED=$((3020 + GPU))
  SESSION="yolo_metrics_gpu${GPU}"
  QUEUE=""; PENDING=0
  RESULT="${ARTIFACT_ROOT}/results/magfilo_v1_yolov8l_s${SEED}/test_best.json"
  if [[ -s "${RESULT}" ]] && grep -q '"semantic_dice"' "${RESULT}"; then
    echo "Skip complete: YOLO metrics seed ${SEED} (${RESULT})"
  else
    QUEUE+=" bash scripts/reevaluate_yolov8_test_seed.sh ${SEED} ${GPU};"
    PENDING=$((PENDING + 1))
  fi
  if [[ "${GPU}" == "0" ]]; then
    RESULT_3024="${ARTIFACT_ROOT}/results/magfilo_v1_yolov8l_s3024/test_best.json"
    if [[ -s "${RESULT_3024}" ]] && grep -q '"semantic_dice"' "${RESULT_3024}"; then
      echo "Skip complete: YOLO metrics seed 3024 (${RESULT_3024})"
    else
      QUEUE+=" bash scripts/reevaluate_yolov8_test_seed.sh 3024 0;"
      PENDING=$((PENDING + 1))
    fi
  fi
  tmux has-session -t "${SESSION}" 2>/dev/null && { echo "Exists: ${SESSION}"; continue; }
  (( PENDING > 0 )) || { echo "No pending YOLO metric runs on GPU ${GPU}"; continue; }
  COMMAND="source ${CONDA_SETUP}; conda activate ${CONDA_ENV}; cd ${PROJECT_ROOT}; export ARTIFACT_ROOT=${ARTIFACT_ROOT}; export DATA_ROOT=${DATA_ROOT}; export PYTHON=${PYTHON}; export PYTHONPATH=${PROJECT_ROOT}/src:${PROJECT_ROOT}; set -euo pipefail;${QUEUE}"
  tmux new-session -d -s "${SESSION}" "bash -lc '${COMMAND}'"
  echo "Started ${SESSION}: YOLO test metrics seed ${SEED} on GPU ${GPU}"
done
