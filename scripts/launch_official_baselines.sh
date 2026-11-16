#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/magfilo}"
PYTHON="${PYTHON:-python}"; CONDA_SETUP="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"
CONDA_ENV="${BASELINE_CONDA_ENV:-finetune-sd}"; mkdir -p "${ARTIFACT_ROOT}/logs"
GROUP="${1:-all}"
case "${GROUP}" in
  group1|10.1)
    MODELS=(mask_rcnn cascade_mask_rcnn htc solov2)
    SESSION_PREFIX="mmdet_group1"
    ;;
  group2|10.2)
    MODELS=(condinst queryinst mask2former rtmdet_ins)
    SESSION_PREFIX="mmdet_group2"
    ;;
  all)
    MODELS=(mask_rcnn cascade_mask_rcnn htc solov2 condinst queryinst mask2former rtmdet_ins)
    SESSION_PREFIX="baseline"
    ;;
  *)
    echo "Usage: bash scripts/launch_official_baselines.sh [group1|group2|all]" >&2
    exit 2
    ;;
esac
SEEDS=(3020 3021 3022 3023 3024)
for GPU in 0 1 2 3; do
  SESSION="${SESSION_PREFIX}_gpu${GPU}"; QUEUE=""; PENDING=0
  INDEX=0
  for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      if (( INDEX % 4 == GPU )); then
        TEST_RESULT="${ARTIFACT_ROOT}/results/baseline_magfilo_v1_${MODEL}_s${SEED}/test_best.json"
        if [[ -s "${TEST_RESULT}" ]]; then
          echo "Skip complete: ${MODEL} seed ${SEED} (${TEST_RESULT})"
          INDEX=$((INDEX + 1))
          continue
        fi
        QUEUE+=" bash scripts/run_with_gpu_limit.sh ${GPU} ${ARTIFACT_ROOT}/logs/${MODEL}_s${SEED}.log bash scripts/run_mmdet_seed.sh ${MODEL} ${SEED} ${GPU};"
        PENDING=$((PENDING + 1))
      fi
      INDEX=$((INDEX + 1))
    done
  done
  tmux has-session -t "${SESSION}" 2>/dev/null && { echo "Exists: ${SESSION}"; continue; }
  if (( PENDING == 0 )); then
    echo "No pending MMDetection baselines on GPU ${GPU}"
    continue
  fi
  COMMAND="source ${CONDA_SETUP}; conda activate ${CONDA_ENV}; cd ${PROJECT_ROOT}; export ARTIFACT_ROOT=${ARTIFACT_ROOT}; export DATA_ROOT=${DATA_ROOT}; export PYTHON=${PYTHON}; export PYTHONPATH=${PROJECT_ROOT}/src:${PROJECT_ROOT}; set -euo pipefail;${QUEUE}"
  tmux new-session -d -s "${SESSION}" "bash -lc '${COMMAND}'"
  echo "Started ${SESSION}: ${PENDING} pending MMDetection ${GROUP} runs on GPU ${GPU}"
done
