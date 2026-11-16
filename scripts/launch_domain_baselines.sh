#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"; DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/magfilo}"
PYTHON="${PYTHON:-python}"; CONDA_SETUP="${CONDA_SETUP:-$(conda info --base)/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-finetune-sd}"; mkdir -p "${ARTIFACT_ROOT}/logs"
METHODS=(fullres_unet edgeattnet maskrcnn_consensus detr_crop_unet); SEEDS=(3020 3021 3022 3023 3024)
for GPU in 0 1 2 3; do
  SESSION="domain_baseline_gpu${GPU}"; QUEUE=""; INDEX=0; PENDING=0
  for METHOD in "${METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      if (( INDEX % 4 == GPU )); then
        TEST_RESULT="${ARTIFACT_ROOT}/results/baseline_magfilo_v1_${METHOD}_s${SEED}/test_best.json"
        if [[ -s "${TEST_RESULT}" ]]; then
          echo "Skip complete: ${METHOD} seed ${SEED} (${TEST_RESULT})"
          INDEX=$((INDEX + 1))
          continue
        fi
        case "${METHOD}" in
          fullres_unet|edgeattnet) SCRIPT="scripts/run_semantic_baseline_seed.sh ${METHOD}";;
          maskrcnn_consensus) SCRIPT="scripts/run_maskrcnn_consensus_seed.sh";;
          detr_crop_unet) SCRIPT="scripts/run_detr_crop_unet_seed.sh";;
        esac
        QUEUE+=" if ! bash scripts/run_with_gpu_limit.sh ${GPU} ${ARTIFACT_ROOT}/logs/${METHOD}_s${SEED}.log bash ${SCRIPT} ${SEED} ${GPU}; then echo '[domain queue] FAILED: ${METHOD} seed ${SEED} on GPU ${GPU}'; fi;"
        PENDING=$((PENDING + 1))
      fi
      INDEX=$((INDEX + 1))
    done
  done
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    if [[ "$(tmux display-message -p -t "${SESSION}" '#{pane_dead}')" == "1" ]]; then
      echo "Removing finished/dead session: ${SESSION}"
      tmux kill-session -t "${SESSION}"
    else
      echo "Exists and running: ${SESSION}"
      continue
    fi
  fi
  if (( PENDING == 0 )); then
    echo "No pending domain baselines on GPU ${GPU}"
    continue
  fi
  COMMAND="source ${CONDA_SETUP}; conda activate ${CONDA_ENV}; cd ${PROJECT_ROOT}; export ARTIFACT_ROOT=${ARTIFACT_ROOT}; export DATA_ROOT=${DATA_ROOT}; export PYTHON=${PYTHON}; export PYTHONPATH=${PROJECT_ROOT}/src:${PROJECT_ROOT}; set -u -o pipefail;${QUEUE} echo '[domain queue] FINISHED: GPU ${GPU}'"
  tmux new-session -d -s "${SESSION}" "bash -lc '${COMMAND}'"
  tmux set-option -t "${SESSION}" remain-on-exit on
  echo "Started ${SESSION}: ${PENDING} pending domain-baseline runs on GPU ${GPU}"
done
