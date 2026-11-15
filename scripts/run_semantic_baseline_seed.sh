#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:?fullres_unet or edgeattnet}"; SEED="${2:?seed}"; GPU="${3:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"; DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/magfilo}"
PYTHON="${PYTHON:-python}"; export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"
export PROJECT_ROOT ARTIFACT_ROOT DATA_ROOT PYTHON
MANIFEST="${PROJECT_ROOT}/splits/magfilo_v1/manifest.csv"
COMMON=(--model "${MODEL}" --data-root "${DATA_ROOT}" --manifest "${MANIFEST}" --artifact-root "${ARTIFACT_ROOT}" --seed "${SEED}" --device "${GPU}")
TRAIN_ARGS=()
[[ "${MODEL}" == "edgeattnet" ]] && TRAIN_ARGS=(--epochs 50 --batch 1 --crop-size 512)
"${PYTHON}" "${PROJECT_ROOT}/baselines/semantic_fullres.py" train "${COMMON[@]}" "${TRAIN_ARGS[@]}"
"${PYTHON}" "${PROJECT_ROOT}/baselines/semantic_fullres.py" eval "${COMMON[@]}" --split val
RESULT="${ARTIFACT_ROOT}/results/baseline_${MODEL}_s${SEED}"
"${PYTHON}" "${PROJECT_ROOT}/baselines/semantic_fullres.py" eval "${COMMON[@]}" --split test --frozen-config "${RESULT}/val_best.json"
