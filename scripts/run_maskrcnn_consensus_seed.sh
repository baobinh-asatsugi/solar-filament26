#!/usr/bin/env bash
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"; DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/magfilo}"
PYTHON="${PYTHON:-python}"; export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"
export PROJECT_ROOT ARTIFACT_ROOT DATA_ROOT PYTHON
MANIFEST="${PROJECT_ROOT}/splits/magfilo_v1/manifest.csv"
COMMON=(--data-root "${DATA_ROOT}" --manifest "${MANIFEST}" --artifact-root "${ARTIFACT_ROOT}" --seed "${SEED}" --device "${GPU}")
"${PYTHON}" "${PROJECT_ROOT}/baselines/maskrcnn_consensus.py" train "${COMMON[@]}"
"${PYTHON}" "${PROJECT_ROOT}/baselines/maskrcnn_consensus.py" eval "${COMMON[@]}" --split val
RESULT="${ARTIFACT_ROOT}/results/baseline_magfilo_v1_maskrcnn_consensus_s${SEED}"
"${PYTHON}" "${PROJECT_ROOT}/baselines/maskrcnn_consensus.py" eval "${COMMON[@]}" --split test --frozen-config "${RESULT}/val_best.json"
