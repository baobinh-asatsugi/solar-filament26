#!/usr/bin/env bash
set -euo pipefail
SEED="${1:?usage: bash scripts/evaluate_final_test.sh SEED}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/magfilo}"
export PROJECT_ROOT ARTIFACT_ROOT DATA_ROOT PYTHON
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"
TAG="magfilo_v1_m2lite_s${SEED}"
"${PYTHON}" -m filament.evaluate_lite \
  --manifest "${PROJECT_ROOT}/splits/magfilo_v1/manifest.csv" \
  --split test --confirm-final-test --prediction-run "${TAG}" \
  --selector-dir "${ARTIFACT_ROOT}/results/${TAG}_selector" \
  --output "${ARTIFACT_ROOT}/results/${TAG}_test.json"
