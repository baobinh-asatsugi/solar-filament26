#!/usr/bin/env bash
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
ZIP_PATH="${ZIP_PATH:-${PROJECT_ROOT}/filament-segmentation-2026.zip}"; PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"; OUT="${ARTIFACT_ROOT}/baseline_work/unetpp_watershed/seed${SEED}"
COMMON=(--manifest "${ARTIFACT_ROOT}/results/fixed_split/manifest.csv" --zip-path "${ZIP_PATH}" --artifact-root "${ARTIFACT_ROOT}" --output-dir "${OUT}" --seed "${SEED}" --device "${GPU}")
"${PYTHON}" "${PROJECT_ROOT}/baselines/unetpp_watershed.py" train "${COMMON[@]}"
"${PYTHON}" "${PROJECT_ROOT}/baselines/unetpp_watershed.py" eval "${COMMON[@]}" --split val
"${PYTHON}" "${PROJECT_ROOT}/baselines/unetpp_watershed.py" eval "${COMMON[@]}" --split test
RESULT="${ARTIFACT_ROOT}/results/baseline_unetpp_watershed_s${SEED}"; mkdir -p "${RESULT}"
cp "${OUT}/val_best.json" "${OUT}/val_pq.csv" "${OUT}/test_best.json" "${OUT}/test_pq.csv" "${RESULT}/"
