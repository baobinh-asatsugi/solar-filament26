#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
ZIP_PATH="${ZIP_PATH:-${PROJECT_ROOT}/filament-segmentation-2026.zip}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"
"${PYTHON}" "${PROJECT_ROOT}/baselines/export_coco_splits.py" \
  --manifest "${ARTIFACT_ROOT}/results/fixed_split/manifest.csv" \
  --artifact-root "${ARTIFACT_ROOT}" --zip-path "${ZIP_PATH}" \
  --output-dir "${ARTIFACT_ROOT}/baseline_data/annotations"
