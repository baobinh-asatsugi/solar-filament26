#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/magfilo}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"
mkdir -p "${ARTIFACT_ROOT}/logs"
"${PYTHON}" "${PROJECT_ROOT}/baselines/export_coco_splits.py" \
  --manifest "${PROJECT_ROOT}/splits/magfilo_v1/manifest.csv" \
  --artifact-root "${ARTIFACT_ROOT}" --data-root "${DATA_ROOT}" \
  --output-dir "${ARTIFACT_ROOT}/baseline_data/annotations"
"${PYTHON}" "${PROJECT_ROOT}/baselines/export_yolo_detection.py" \
  --data-root "${DATA_ROOT}" \
  --manifest "${PROJECT_ROOT}/splits/magfilo_v1/manifest.csv" \
  --output-dir "${ARTIFACT_ROOT}/baseline_data/yolo_detection"
