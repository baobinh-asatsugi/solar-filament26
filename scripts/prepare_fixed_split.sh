#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/magfilo}"
PYTHON="${PYTHON:-python}"
export PROJECT_ROOT ARTIFACT_ROOT DATA_ROOT PYTHON
export PYTHONPATH="${PROJECT_ROOT}/src"
cd "${PROJECT_ROOT}"
mkdir -p "${ARTIFACT_ROOT}/logs"

"${PYTHON}" -m filament.create_fixed_annotated_split \
  --data-root "${DATA_ROOT}" \
  --output-dir "${PROJECT_ROOT}/splits/magfilo_v1" \
  --seed 3020 --val-fraction 0.15 --test-fraction 0.20 --physical-images 958

"${PYTHON}" -m filament.prepare \
  --data-root "${DATA_ROOT}" \
  --input-manifest "${PROJECT_ROOT}/splits/magfilo_v1/manifest.csv" \
  --dataset-name m2_lite_magfilo_v1 \
  --annotation-strategy medoid \
  --val-annotation-strategy all \
  --seed 3020

echo "Manifest: ${PROJECT_ROOT}/splits/magfilo_v1/manifest.csv"
echo "YOLO data: ${ARTIFACT_ROOT}/datasets/m2_lite_magfilo_v1/dataset.yaml"
