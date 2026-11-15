#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src"
cd "${PROJECT_ROOT}"

"${PYTHON}" -m filament.create_fixed_annotated_split \
  --output-dir "${ARTIFACT_ROOT}/results/fixed_split" \
  --seed 3020 --val-fraction 0.15 --test-fraction 0.20

"${PYTHON}" -m filament.prepare \
  --input-manifest "${ARTIFACT_ROOT}/results/fixed_split/manifest.csv" \
  --dataset-name m2_lite_fixed \
  --annotation-strategy medoid \
  --val-annotation-strategy all \
  --seed 3020

echo "Manifest: ${ARTIFACT_ROOT}/results/fixed_split/manifest.csv"
echo "YOLO data: ${ARTIFACT_ROOT}/datasets/m2_lite_fixed/dataset.yaml"
