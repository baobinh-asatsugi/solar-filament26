#!/usr/bin/env bash
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
ZIP_PATH="${ZIP_PATH:-${PROJECT_ROOT}/filament-segmentation-2026.zip}"; PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"; RUN="yolov8l_s${SEED}"
"${PYTHON}" -m filament.train --run-name "${RUN}" --model yolov8l-seg.pt \
  --data-yaml "${ARTIFACT_ROOT}/datasets/m2_lite_fixed/dataset.yaml" --imgsz 2048 --epochs 30 --batch 1 \
  --workers 6 --device "${GPU}" --seed "${SEED}" --mosaic 0 --mixup 0 --translate 0 --scale .1 --fliplr .5 --flipud .5 --close-mosaic 0
OUT="${ARTIFACT_ROOT}/results/${RUN}"
"${PYTHON}" "${PROJECT_ROOT}/baselines/eval_yolo_pq.py" --checkpoint "${ARTIFACT_ROOT}/runs/${RUN}/weights/best.pt" \
  --manifest "${ARTIFACT_ROOT}/results/fixed_split/manifest.csv" --split val --zip-path "${ZIP_PATH}" \
  --artifact-root "${ARTIFACT_ROOT}" --device "${GPU}" --output-dir "${OUT}"
"${PYTHON}" "${PROJECT_ROOT}/baselines/eval_yolo_pq.py" --checkpoint "${ARTIFACT_ROOT}/runs/${RUN}/weights/best.pt" \
  --manifest "${ARTIFACT_ROOT}/results/fixed_split/manifest.csv" --split test --zip-path "${ZIP_PATH}" \
  --artifact-root "${ARTIFACT_ROOT}" --device "${GPU}" --output-dir "${OUT}" --frozen-config "${OUT}/val_best.json"
