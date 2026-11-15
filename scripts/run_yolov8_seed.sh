#!/usr/bin/env bash
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/magfilo}"; PYTHON="${PYTHON:-python}"
export PROJECT_ROOT ARTIFACT_ROOT DATA_ROOT PYTHON
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"; RUN="magfilo_v1_yolov8l_s${SEED}"
"${PYTHON}" -m filament.train --run-name "${RUN}" --model yolov8l-seg.pt \
  --data-yaml "${ARTIFACT_ROOT}/datasets/m2_lite_magfilo_v1/dataset.yaml" --imgsz 2048 --epochs 30 --batch 1 \
  --workers 6 --device "${GPU}" --seed "${SEED}" --mosaic 0 --mixup 0 --translate 0 --scale .1 --fliplr .5 --flipud .5 --close-mosaic 0
OUT="${ARTIFACT_ROOT}/results/${RUN}"
"${PYTHON}" "${PROJECT_ROOT}/baselines/eval_yolo_pq.py" --checkpoint "${ARTIFACT_ROOT}/runs/${RUN}/weights/best.pt" \
  --manifest "${PROJECT_ROOT}/splits/magfilo_v1/manifest.csv" --split val --data-root "${DATA_ROOT}" \
  --artifact-root "${ARTIFACT_ROOT}" --device "${GPU}" --output-dir "${OUT}"
"${PYTHON}" "${PROJECT_ROOT}/baselines/eval_yolo_pq.py" --checkpoint "${ARTIFACT_ROOT}/runs/${RUN}/weights/best.pt" \
  --manifest "${PROJECT_ROOT}/splits/magfilo_v1/manifest.csv" --split test --data-root "${DATA_ROOT}" \
  --artifact-root "${ARTIFACT_ROOT}" --device "${GPU}" --output-dir "${OUT}" --frozen-config "${OUT}/val_best.json"
