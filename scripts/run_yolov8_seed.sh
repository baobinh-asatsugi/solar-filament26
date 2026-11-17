#!/usr/bin/env bash
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/magfilo}"; PYTHON="${PYTHON:-python}"
export PROJECT_ROOT ARTIFACT_ROOT DATA_ROOT PYTHON
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"; RUN="magfilo_v1_yolov8l_s${SEED}"
LOG="${ARTIFACT_ROOT}/logs/yolov8_s${SEED}.log"
BEST="${ARTIFACT_ROOT}/runs/${RUN}/weights/best.pt"
YOLO_WORKERS="${YOLO_WORKERS:-6}"
YOLO_MODEL="${YOLO_MODEL:-${PROJECT_ROOT}/weights/yolov8l-seg.pt}"
if [[ ! -s "${YOLO_MODEL}" ]]; then
  echo "[YOLO seed ${SEED}] local weights missing; falling back to Ultralytics download" | tee -a "${LOG}"
  YOLO_MODEL="yolov8l-seg.pt"
fi
GATE=(bash "${PROJECT_ROOT}/scripts/run_with_gpu_limit.sh" "${GPU}" "${LOG}")

if [[ ! -s "${BEST}" ]]; then
  "${GATE[@]}" "${PYTHON}" -m filament.train --run-name "${RUN}" --model "${YOLO_MODEL}" \
    --data-yaml "${ARTIFACT_ROOT}/datasets/m2_lite_magfilo_v1/dataset.yaml" --imgsz 2048 --epochs 30 --batch 1 \
    --workers "${YOLO_WORKERS}" --device "${GPU}" --seed "${SEED}" --mosaic 0 --mixup 0 --translate 0 --scale .1 --fliplr .5 --flipud .5 --close-mosaic 0
else
  echo "[YOLO seed ${SEED}] existing best checkpoint; skipping training" | tee -a "${LOG}"
fi
OUT="${ARTIFACT_ROOT}/results/${RUN}"
if [[ ! -s "${OUT}/val_best.json" ]]; then
  "${GATE[@]}" "${PYTHON}" "${PROJECT_ROOT}/baselines/eval_yolo_pq.py" --checkpoint "${BEST}" \
    --manifest "${PROJECT_ROOT}/splits/magfilo_v1/manifest.csv" --split val --data-root "${DATA_ROOT}" \
    --artifact-root "${ARTIFACT_ROOT}" --device "${GPU}" --output-dir "${OUT}"
else
  echo "[YOLO seed ${SEED}] existing val result; skipping val" | tee -a "${LOG}"
fi
if [[ ! -s "${OUT}/test_best.json" ]]; then
  "${GATE[@]}" "${PYTHON}" "${PROJECT_ROOT}/baselines/eval_yolo_pq.py" --checkpoint "${BEST}" \
    --manifest "${PROJECT_ROOT}/splits/magfilo_v1/manifest.csv" --split test --data-root "${DATA_ROOT}" \
    --artifact-root "${ARTIFACT_ROOT}" --device "${GPU}" --output-dir "${OUT}" --frozen-config "${OUT}/val_best.json"
else
  echo "[YOLO seed ${SEED}] existing test result; skipping test" | tee -a "${LOG}"
fi
