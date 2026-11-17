#!/usr/bin/env bash
set -euo pipefail
SEED="${1:?seed}"; GPU="${2:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"; DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/magfilo}"
PYTHON="${PYTHON:-python}"; export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"
export PROJECT_ROOT ARTIFACT_ROOT DATA_ROOT PYTHON
MANIFEST="${PROJECT_ROOT}/splits/magfilo_v1/manifest.csv"
DETECTION_DATA="${ARTIFACT_ROOT}/baseline_data/magfilo_v1_yolo_detection/dataset.yaml"
RESULT="${ARTIFACT_ROOT}/results/baseline_magfilo_v1_detr_crop_unet_s${SEED}"
if [[ -s "${RESULT}/test_best.json" ]]; then
  echo "[detr_crop_unet seed ${SEED}] test result exists; skipping completed run"
  exit 0
fi
"${PYTHON}" "${PROJECT_ROOT}/baselines/train_rtdetr.py" --data "${DETECTION_DATA}" --artifact-root "${ARTIFACT_ROOT}" --seed "${SEED}" --device "${GPU}" --workers "${YOLO_WORKERS:-6}"
DETECTOR="${ARTIFACT_ROOT}/runs/magfilo_v1_detr_crop_unet_detector_s${SEED}/weights/best.pt"
COMMON=(--data-root "${DATA_ROOT}" --manifest "${MANIFEST}" --artifact-root "${ARTIFACT_ROOT}" --seed "${SEED}" --device "${GPU}" --detector "${DETECTOR}")
"${PYTHON}" "${PROJECT_ROOT}/baselines/detr_crop_unet.py" train-refiner "${COMMON[@]}"
"${PYTHON}" "${PROJECT_ROOT}/baselines/detr_crop_unet.py" eval "${COMMON[@]}" --split val
"${PYTHON}" "${PROJECT_ROOT}/baselines/detr_crop_unet.py" eval "${COMMON[@]}" --split test --frozen-config "${RESULT}/val_best.json"
