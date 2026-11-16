#!/usr/bin/env bash
set -euo pipefail
SEED="${1:?usage: bash scripts/train_one_seed.sh SEED [GPU]}"
GPU="${2:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/magfilo}"
export PROJECT_ROOT ARTIFACT_ROOT DATA_ROOT PYTHON
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"
mkdir -p "${ARTIFACT_ROOT}/logs"

MANIFEST="${PROJECT_ROOT}/splits/magfilo_v1/manifest.csv"
DATA="${ARTIFACT_ROOT}/datasets/m2_lite_magfilo_v1/dataset.yaml"
TAG="magfilo_v1_m2lite_s${SEED}"
FINAL_RESULT="${ARTIFACT_ROOT}/results/${TAG}_val.json"
if [[ -s "${FINAL_RESULT}" ]]; then
  echo "[M2-Lite seed ${SEED}] validation result exists; skipping completed run"
  exit 0
fi
DET="${TAG}_detector"
SEM="${TAG}_semantic"
SIRA="${TAG}_sira"
SELECTOR="${TAG}_selector"
DET_BEST="${ARTIFACT_ROOT}/runs/${DET}/weights/best.pt"
SEM_BEST="${ARTIFACT_ROOT}/runs/${SEM}/best.pt"
SIRA_BEST="${ARTIFACT_ROOT}/runs/${SIRA}/best.pt"
SIRA_COMPLETE="${ARTIFACT_ROOT}/runs/${SIRA}/train_complete.json"
SELECTOR_DIR="${ARTIFACT_ROOT}/results/${SELECTOR}"
PREDICTION_ROOT="${ARTIFACT_ROOT}/predictions/${TAG}"
SIRA_BATCH="${SIRA_BATCH:-2}"
REFINE_BATCH="${REFINE_BATCH:-2}"

if [[ -s "${DET_BEST}" ]]; then
  echo "[M2-Lite seed ${SEED}] detector checkpoint exists; skipping YOLO training"
else
  "${PYTHON}" -m filament.train --run-name "${DET}" --model yolov8l-seg.pt --data-yaml "${DATA}" \
    --imgsz 2048 --epochs 30 --batch 1 --workers 6 --device "${GPU}" --seed "${SEED}" \
    --mosaic 0 --mixup 0 --translate 0 --scale 0.1 --fliplr 0.5 --flipud 0.5 --close-mosaic 0
fi

if [[ -s "${SEM_BEST}" ]]; then
  echo "[M2-Lite seed ${SEED}] semantic checkpoint exists; skipping semantic training"
else
  "${PYTHON}" -m filament.train_semantic --run-name "${SEM}" --manifest "${MANIFEST}" \
    --architecture unetplusplus --encoder efficientnet-b3 --feature-mode radial \
    --annotation-strategy medoid --val-annotation-strategy duplicate --tile-size 768 \
    --train-samples 4000 --val-samples 1200 --positive-fraction 0.75 --epochs 20 \
    --batch 2 --workers 6 --centerline-weight 0.1 --skeleton-recall-weight 0.2 \
    --device "${GPU}" --seed "${SEED}"
fi

for SPLIT in train val test; do
  if "${PYTHON}" -m filament.pipeline_state --manifest "${MANIFEST}" --prediction-root "${PREDICTION_ROOT}" --split "${SPLIT}" --stage prediction; then
    echo "[M2-Lite seed ${SEED}] ${SPLIT} prediction cache complete; skipping"
  else
    "${PYTHON}" -m filament.predict_split --manifest "${MANIFEST}" --split "${SPLIT}" --run-name "${TAG}" \
      --detector "${DET_BEST}" --semantic "${SEM_BEST}" --device "${GPU}"
    "${PYTHON}" -m filament.pipeline_state --manifest "${MANIFEST}" --prediction-root "${PREDICTION_ROOT}" --split "${SPLIT}" --stage prediction
  fi
done

if [[ -s "${SIRA_COMPLETE}" && -s "${SIRA_BEST}" ]]; then
  echo "[M2-Lite seed ${SEED}] SIRA training complete; skipping"
else
  "${PYTHON}" -m filament.train_sira --run-name "${SIRA}" --manifest "${MANIFEST}" \
    --bank-root "${PREDICTION_ROOT}/raw" --semantic-root "${PREDICTION_ROOT}/semantic" \
    --crop-size 512 --architecture unet --encoder resnet34 --semantic-channel \
    --epochs 25 --batch "${SIRA_BATCH}" --workers 6 --device "${GPU}" --seed "${SEED}" --resume
fi

for SPLIT in train val test; do
  if "${PYTHON}" -m filament.pipeline_state --manifest "${MANIFEST}" --prediction-root "${PREDICTION_ROOT}" --split "${SPLIT}" --stage refined; then
    echo "[M2-Lite seed ${SEED}] ${SPLIT} refined cache complete; skipping"
  else
    "${PYTHON}" -m filament.refine_split --manifest "${MANIFEST}" --split "${SPLIT}" --run-name "${TAG}" \
      --checkpoint "${SIRA_BEST}" --device "${GPU}" --batch "${REFINE_BATCH}"
    "${PYTHON}" -m filament.pipeline_state --manifest "${MANIFEST}" --prediction-root "${PREDICTION_ROOT}" --split "${SPLIT}" --stage refined
  fi
done

if [[ -s "${SELECTOR_DIR}/quality.joblib" && -s "${SELECTOR_DIR}/best_decoder.json" ]]; then
  echo "[M2-Lite seed ${SEED}] selector result exists; skipping selector training"
else
  "${PYTHON}" -m filament.train_selector --manifest "${MANIFEST}" --run-name "${SELECTOR}" \
    --prediction-run "${TAG}" --seed "${SEED}"
fi

"${PYTHON}" -m filament.evaluate_lite --manifest "${MANIFEST}" --split val --prediction-run "${TAG}" \
  --selector-dir "${SELECTOR_DIR}" --output "${FINAL_RESULT}"

echo "Seed ${SEED} complete. Validate first; final test remains locked."
