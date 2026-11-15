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
DATA="${ARTIFACT_ROOT}/datasets/m2_lite_fixed/dataset.yaml"
TAG="m2lite_s${SEED}"
DET="${TAG}_detector"
SEM="${TAG}_semantic"
SIRA="${TAG}_sira"
SELECTOR="${TAG}_selector"

"${PYTHON}" -m filament.train --run-name "${DET}" --model yolov8l-seg.pt --data-yaml "${DATA}" \
  --imgsz 2048 --epochs 30 --batch 1 --workers 6 --device "${GPU}" --seed "${SEED}" \
  --mosaic 0 --mixup 0 --translate 0 --scale 0.1 --fliplr 0.5 --flipud 0.5 --close-mosaic 0

"${PYTHON}" -m filament.train_semantic --run-name "${SEM}" --manifest "${MANIFEST}" \
  --architecture unetplusplus --encoder efficientnet-b3 --feature-mode radial \
  --annotation-strategy medoid --val-annotation-strategy duplicate --tile-size 768 \
  --train-samples 4000 --val-samples 1200 --positive-fraction 0.75 --epochs 20 \
  --batch 2 --workers 6 --centerline-weight 0.1 --skeleton-recall-weight 0.2 \
  --device "${GPU}" --seed "${SEED}"

for SPLIT in train val test; do
  "${PYTHON}" -m filament.predict_split --manifest "${MANIFEST}" --split "${SPLIT}" --run-name "${TAG}" \
    --detector "${ARTIFACT_ROOT}/runs/${DET}/weights/best.pt" \
    --semantic "${ARTIFACT_ROOT}/runs/${SEM}/best.pt" --device "${GPU}"
done

"${PYTHON}" -m filament.train_sira --run-name "${SIRA}" --manifest "${MANIFEST}" \
  --bank-root "${ARTIFACT_ROOT}/predictions/${TAG}/raw" \
  --semantic-root "${ARTIFACT_ROOT}/predictions/${TAG}/semantic" \
  --crop-size 512 --architecture unet --encoder resnet34 --semantic-channel \
  --epochs 25 --batch 4 --workers 6 --device "${GPU}" --seed "${SEED}"

for SPLIT in train val test; do
  "${PYTHON}" -m filament.refine_split --manifest "${MANIFEST}" --split "${SPLIT}" --run-name "${TAG}" \
    --checkpoint "${ARTIFACT_ROOT}/runs/${SIRA}/best.pt" --device "${GPU}" --batch 4
done

"${PYTHON}" -m filament.train_selector --manifest "${MANIFEST}" --run-name "${SELECTOR}" \
  --prediction-run "${TAG}" --seed "${SEED}"

"${PYTHON}" -m filament.evaluate_lite --manifest "${MANIFEST}" --split val --prediction-run "${TAG}" \
  --selector-dir "${ARTIFACT_ROOT}/results/${SELECTOR}" \
  --output "${ARTIFACT_ROOT}/results/${TAG}_val.json"

echo "Seed ${SEED} complete. Validate first; final test remains locked."
