#!/usr/bin/env bash
set -euo pipefail
METHOD="${1:?usage: run_cecs_method_seed.sh METHOD SEED [DEVICE]}"
SEED="${2:?seed}"
DEVICE="${3:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/ml26}"
export PROJECT_ROOT

case "${METHOD}" in
  m2_lite) exec bash "${PROJECT_ROOT}/scripts/train_one_seed.sh" "${SEED}" "${DEVICE}" ;;
  yolov8l) exec bash "${PROJECT_ROOT}/scripts/run_yolov8_seed.sh" "${SEED}" "${DEVICE}" ;;
  unetpp_watershed) exec bash "${PROJECT_ROOT}/scripts/run_unetpp_watershed_seed.sh" "${SEED}" "${DEVICE}" ;;
  fullres_unet|edgeattnet) exec bash "${PROJECT_ROOT}/scripts/run_semantic_baseline_seed.sh" "${METHOD}" "${SEED}" "${DEVICE}" ;;
  maskrcnn_consensus) exec bash "${PROJECT_ROOT}/scripts/run_maskrcnn_consensus_seed.sh" "${SEED}" "${DEVICE}" ;;
  detr_crop_unet) exec bash "${PROJECT_ROOT}/scripts/run_detr_crop_unet_seed.sh" "${SEED}" "${DEVICE}" ;;
  mask_rcnn|cascade_mask_rcnn|htc|solov2|condinst|queryinst|mask2former|rtmdet_ins)
    exec bash "${PROJECT_ROOT}/scripts/run_mmdet_seed.sh" "${METHOD}" "${SEED}" "${DEVICE}"
    ;;
  *) echo "Unknown CECS method: ${METHOD}" >&2; exit 2 ;;
esac
