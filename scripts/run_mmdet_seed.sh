#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:?model key}"; SEED="${2:?seed}"; GPU="${3:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/magfilo}"; PYTHON="${PYTHON:-python}"
export PROJECT_ROOT ARTIFACT_ROOT DATA_ROOT PYTHON
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"; export CUDA_VISIBLE_DEVICES="${GPU}"
UPSTREAM="${ARTIFACT_ROOT}/upstream/mmdetection-v3.3.0"
case "${MODEL}" in
  mask_rcnn) CONFIG="configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py";;
  cascade_mask_rcnn) CONFIG="configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py";;
  htc) CONFIG="configs/htc/htc_r50_fpn_1x_coco.py";;
  solov2) CONFIG="configs/solov2/solov2_r50_fpn_1x_coco.py";;
  condinst) CONFIG="configs/condinst/condinst_r50_fpn_ms-poly-90k_coco_instance.py";;
  queryinst) CONFIG="configs/queryinst/queryinst_r50_fpn_1x_coco.py";;
  mask2former) CONFIG="configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco.py";;
  rtmdet_ins) CONFIG="configs/rtmdet/rtmdet-ins_l_8xb32-300e_coco.py";;
  *) echo "Unknown model: ${MODEL}" >&2; exit 2;;
esac
IMAGE_ROOT="${DATA_ROOT}/images"
WORK="${ARTIFACT_ROOT}/baseline_work/magfilo_v1/${MODEL}/seed${SEED}"; ANN="${ARTIFACT_ROOT}/baseline_data/annotations"
"${PYTHON}" "${PROJECT_ROOT}/baselines/mmdet_adapter.py" --config "${UPSTREAM}/${CONFIG}" \
  --train-json "${ANN}/train.json" --val-json "${ANN}/val.json" --test-json "${ANN}/test.json" \
  --image-root "${IMAGE_ROOT}" --work-dir "${WORK}" --seed "${SEED}"
CHECKPOINT="$(cat "${WORK}/best_checkpoint")"; [[ "${CHECKPOINT}" = /* ]] || CHECKPOINT="${WORK}/${CHECKPOINT}"
OUT="${ARTIFACT_ROOT}/results/baseline_magfilo_v1_${MODEL}_s${SEED}"
"${PYTHON}" "${PROJECT_ROOT}/baselines/eval_mmdet_pq.py" --config "${WORK}/adapted_config.py" --checkpoint "${CHECKPOINT}" \
  --manifest "${PROJECT_ROOT}/splits/magfilo_v1/manifest.csv" --split val --data-root "${DATA_ROOT}" \
  --artifact-root "${ARTIFACT_ROOT}" --device cuda:0 --output-dir "${OUT}"
"${PYTHON}" "${PROJECT_ROOT}/baselines/eval_mmdet_pq.py" --config "${WORK}/adapted_config.py" --checkpoint "${CHECKPOINT}" \
  --manifest "${PROJECT_ROOT}/splits/magfilo_v1/manifest.csv" --split test --data-root "${DATA_ROOT}" \
  --artifact-root "${ARTIFACT_ROOT}" --device cuda:0 --output-dir "${OUT}" --frozen-config "${OUT}/val_best.json"
