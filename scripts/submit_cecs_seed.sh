#!/usr/bin/env bash
set -euo pipefail
METHOD="${1:?usage: bash scripts/submit_cecs_seed.sh METHOD SEED}"
SEED="${2:?usage: bash scripts/submit_cecs_seed.sh METHOD SEED}"
case "${SEED}" in 3020|3021|3022|3023|3024) ;; *) echo "Unsupported seed: ${SEED}" >&2; exit 2 ;; esac

export PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/ml26}"
export DATA_ROOT="${DATA_ROOT:-${HOME}/datasets/magfilo}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/outputs}"
export VENV_PATH="${VENV_PATH:-${HOME}/.venv}"
export PYTHON="${VENV_PATH}/bin/python"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"
source "${VENV_PATH}/bin/activate"
mkdir -p "${ARTIFACT_ROOT}/logs" "${PROJECT_ROOT}/slurm_logs"
"${PYTHON}" "${PROJECT_ROOT}/scripts/check_cecs_env.py"

case "${METHOD}" in
  m2_lite|cascade_mask_rcnn|htc|queryinst|mask2former) TIME_LIMIT="12:00:00" ;;
  yolov8l|unetpp_watershed|fullres_unet|edgeattnet|maskrcnn_consensus|detr_crop_unet|mask_rcnn|solov2|condinst|rtmdet_ins) TIME_LIMIT="08:00:00" ;;
  *) echo "Unknown method: ${METHOD}" >&2; exit 2 ;;
esac

response="$(sbatch --parsable --job-name="ml26-${METHOD}-s${SEED}" \
  --array=0 --cpus-per-task=4 --mem=16G --time="${TIME_LIMIT}" \
  --export="ALL,METHOD=${METHOD},SINGLE_SEED=${SEED},SEEDS_PER_GPU=1,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},ARTIFACT_ROOT=${ARTIFACT_ROOT},VENV_PATH=${VENV_PATH}" \
  jobs/cecs_method.sbatch)"
echo "Submitted ${METHOD} seed ${SEED}: ${response%%;*} (time=${TIME_LIMIT}, gpu=1, mem=16G)"
