#!/usr/bin/env bash
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/ml26}"
export DATA_ROOT="${DATA_ROOT:-${HOME}/datasets/magfilom}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-${HOME}/ml26_artifacts}"
export VENV_PATH="${VENV_PATH:-${HOME}/.venv}"
export PYTHON="${VENV_PATH}/bin/python"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"
source "${VENV_PATH}/bin/activate"
mkdir -p "${ARTIFACT_ROOT}/logs" "${PROJECT_ROOT}/slurm_logs"
test -d "${DATA_ROOT}" || { echo "Missing DATA_ROOT: ${DATA_ROOT}" >&2; exit 1; }
test -x "${PYTHON}" || { echo "Missing virtualenv Python: ${PYTHON}" >&2; exit 1; }
test -s "splits/magfilo_v1/manifest.csv" || { echo "Run scripts/prepare_cecs.sh first" >&2; exit 1; }

METHODS=(
  m2_lite yolov8l unetpp_watershed
  fullres_unet edgeattnet maskrcnn_consensus detr_crop_unet
  mask_rcnn cascade_mask_rcnn htc solov2
  condinst queryinst mask2former rtmdet_ins
)
if (( $# > 0 )); then
  METHODS=("$@")
fi
job_ids=()
previous_job=""
for method in "${METHODS[@]}"; do
  dependency_args=()
  [[ -n "${previous_job}" ]] && dependency_args=(--dependency="afterany:${previous_job}")
  response="$(sbatch --parsable "${dependency_args[@]}" --job-name="ml26-${method}" \
    --export="ALL,METHOD=${method},PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},ARTIFACT_ROOT=${ARTIFACT_ROOT},VENV_PATH=${VENV_PATH}" \
    jobs/cecs_method.sbatch)"
  job_id="${response%%;*}"
  job_ids+=("${job_id}")
  previous_job="${job_id}"
  echo "Submitted ${method}: ${job_id}"
done

summary="$(sbatch --parsable --dependency="afterany:${previous_job}" \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},ARTIFACT_ROOT=${ARTIFACT_ROOT},VENV_PATH=${VENV_PATH}" \
  jobs/cecs_summary.sbatch)"
echo "Submitted summary: ${summary%%;*}"
echo "Monitor: squeue --me -o '%.18i %.28j %.2t %.10M %.6D %R %b'"
