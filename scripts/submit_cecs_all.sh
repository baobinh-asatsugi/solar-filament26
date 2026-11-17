#!/usr/bin/env bash
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/ml26}"
export DATA_ROOT="${DATA_ROOT:-${HOME}/datasets/magfilo}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/outputs}"
export VENV_PATH="${VENV_PATH:-${HOME}/.venv}"
export PYTHON="${VENV_PATH}/bin/python"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"
source "${VENV_PATH}/bin/activate"
mkdir -p "${ARTIFACT_ROOT}/logs" "${PROJECT_ROOT}/slurm_logs"
test -d "${DATA_ROOT}" || { echo "Missing DATA_ROOT: ${DATA_ROOT}" >&2; exit 1; }
test -x "${PYTHON}" || { echo "Missing virtualenv Python: ${PYTHON}" >&2; exit 1; }
test -s "splits/magfilo_v1/manifest.csv" || { echo "Run scripts/prepare_cecs.sh first" >&2; exit 1; }
test -s "${ARTIFACT_ROOT}/datasets/m2_lite_magfilo_v1/dataset.yaml" || {
  echo "Missing prepared M2/YOLO dataset. Run: bash scripts/prepare_cecs_data.sh" >&2
  exit 1
}
test -s "${ARTIFACT_ROOT}/baseline_data/annotations/train.json" || {
  echo "Missing prepared baseline annotations. Run: bash scripts/prepare_cecs_data.sh" >&2
  exit 1
}
"${PYTHON}" "${PROJECT_ROOT}/scripts/check_cecs_env.py"
mkdir -p "${ARTIFACT_ROOT}/environment"
"${PYTHON}" -m pip freeze --all > "${ARTIFACT_ROOT}/environment/requirements-cecs.lock.txt"

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
HEAVY_METHODS=(maskrcnn_consensus cascade_mask_rcnn htc queryinst mask2former)
LONG_METHODS=(m2_lite maskrcnn_consensus cascade_mask_rcnn htc queryinst mask2former)
is_heavy() {
  local candidate="$1"
  local heavy
  for heavy in "${HEAVY_METHODS[@]}"; do
    [[ "${candidate}" == "${heavy}" ]] && return 0
  done
  return 1
}
is_long() {
  local candidate="$1"
  local long_method
  for long_method in "${LONG_METHODS[@]}"; do
    [[ "${candidate}" == "${long_method}" ]] && return 0
  done
  return 1
}

for method in "${METHODS[@]}"; do
  dependency_args=()
  [[ -n "${previous_job}" ]] && dependency_args=(--dependency="afterany:${previous_job}")
  if is_heavy "${method}"; then
    array_spec="0-4%2"
    seeds_per_gpu=1
    echo "Resource policy ${method}: heavy, 1 seed/GPU, 2 GPUs, mem=16G/task"
  else
    array_spec="0-1%2"
    seeds_per_gpu=2
    echo "Resource policy ${method}: standard, 2 seeds/GPU, 2 GPUs, mem=16G/task"
  fi
  if is_long "${method}"; then
    time_limit="12:00:00"
  else
    time_limit="08:00:00"
  fi
  echo "Time policy ${method}: ${time_limit}"
  submit_args=(
    --parsable "${dependency_args[@]}" --job-name="ml26-${method}"
    --array="${array_spec}" --cpus-per-task=4 --mem=16G --time="${time_limit}"
    --export="ALL,METHOD=${method},SEEDS_PER_GPU=${seeds_per_gpu},PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},ARTIFACT_ROOT=${ARTIFACT_ROOT},VENV_PATH=${VENV_PATH}"
    jobs/cecs_method.sbatch
  )
  if ! response="$(sbatch "${submit_args[@]}")"; then
    echo "sbatch did not return cleanly; checking whether Slurm accepted ml26-${method}..." >&2
    sleep 2
    existing="$(squeue -h -u "${USER}" -n "ml26-${method}" -o '%A' | head -n 1)"
    if [[ -z "${existing}" ]]; then
      echo "No accepted job found for ml26-${method}; aborting." >&2
      exit 1
    fi
    response="${existing}"
    echo "Recovered accepted job after submit timeout: ${method} ${existing}" >&2
  fi
  job_id="${response%%;*}"
  job_ids+=("${job_id}")
  previous_job="${job_id}"
  echo "Submitted ${method}: ${job_id}"
done

if summary="$(sbatch --parsable --dependency="afterany:${previous_job}" \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},ARTIFACT_ROOT=${ARTIFACT_ROOT},VENV_PATH=${VENV_PATH}" \
  jobs/cecs_summary.sbatch)"; then
  echo "Submitted summary: ${summary%%;*}"
else
  echo "WARNING: summary submission timed out; run scripts/summarize_five_seed_results.sh after completion" >&2
fi
echo "Monitor: squeue --me -o '%.18i %.28j %.2t %.10M %.6D %R %b'"
