#!/bin/bash
#SBATCH --job-name=ml26-method
#SBATCH --qos=student
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=slurm_logs/%x-%A_%a.out

set -uo pipefail
METHOD="${METHOD:?submit with --export=ALL,METHOD=...}"
PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/ml26}"
DATA_ROOT="${DATA_ROOT:-${HOME}/datasets/magfilo}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/outputs}"
VENV_PATH="${VENV_PATH:-${HOME}/.venv}"
export PROJECT_ROOT DATA_ROOT ARTIFACT_ROOT VENV_PATH
export PYTHON="${VENV_PATH}/bin/python"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"

# Two seeds share each allocated GPU. Memory-sensitive batch sizes are reduced;
# model architectures, objectives, schedules, split and 2048 input stay fixed.
export MMDET_BATCH_SIZE="${MMDET_BATCH_SIZE:-1}"
export BASELINE_BATCH="${BASELINE_BATCH:-1}"
export UNETPP_BATCH="${UNETPP_BATCH:-1}"
export UNETPP_WORKERS="${UNETPP_WORKERS:-1}"
export M2_SEMANTIC_BATCH="${M2_SEMANTIC_BATCH:-1}"
export M2_WORKERS="${M2_WORKERS:-1}"
export YOLO_WORKERS="${YOLO_WORKERS:-1}"
export SIRA_BATCH="${SIRA_BATCH:-1}"
export REFINE_BATCH="${REFINE_BATCH:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

cd "${PROJECT_ROOT}"
source "${VENV_PATH}/bin/activate"
mkdir -p "${ARTIFACT_ROOT}/logs" "${PROJECT_ROOT}/slurm_logs"
"${PYTHON}" "${PROJECT_ROOT}/scripts/check_cecs_env.py"

if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
  seeds=(3020 3021)
  tail_seed=3024
else
  seeds=(3022 3023)
  tail_seed=""
fi

run_seed() {
  local seed="$1"
  local log="${ARTIFACT_ROOT}/logs/cecs_${METHOD}_s${seed}.log"
  echo "[$(date --iso-8601=seconds)] START method=${METHOD} seed=${seed} job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}" | tee -a "${log}"
  set +e
  bash "${PROJECT_ROOT}/scripts/run_cecs_method_seed.sh" "${METHOD}" "${seed}" 0 2>&1 | tee -a "${log}"
  local status="${PIPESTATUS[0]}"
  set -e
  echo "[$(date --iso-8601=seconds)] END method=${METHOD} seed=${seed} status=${status}" | tee -a "${log}"
  return "${status}"
}

pids=()
seed_names=()
for seed in "${seeds[@]}"; do
  run_seed "${seed}" &
  pids+=("$!")
  seed_names+=("${seed}")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "FAILED method=${METHOD} seed=${seed_names[$index]}" >&2
    failed=1
  fi
done

if [[ -n "${tail_seed}" ]] && ! run_seed "${tail_seed}"; then
  echo "FAILED method=${METHOD} seed=${tail_seed}" >&2
  failed=1
fi
exit "${failed}"
