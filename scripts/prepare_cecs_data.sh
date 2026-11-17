#!/usr/bin/env bash
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/ml26}"
export DATA_ROOT="${DATA_ROOT:-${HOME}/datasets/magfilo}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/outputs}"
export VENV_PATH="${VENV_PATH:-${HOME}/.venv}"
export PYTHON="${PYTHON:-${VENV_PATH}/bin/python}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"

cd "${PROJECT_ROOT}"
source "${VENV_PATH}/bin/activate"
mkdir -p "${ARTIFACT_ROOT}/logs"
test -d "${DATA_ROOT}" || { echo "Missing DATA_ROOT: ${DATA_ROOT}" >&2; exit 1; }
"${PYTHON}" "${PROJECT_ROOT}/scripts/check_cecs_env.py"

bash scripts/prepare_fixed_split.sh
bash scripts/prepare_baselines.sh

test -s "${ARTIFACT_ROOT}/datasets/m2_lite_magfilo_v1/dataset.yaml"
test -s "${ARTIFACT_ROOT}/baseline_data/annotations/train.json"
echo "CECS data preparation complete: ${DATA_ROOT} -> ${ARTIFACT_ROOT}"
