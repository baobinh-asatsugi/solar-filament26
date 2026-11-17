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
"${PYTHON}" -m pip install -r requirements.txt
"${PYTHON}" -m pip install -e . --no-deps
"${PYTHON}" - <<'PY'
import sys
import torch
print("Python:", sys.executable)
print("Torch:", torch.__version__, "CUDA build:", torch.version.cuda)
if torch.version.cuda is None:
    raise SystemExit("The CECS virtual environment has a CPU-only PyTorch build")
print("CUDA device visible on this node:", torch.cuda.is_available())
PY
bash scripts/prepare_fixed_split.sh
bash scripts/prepare_baselines.sh
if [[ "${INSTALL_MMDET:-1}" == "1" ]]; then
  bash scripts/setup_official_baselines.sh
fi
# OpenMIM/MMCV dependency resolution may upgrade NumPy indirectly. Restore the
# ABI-compatible pair last, after every optional baseline has been installed.
"${PYTHON}" -m pip install --force-reinstall --no-cache-dir \
  numpy==1.26.4 opencv-python-headless==4.11.0.86
"${PYTHON}" "${PROJECT_ROOT}/scripts/check_cecs_env.py"
echo "CECS preparation complete: ${DATA_ROOT} -> ${ARTIFACT_ROOT}"
