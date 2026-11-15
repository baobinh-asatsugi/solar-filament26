#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-python}"
UPSTREAM="${ARTIFACT_ROOT}/upstream/mmdetection-v3.3.0"
mkdir -p "${ARTIFACT_ROOT}/upstream"
if [[ ! -d "${UPSTREAM}/.git" ]]; then
  git clone --depth 1 --branch v3.3.0 https://github.com/open-mmlab/mmdetection.git "${UPSTREAM}"
fi
"${PYTHON}" -m pip install -U openmim
"${PYTHON}" -m mim install 'mmengine>=0.7.1,<1.0.0' 'mmcv>=2.0.0rc4,<2.2.0'
"${PYTHON}" -m pip install numpy==1.26.4 pandas scipy pycocotools opencv-python-headless tqdm PyYAML
"${PYTHON}" -m pip install -e "${UPSTREAM}" --no-deps
"${PYTHON}" - <<'PY'
import mmcv, mmengine, mmdet
print("mmcv", mmcv.__version__, "mmengine", mmengine.__version__, "mmdet", mmdet.__version__)
PY
"${PYTHON}" "${PROJECT_ROOT}/baselines/smoke_mmdet_configs.py" --upstream "${UPSTREAM}"
