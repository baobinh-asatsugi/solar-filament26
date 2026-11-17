#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-python}"
UPSTREAM="${ARTIFACT_ROOT}/upstream/mmdetection-v3.3.0"
mkdir -p "${ARTIFACT_ROOT}/upstream"
"${PYTHON}" - <<'PY'
import sys
import torch
print("Phase 9 Python:", sys.executable)
print("Phase 9 PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)
if sys.prefix == sys.base_prefix:
    raise RuntimeError("MMDetection setup must run inside an activated virtual environment")
PY
if [[ ! -d "${UPSTREAM}/.git" ]]; then
  git clone --depth 1 --branch v3.3.0 https://github.com/open-mmlab/mmdetection.git "${UPSTREAM}"
fi
"${PYTHON}" -m pip install -U openmim
"${PYTHON}" -m mim install 'mmengine>=0.7.1,<1.0.0' 'mmcv>=2.0.0rc4,<2.2.0'
"${PYTHON}" -m pip install \
  numpy==1.26.4 pandas scipy pycocotools opencv-python-headless tqdm PyYAML \
  shapely terminaltables
# MMDetection's setup imports torch while determining build requirements and
# its pinned backend predates PEP 660 editable installs. Build a regular local
# wheel without isolation so it sees torch from finetune-sd.
"${PYTHON}" -m pip install "${UPSTREAM}" --no-deps --no-build-isolation
"${PYTHON}" - <<'PY'
import mmcv, mmengine, mmdet, shapely, terminaltables
print("mmcv", mmcv.__version__, "mmengine", mmengine.__version__, "mmdet", mmdet.__version__)
print("shapely", shapely.__version__, "terminaltables", getattr(terminaltables, "__version__", "installed"))
PY
"${PYTHON}" "${PROJECT_ROOT}/baselines/smoke_mmdet_configs.py" --upstream "${UPSTREAM}"
