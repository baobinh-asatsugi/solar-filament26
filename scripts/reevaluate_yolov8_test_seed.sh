#!/usr/bin/env bash
set -euo pipefail

SEED="${1:?usage: bash scripts/reevaluate_yolov8_test_seed.sh SEED [GPU]}"
GPU="${2:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/magfilo}"
PYTHON="${PYTHON:-python}"
export PROJECT_ROOT ARTIFACT_ROOT DATA_ROOT PYTHON
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"

RUN="magfilo_v1_yolov8l_s${SEED}"
BEST="${ARTIFACT_ROOT}/runs/${RUN}/weights/best.pt"
OUT="${ARTIFACT_ROOT}/results/${RUN}"
FROZEN="${OUT}/val_best.json"
LOG="${ARTIFACT_ROOT}/logs/yolov8_metrics_s${SEED}.log"

[[ -s "${BEST}" ]] || { echo "Missing checkpoint: ${BEST}" >&2; exit 1; }
[[ -s "${FROZEN}" ]] || { echo "Missing frozen validation config: ${FROZEN}" >&2; exit 1; }
mkdir -p "${OUT}" "$(dirname "${LOG}")"
if [[ -s "${OUT}/test_best.json" && ! -e "${OUT}/test_best.pq_only.json" ]]; then
  cp "${OUT}/test_best.json" "${OUT}/test_best.pq_only.json"
fi

bash "${PROJECT_ROOT}/scripts/run_with_gpu_limit.sh" "${GPU}" "${LOG}" \
  "${PYTHON}" "${PROJECT_ROOT}/baselines/eval_yolo_pq.py" \
  --checkpoint "${BEST}" --manifest "${PROJECT_ROOT}/splits/magfilo_v1/manifest.csv" \
  --split test --data-root "${DATA_ROOT}" --artifact-root "${ARTIFACT_ROOT}" \
  --device "${GPU}" --output-dir "${OUT}" --frozen-config "${FROZEN}"

"${PYTHON}" - "${OUT}/test_best.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
required = {"semantic_dice", "semantic_iou", "mean_semantic_dice", "mean_semantic_iou"}
missing = sorted(required - payload.keys())
if missing:
    raise RuntimeError(f"Semantic metrics missing from {path}: {missing}")
print(json.dumps({key: payload[key] for key in ("global_pq", *sorted(required))}, indent=2))
PY
