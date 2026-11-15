#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
PYTHON="${PYTHON:-python}"; export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"
"${PYTHON}" "${PROJECT_ROOT}/baselines/summarize_all.py" --artifact-root "${ARTIFACT_ROOT}"
