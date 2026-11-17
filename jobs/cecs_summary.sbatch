#!/bin/bash
#SBATCH --job-name=ml26-summary
#SBATCH --qos=student
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/ml26}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${HOME}/ml26_artifacts}"
VENV_PATH="${VENV_PATH:-${HOME}/.venv}"
export PROJECT_ROOT ARTIFACT_ROOT VENV_PATH PYTHON="${VENV_PATH}/bin/python"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"
source "${VENV_PATH}/bin/activate"
bash scripts/summarize_five_seed_results.sh
