#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: run_with_gpu_limit.sh GPU LOG_FILE COMMAND [ARGS...]}"
LOG_FILE="${2:?log file}"
shift 2
(( $# > 0 )) || { echo "ERROR: missing command" >&2; exit 2; }

MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-2}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-15}"
GPU_START_TIMEOUT="${GPU_START_TIMEOUT:-120}"
GPU_IGNORED_PROCESS_REGEX="${GPU_IGNORED_PROCESS_REGEX:-soffice\\.bin$}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$(pwd)/artifacts}"
SCHEDULER_ROOT="${ARTIFACT_ROOT}/gpu_scheduler"
LOCK_FILE="${SCHEDULER_ROOT}/gpu${GPU}.lock"

mkdir -p "${SCHEDULER_ROOT}" "$(dirname "${LOG_FILE}")"
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not found" >&2; exit 127; }
command -v flock >/dev/null || { echo "ERROR: flock not found" >&2; exit 127; }

active_jobs() {
  local processes
  if ! processes="$(
    nvidia-smi -i "${GPU}" \
      --query-compute-apps=pid,process_name \
      --format=csv,noheader,nounits 2>/dev/null
  )"; then
    # Fail closed: a transient nvidia-smi error must not oversubscribe the GPU.
    echo "${MAX_JOBS_PER_GPU}"
    return
  fi
  awk -F',' -v ignored="${GPU_IGNORED_PROCESS_REGEX}" '
    {
      pid = $1
      name = tolower($2)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", pid)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
      if (pid ~ /^[0-9]+$/ && name !~ ignored) active++
    }
    END { print active + 0 }
  ' <<<"${processes}"
}

run_logged() {
  set -o pipefail
  "$@" 2>&1 | tee "${LOG_FILE}"
}

while true; do
  exec 9>>"${LOCK_FILE}"
  flock 9
  BEFORE="$(active_jobs)"
  if (( BEFORE < MAX_JOBS_PER_GPU )); then
    echo "[GPU gate] gpu=${GPU} active=${BEFORE}/${MAX_JOBS_PER_GPU}; starting: $*" | tee -a "${LOG_FILE}"
    run_logged "$@" &
    CHILD_PID=$!

    # Keep the allocation lock until nvidia-smi observes the new CUDA process.
    # This closes the race where two launchers simultaneously see one free slot.
    REGISTERED=0
    for ((SECOND = 0; SECOND < GPU_START_TIMEOUT; SECOND++)); do
      if ! kill -0 "${CHILD_PID}" 2>/dev/null; then
        break
      fi
      CURRENT="$(active_jobs)"
      if (( CURRENT > BEFORE )); then
        REGISTERED=1
        break
      fi
      sleep 1
    done
    flock -u 9
    exec 9>&-

    if (( REGISTERED == 0 )); then
      echo "[GPU gate] gpu=${GPU}: CUDA registration not observed within ${GPU_START_TIMEOUT}s; continuing to monitor job" \
        | tee -a "${LOG_FILE}"
    fi
    wait "${CHILD_PID}"
    exit $?
  fi

  echo "[GPU gate] gpu=${GPU} active=${BEFORE}/${MAX_JOBS_PER_GPU}; waiting ${GPU_POLL_SECONDS}s: $*" \
    | tee -a "${LOG_FILE}"
  flock -u 9
  exec 9>&-
  sleep "${GPU_POLL_SECONDS}"
done
