#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/videogpa/wan22_5b_t2v_smoke.yaml}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"

PY_CMD=()
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_CMD=("${PYTHON_BIN}")
else
  CONDA_ENV="${VIDEOGPA_CONDA_ENV:-wan22_videogpa}"
  PY_CMD=(conda run -n "${CONDA_ENV}" python)
fi

FALLBACK_ARGS=()
if [[ "${DISABLE_DEBUG_FALLBACK:-0}" == "1" ]]; then
  FALLBACK_ARGS+=(--disable-debug-fallback)
fi

"${PY_CMD[@]}" "${SCRIPT_DIR}/score_preferences.py" \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --input-json "${RUN_DIR}/manifests/candidate_groups.json" \
  --output-json "${RUN_DIR}/manifests/scored_candidates.json" \
  --pairs-json "${RUN_DIR}/manifests/preference_pairs.json" \
  "${FALLBACK_ARGS[@]}"
