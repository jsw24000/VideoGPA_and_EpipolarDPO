#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../../env/require_profile.sh"
vgm_require_profile

CONFIG="${CONFIG:?CONFIG is required}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
GPU_ID="${GPU_ID:-0}"
GPU_IDS="${GPU_IDS:-${GPU_ID}}"
MAX_GROUPS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-groups)
      MAX_GROUPS="$2"
      shift 2
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

PY_CMD=()
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_CMD=("${PYTHON_BIN}")
else
  CONDA_ENV="${VIDEOGPA_CONDA_ENV:-wan22_videogpa}"
  PY_CMD=(conda run -n "${CONDA_ENV}" python)
fi

FORCE_ARGS=()
if [[ "${FORCE:-0}" == "1" ]]; then
  FORCE_ARGS+=(--force)
fi
MAX_GROUP_ARGS=()
if [[ -n "${MAX_GROUPS}" ]]; then
  MAX_GROUP_ARGS+=(--max-groups "${MAX_GROUPS}")
fi

ORDER_JSON="$("${PY_CMD[@]}" -m vgm_common.config --config "${CONFIG}" --run-dir "${RUN_DIR}" --print source_candidate_manifest)"
IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
if (( ${#GPU_LIST[@]} > 1 )); then
  scored_manifests=()
  pids=()
  for shard_index in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[${shard_index}]}"
    scored_manifest="${RUN_DIR}/manifests/scored_candidates.shard_${shard_index}.json"
    shard_log="${RUN_DIR}/logs/epipolar_scoring.shard_${shard_index}.log"
    scored_manifests+=("${scored_manifest}")
    printf '[02_score_epipolar] shard %s/%s on GPU %s\n' "${shard_index}" "${#GPU_LIST[@]}" "${gpu}"
    (
      "${PY_CMD[@]}" "${SCRIPT_DIR}/02_score_epipolar.py" \
        --config "${CONFIG}" \
        --run-dir "${RUN_DIR}" \
        --output-json "${scored_manifest}" \
        --gpu-id "${gpu}" \
        --shard-index "${shard_index}" \
        --num-shards "${#GPU_LIST[@]}" \
        "${MAX_GROUP_ARGS[@]}" \
        "${FORCE_ARGS[@]}"
    ) >"${shard_log}" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [[ "${status}" != "0" ]]; then
    printf '[02_score_epipolar] one or more scoring shards failed; see %s/logs/epipolar_scoring.shard_*.log\n' "${RUN_DIR}" >&2
    exit "${status}"
  fi
  "${PY_CMD[@]}" "${SCRIPT_DIR}/merge_shards.py" scored \
    --output "${RUN_DIR}/manifests/scored_candidates.json" \
    --order-json "${ORDER_JSON}" \
    "${scored_manifests[@]}"
else
  "${PY_CMD[@]}" "${SCRIPT_DIR}/02_score_epipolar.py" \
    --config "${CONFIG}" \
    --run-dir "${RUN_DIR}" \
    --output-json "${RUN_DIR}/manifests/scored_candidates.json" \
    --gpu-id "${GPU_ID}" \
    "${MAX_GROUP_ARGS[@]}" \
    "${FORCE_ARGS[@]}"
fi
