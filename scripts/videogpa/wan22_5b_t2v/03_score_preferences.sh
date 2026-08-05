#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../../env/require_profile.sh"
vgm_require_profile
CONFIG="${CONFIG:-${VGM_REPO_ROOT}/configs/videogpa/wan22_5b_t2v_smoke.yaml}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
GPU_ID="${GPU_ID:-0}"
GPU_IDS="${GPU_IDS:-${GPU_ID}}"

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

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
if (( ${#GPU_LIST[@]} > 1 )); then
  scored_manifests=()
  pair_manifests=()
  pids=()
  for shard_index in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[${shard_index}]}"
    scored_manifest="${RUN_DIR}/manifests/scored_candidates.shard_${shard_index}.json"
    pair_manifest="${RUN_DIR}/manifests/preference_pairs.shard_${shard_index}.json"
    shard_log="${RUN_DIR}/logs/scoring.shard_${shard_index}.log"
    scored_manifests+=("${scored_manifest}")
    pair_manifests+=("${pair_manifest}")
    printf '[03_score_preferences] shard %s/%s on GPU %s\n' "${shard_index}" "${#GPU_LIST[@]}" "${gpu}"
    (
      "${PY_CMD[@]}" "${SCRIPT_DIR}/score_preferences.py" \
        --config "${CONFIG}" \
        --run-dir "${RUN_DIR}" \
        --input-json "${RUN_DIR}/manifests/candidate_groups.json" \
        --output-json "${scored_manifest}" \
        --pairs-json "${pair_manifest}" \
        --gpu-id "${gpu}" \
        --shard-index "${shard_index}" \
        --num-shards "${#GPU_LIST[@]}" \
        --allow-insufficient-pairs \
        "${FALLBACK_ARGS[@]}"
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
    printf '[03_score_preferences] one or more scoring shards failed; see %s/logs/scoring.shard_*.log\n' "${RUN_DIR}" >&2
    exit "${status}"
  fi
  "${PY_CMD[@]}" "${SCRIPT_DIR}/merge_shards.py" scored-pairs \
    --scored-output "${RUN_DIR}/manifests/scored_candidates.json" \
    --pairs-output "${RUN_DIR}/manifests/preference_pairs.json" \
    --order-json "${RUN_DIR}/manifests/input_subset.json" \
    --scored-inputs "${scored_manifests[@]}" \
    --pair-inputs "${pair_manifests[@]}"
else
  "${PY_CMD[@]}" "${SCRIPT_DIR}/score_preferences.py" \
    --config "${CONFIG}" \
    --run-dir "${RUN_DIR}" \
    --input-json "${RUN_DIR}/manifests/candidate_groups.json" \
    --output-json "${RUN_DIR}/manifests/scored_candidates.json" \
    --pairs-json "${RUN_DIR}/manifests/preference_pairs.json" \
    --gpu-id "${GPU_ID}" \
    "${FALLBACK_ARGS[@]}"
fi
