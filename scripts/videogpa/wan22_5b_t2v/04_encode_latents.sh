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
FORCE_ARGS=()
if [[ "${FORCE:-0}" == "1" ]]; then
  FORCE_ARGS+=(--force)
fi

PY_CMD=()
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_CMD=("${PYTHON_BIN}")
else
  CONDA_ENV="${VIDEOGPA_CONDA_ENV:-wan22_videogpa}"
  PY_CMD=(conda run -n "${CONDA_ENV}" python)
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
if (( ${#GPU_LIST[@]} > 1 )); then
  encoded_manifests=()
  pids=()
  for shard_index in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[${shard_index}]}"
    encoded_manifest="${RUN_DIR}/manifests/encoded_pairs.shard_${shard_index}.json"
    shard_log="${RUN_DIR}/logs/encoding.shard_${shard_index}.log"
    encoded_manifests+=("${encoded_manifest}")
    printf '[04_encode_latents] shard %s/%s on GPU %s\n' "${shard_index}" "${#GPU_LIST[@]}" "${gpu}"
    (
      "${PY_CMD[@]}" "${VGM_REPO_ROOT}/VideoGPA/train/Wan2.2-T2V-5B/02_encode.py" \
        --config "${CONFIG}" \
        --run-dir "${RUN_DIR}" \
        --input_json "${RUN_DIR}/manifests/preference_pairs.json" \
        --output_json "${encoded_manifest}" \
        --encoded_root "${RUN_DIR}/encoded" \
        --gpu_id "${gpu}" \
        --shard_index "${shard_index}" \
        --num_shards "${#GPU_LIST[@]}" \
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
    printf '[04_encode_latents] one or more encoding shards failed; see %s/logs/encoding.shard_*.log\n' "${RUN_DIR}" >&2
    exit "${status}"
  fi
  "${PY_CMD[@]}" "${SCRIPT_DIR}/merge_shards.py" encoded \
    --output "${RUN_DIR}/manifests/encoded_pairs.json" \
    --order-json "${RUN_DIR}/manifests/input_subset.json" \
    "${encoded_manifests[@]}"
else
  "${PY_CMD[@]}" "${VGM_REPO_ROOT}/VideoGPA/train/Wan2.2-T2V-5B/02_encode.py" \
    --config "${CONFIG}" \
    --run-dir "${RUN_DIR}" \
    --input_json "${RUN_DIR}/manifests/preference_pairs.json" \
    --output_json "${RUN_DIR}/manifests/encoded_pairs.json" \
    --encoded_root "${RUN_DIR}/encoded" \
    --gpu_id "${GPU_ID}" \
    "${FORCE_ARGS[@]}"
fi
