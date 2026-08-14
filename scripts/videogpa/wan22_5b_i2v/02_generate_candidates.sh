#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T2V_SCRIPT_DIR="${SCRIPT_DIR}/../wan22_5b_t2v"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../../env/require_profile.sh"
vgm_require_profile
CONFIG="${CONFIG:-${VGM_REPO_ROOT}/configs/videogpa/wan22_5b_i2v_formal.yaml}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
MODE="${MODE:-formal}"
GPU_ID="${GPU_ID:-0}"
GPU_IDS="${GPU_IDS:-${GPU_ID}}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  printf '[02_generate_candidates_i2v] CUDA_VISIBLE_DEVICES=%s is set.\n' "${CUDA_VISIBLE_DEVICES}" >&2
  printf '[02_generate_candidates_i2v] This script follows the T2V formal generator and passes physical --gpu_id values.\n' >&2
  printf '[02_generate_candidates_i2v] Start from a clean shell or run: unset CUDA_VISIBLE_DEVICES\n' >&2
  exit 2
fi

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

if [[ "${MODE}" == "micro" ]]; then
  "${PY_CMD[@]}" "${VGM_REPO_ROOT}/VideoGPA/generate/Wan2.2-I2V-5B.py" \
    --config "${CONFIG}" \
    --run-dir "${RUN_DIR}" \
    --input_json "${RUN_DIR}/manifests/input_subset.json" \
    --output_dir "${RUN_DIR}/candidates_micro" \
    --candidate_groups_json "${RUN_DIR}/manifests/candidate_groups_micro.json" \
    --gpu_id "${GPU_ID}" \
    --num_prompts 1 \
    --candidates_per_prompt 1 \
    "${FORCE_ARGS[@]}"
else
  IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
  if (( ${#GPU_LIST[@]} > 1 )); then
    shard_manifests=()
    pids=()
    for shard_index in "${!GPU_LIST[@]}"; do
      gpu="${GPU_LIST[${shard_index}]}"
      shard_manifest="${RUN_DIR}/manifests/candidate_groups.shard_${shard_index}.json"
      shard_log="${RUN_DIR}/logs/generation.shard_${shard_index}.log"
      shard_manifests+=("${shard_manifest}")
      printf '[02_generate_candidates_i2v] shard %s/%s on GPU %s -> %s\n' \
        "${shard_index}" "${#GPU_LIST[@]}" "${gpu}" "${shard_manifest}"
      (
        "${PY_CMD[@]}" "${VGM_REPO_ROOT}/VideoGPA/generate/Wan2.2-I2V-5B.py" \
          --config "${CONFIG}" \
          --run-dir "${RUN_DIR}" \
          --input_json "${RUN_DIR}/manifests/input_subset.json" \
          --output_dir "${RUN_DIR}/candidates" \
          --candidate_groups_json "${shard_manifest}" \
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
      printf '[02_generate_candidates_i2v] one or more generation shards failed; see %s/logs/generation.shard_*.log\n' "${RUN_DIR}" >&2
      exit "${status}"
    fi
    "${PY_CMD[@]}" "${T2V_SCRIPT_DIR}/merge_shards.py" groups \
      --output "${RUN_DIR}/manifests/candidate_groups.json" \
      --order-json "${RUN_DIR}/manifests/input_subset.json" \
      "${shard_manifests[@]}"
  else
    "${PY_CMD[@]}" "${VGM_REPO_ROOT}/VideoGPA/generate/Wan2.2-I2V-5B.py" \
      --config "${CONFIG}" \
      --run-dir "${RUN_DIR}" \
      --input_json "${RUN_DIR}/manifests/input_subset.json" \
      --output_dir "${RUN_DIR}/candidates" \
      --candidate_groups_json "${RUN_DIR}/manifests/candidate_groups.json" \
      --gpu_id "${GPU_ID}" \
      "${FORCE_ARGS[@]}"
  fi
fi
