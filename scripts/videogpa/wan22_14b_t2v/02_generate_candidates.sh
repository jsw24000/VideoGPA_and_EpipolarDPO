#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../../env/require_profile.sh"
vgm_require_profile
CONFIG="${CONFIG:-${VGM_REPO_ROOT}/configs/videogpa/wan22_14b_t2v_formal.yaml}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
MODE="${MODE:-formal}"
GPU_ID="${GPU_ID:-0}"
GPU_IDS="${GPU_IDS:-${GPU_ID}}"
A14B_PARALLEL_MODE="${A14B_PARALLEL_MODE:-distributed}"

FORCE_ARGS=()
if [[ "${FORCE:-0}" == "1" ]]; then
  FORCE_ARGS+=(--force)
fi

RUNTIME_ARGS=()
if [[ -n "${OFFLOAD_MODEL:-}" ]]; then
  case "${OFFLOAD_MODEL}" in
    1|true|TRUE|yes|YES)
      RUNTIME_ARGS+=(--offload_model)
      ;;
    0|false|FALSE|no|NO)
      RUNTIME_ARGS+=(--no-offload_model)
      ;;
    *)
      printf '[wan22_14b_t2v] OFFLOAD_MODEL must be 0/1 or true/false, got %s\n' "${OFFLOAD_MODEL}" >&2
      exit 2
      ;;
  esac
fi
if [[ -n "${CACHE_TEXT_EMBEDDINGS:-}" ]]; then
  case "${CACHE_TEXT_EMBEDDINGS}" in
    1|true|TRUE|yes|YES)
      RUNTIME_ARGS+=(--cache_text_embeddings)
      ;;
    0|false|FALSE|no|NO)
      RUNTIME_ARGS+=(--no-cache_text_embeddings)
      ;;
    *)
      printf '[wan22_14b_t2v] CACHE_TEXT_EMBEDDINGS must be 0/1 or true/false, got %s\n' "${CACHE_TEXT_EMBEDDINGS}" >&2
      exit 2
      ;;
  esac
fi

PY_CMD=()
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_CMD=("${PYTHON_BIN}")
else
  CONDA_ENV="${VIDEOGPA_CONDA_ENV:-wan22_videogpa}"
  PY_CMD=(conda run --no-capture-output -n "${CONDA_ENV}" python)
fi

run_a14b() {
  local output_dir="$1"
  local output_manifest="$2"
  local num_prompts_args=("${@:3}")
  IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
  if (( ${#GPU_LIST[@]} > 1 )); then
    if [[ "${A14B_PARALLEL_MODE}" == "throughput" ]]; then
      shard_manifests=()
      pids=()
      for shard_index in "${!GPU_LIST[@]}"; do
        gpu="${GPU_LIST[${shard_index}]}"
        shard_manifest="${RUN_DIR}/manifests/candidate_groups.shard_${shard_index}.json"
        shard_log="${RUN_DIR}/logs/generation.shard_${shard_index}.log"
        shard_manifests+=("${shard_manifest}")
        printf '[wan22_14b_t2v] throughput shard %s/%s on physical GPU %s -> %s\n' \
          "${shard_index}" "${#GPU_LIST[@]}" "${gpu}" "${shard_manifest}"
        (
          CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY_CMD[@]}" "${VGM_REPO_ROOT}/VideoGPA/generate/Wan2.2-A14B.py" \
            --config "${CONFIG}" \
            --run-dir "${RUN_DIR}" \
            --input_json "${RUN_DIR}/manifests/input_subset.json" \
            --output_dir "${output_dir}" \
            --candidate_groups_json "${shard_manifest}" \
            --gpu_id 0 \
            --shard_index "${shard_index}" \
            --num_shards "${#GPU_LIST[@]}" \
            "${num_prompts_args[@]}" \
            "${RUNTIME_ARGS[@]}" \
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
        printf '[wan22_14b_t2v] one or more throughput shards failed; see %s/logs/generation.shard_*.log\n' "${RUN_DIR}" >&2
        exit "${status}"
      fi
      "${PY_CMD[@]}" "${SCRIPT_DIR}/../wan22_5b_t2v/merge_shards.py" groups \
        --output "${output_manifest}" \
        --order-json "${RUN_DIR}/manifests/input_subset.json" \
        "${shard_manifests[@]}"
      return
    fi
    if [[ "${A14B_PARALLEL_MODE}" != "distributed" ]]; then
      printf '[wan22_14b_t2v] A14B_PARALLEL_MODE must be distributed or throughput, got %s\n' "${A14B_PARALLEL_MODE}" >&2
      exit 2
    fi
    extra_args=()
    if [[ "${DIT_FSDP:-1}" == "1" ]]; then
      extra_args+=(--dit_fsdp)
    fi
    if [[ "${T5_FSDP:-1}" == "1" ]]; then
      extra_args+=(--t5_fsdp)
    fi
    if [[ "${USE_SP:-1}" == "1" ]]; then
      extra_args+=(--ulysses_size "${ULYSSES_SIZE:-${#GPU_LIST[@]}}")
    fi
    printf '[wan22_14b_t2v] distributed A14B generation: GPU_IDS=%s DIT_FSDP=%s T5_FSDP=%s USE_SP=%s ULYSSES_SIZE=%s\n' \
      "${GPU_IDS}" "${DIT_FSDP:-1}" "${T5_FSDP:-1}" "${USE_SP:-1}" "${ULYSSES_SIZE:-${#GPU_LIST[@]}}"
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" PYTHONUNBUFFERED=1 "${PY_CMD[@]}" -m torch.distributed.run \
      --standalone \
      --nnodes=1 \
      --nproc_per_node="${#GPU_LIST[@]}" \
      "${VGM_REPO_ROOT}/VideoGPA/generate/Wan2.2-A14B.py" \
      --config "${CONFIG}" \
      --run-dir "${RUN_DIR}" \
      --input_json "${RUN_DIR}/manifests/input_subset.json" \
      --output_dir "${output_dir}" \
      --candidate_groups_json "${output_manifest}" \
      --gpu_id 0 \
      "${extra_args[@]}" \
      "${num_prompts_args[@]}" \
      "${RUNTIME_ARGS[@]}" \
      "${FORCE_ARGS[@]}"
  else
    PYTHONUNBUFFERED=1 "${PY_CMD[@]}" "${VGM_REPO_ROOT}/VideoGPA/generate/Wan2.2-A14B.py" \
      --config "${CONFIG}" \
      --run-dir "${RUN_DIR}" \
      --input_json "${RUN_DIR}/manifests/input_subset.json" \
      --output_dir "${output_dir}" \
      --candidate_groups_json "${output_manifest}" \
      --gpu_id "${GPU_ID}" \
      "${num_prompts_args[@]}" \
      "${RUNTIME_ARGS[@]}" \
      "${FORCE_ARGS[@]}"
  fi
}

if [[ "${MODE}" == "micro" ]]; then
  IFS=',' read -r -a MODE_GPU_LIST <<< "${GPU_IDS}"
  if [[ "${A14B_PARALLEL_MODE}" == "throughput" && ${#MODE_GPU_LIST[@]} -gt 1 ]]; then
    run_a14b "${RUN_DIR}/candidates_micro" "${RUN_DIR}/manifests/candidate_groups_micro.json" --num_prompts "${#MODE_GPU_LIST[@]}" --candidates_per_prompt 1
  else
    run_a14b "${RUN_DIR}/candidates_micro" "${RUN_DIR}/manifests/candidate_groups_micro.json" --num_prompts 1 --candidates_per_prompt 1
  fi
else
  run_a14b "${RUN_DIR}/candidates" "${RUN_DIR}/manifests/candidate_groups.json"
fi
