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

run_a14b() {
  local output_dir="$1"
  local output_manifest="$2"
  local num_prompts_args=("${@:3}")
  IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
  if (( ${#GPU_LIST[@]} > 1 )); then
    extra_args=()
    if [[ "${DIT_FSDP:-1}" == "1" ]]; then
      extra_args+=(--dit_fsdp)
    fi
    if [[ "${T5_FSDP:-0}" == "1" ]]; then
      extra_args+=(--t5_fsdp)
    fi
    if [[ "${USE_SP:-0}" == "1" ]]; then
      extra_args+=(--use_sp)
    fi
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY_CMD[@]}" -m torch.distributed.run \
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
      "${FORCE_ARGS[@]}"
  else
    "${PY_CMD[@]}" "${VGM_REPO_ROOT}/VideoGPA/generate/Wan2.2-A14B.py" \
      --config "${CONFIG}" \
      --run-dir "${RUN_DIR}" \
      --input_json "${RUN_DIR}/manifests/input_subset.json" \
      --output_dir "${output_dir}" \
      --candidate_groups_json "${output_manifest}" \
      --gpu_id "${GPU_ID}" \
      "${num_prompts_args[@]}" \
      "${FORCE_ARGS[@]}"
  fi
}

if [[ "${MODE}" == "micro" ]]; then
  run_a14b "${RUN_DIR}/candidates_micro" "${RUN_DIR}/manifests/candidate_groups_micro.json" --num_prompts 1 --candidates_per_prompt 1
else
  run_a14b "${RUN_DIR}/candidates" "${RUN_DIR}/manifests/candidate_groups.json"
fi
