#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../../env/require_profile.sh"
vgm_require_profile
CONFIG="${CONFIG:-${VGM_REPO_ROOT}/configs/videogpa/wan22_5b_t2v_smoke.yaml}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
GPU_ID="${GPU_ID:-0}"
LORA_PATH="${LORA_PATH:-${RUN_DIR}/checkpoints/step_000005}"

PY_CMD=()
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_CMD=("${PYTHON_BIN}")
else
  CONDA_ENV="${VIDEOGPA_CONDA_ENV:-wan22_videogpa}"
  PY_CMD=(conda run -n "${CONDA_ENV}" python)
fi

"${PY_CMD[@]}" "${VGM_REPO_ROOT}/VideoGPA/generate/Wan2.2-T2V-5B.py" \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --input_json "${RUN_DIR}/manifests/input_subset.json" \
  --output_dir "${RUN_DIR}/comparisons/base" \
  --candidate_groups_json "${RUN_DIR}/comparisons/base_manifest.json" \
  --gpu_id "${GPU_ID}" \
  --num_prompts 1 \
  --candidate_seeds 2001 \
  --candidates_per_prompt 1

"${PY_CMD[@]}" "${VGM_REPO_ROOT}/VideoGPA/generate/Wan2.2-T2V-5B.py" \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --input_json "${RUN_DIR}/manifests/input_subset.json" \
  --output_dir "${RUN_DIR}/comparisons/lora" \
  --candidate_groups_json "${RUN_DIR}/comparisons/lora_manifest.json" \
  --gpu_id "${GPU_ID}" \
  --num_prompts 1 \
  --candidate_seeds 2001 \
  --candidates_per_prompt 1 \
  --lora_path "${LORA_PATH}"

"${PY_CMD[@]}" "${SCRIPT_DIR}/write_smoke_summary.py" \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --write-comparison-only
