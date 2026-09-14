#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../../env/require_profile.sh"
vgm_require_profile

CONFIG="${CONFIG:-${VGM_REPO_ROOT}/configs/videogpa/wan22_14b_t2v_training.yaml}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:?SOURCE_RUN_DIR must contain the completed encoded A14B pairs}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be a new expert-specific training directory}"
EXPERT_MODE="${EXPERT_MODE:?EXPERT_MODE must be high or low}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MAX_STEPS="${MAX_STEPS:-}"
SAVE_STEPS="${SAVE_STEPS:-}"
TRAINING_SHIFT="${TRAINING_SHIFT:-5.0}"
PAIR_SCORE_MODE="${PAIR_SCORE_MODE:-separate}"
RESUME="${RESUME:-0}"
MEMORY_PROBE="${MEMORY_PROBE:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF

case "${EXPERT_MODE}" in high|low) ;; *) printf 'EXPERT_MODE must be high or low\n' >&2; exit 2 ;; esac
case "${PAIR_SCORE_MODE}" in separate|stacked) ;; *) printf 'PAIR_SCORE_MODE must be separate or stacked\n' >&2; exit 2 ;; esac
[[ "$(realpath -m "${SOURCE_RUN_DIR}")" != "$(realpath -m "${OUTPUT_DIR}")" ]] || {
  printf 'OUTPUT_DIR must be separate from SOURCE_RUN_DIR\n' >&2
  exit 2
}
test -s "${SOURCE_RUN_DIR}/manifests/encoded_pairs.json"

if [[ -d "${OUTPUT_DIR}" ]] && find "${OUTPUT_DIR}" -mindepth 1 -print -quit | grep -q .; then
  [[ "${RESUME}" == "1" ]] || {
    printf 'Refusing non-empty OUTPUT_DIR without RESUME=1: %s\n' "${OUTPUT_DIR}" >&2
    exit 2
  }
fi
mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/reports"

PY_CMD=()
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_CMD=("${PYTHON_BIN}")
else
  PY_CMD=(conda run --no-capture-output -n "${VIDEOGPA_CONDA_ENV:-wan22_videogpa}" python)
fi

ARGS=(
  "${VGM_REPO_ROOT}/VideoGPA/train/Wan2.2-T2V-5B/03_train.py"
  --config "${CONFIG}"
  --run-dir "${SOURCE_RUN_DIR}"
  --metadata_path "${SOURCE_RUN_DIR}/manifests/encoded_pairs.json"
  --output_dir "${OUTPUT_DIR}"
  --expert-mode "${EXPERT_MODE}"
  --reference-mode shared_base
  --timestep-mode shifted_scheduler
  --training-shift "${TRAINING_SHIFT}"
  --distributed-strategy fsdp_full_shard
  --backward-mode sequential_recompute
  --pair-score-mode "${PAIR_SCORE_MODE}"
)
[[ -n "${MAX_STEPS}" ]] && ARGS+=(--max_train_steps "${MAX_STEPS}")
[[ -n "${SAVE_STEPS}" ]] && ARGS+=(--save-steps "${SAVE_STEPS}")
[[ "${RESUME}" == "1" ]] && ARGS+=(--resume)
[[ "${MEMORY_PROBE}" == "1" ]] && ARGS+=(--memory-probe)
[[ -n "${STOP_AFTER_STEP:-}" ]] && ARGS+=(--stop-after-step "${STOP_AFTER_STEP}")
[[ -n "${WARMUP_STEPS:-}" ]] && ARGS+=(--warmup-steps "${WARMUP_STEPS}")

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
(( ${#GPU_LIST[@]} >= 2 )) || { printf 'FSDP requires at least two GPUs\n' >&2; exit 2; }

STAMP="$(date +%Y%m%d_%H%M%S)"
{
  printf 'SOURCE_RUN_DIR=%s\n' "${SOURCE_RUN_DIR}"
  printf 'OUTPUT_DIR=%s\n' "${OUTPUT_DIR}"
  printf 'EXPERT_MODE=%s\n' "${EXPERT_MODE}"
  printf 'GPU_IDS=%s\n' "${GPU_IDS}"
  printf 'MAX_STEPS=%s\n' "${MAX_STEPS}"
  printf 'SAVE_STEPS=%s\n' "${SAVE_STEPS}"
  printf 'TRAINING_SHIFT=%s\n' "${TRAINING_SHIFT}"
  printf 'PAIR_SCORE_MODE=%s\n' "${PAIR_SCORE_MODE}"
  printf 'RESUME=%s\n' "${RESUME}"
  printf 'STOP_AFTER_STEP=%s\n' "${STOP_AFTER_STEP:-}"
  printf 'PYTORCH_CUDA_ALLOC_CONF=%s\n' "${PYTORCH_CUDA_ALLOC_CONF}"
} > "${OUTPUT_DIR}/launch.${STAMP}.txt"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY_CMD[@]}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node="${#GPU_LIST[@]}" \
  "${ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/logs/training.${STAMP}.log"

printf 'Expert training invocation PASS: %s\n' "${OUTPUT_DIR}"
