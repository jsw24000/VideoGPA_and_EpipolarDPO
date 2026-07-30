#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[run_wan22_compare_generate] %s\n' "$*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env/require_profile.sh"
vgm_require_profile

ENV_NAME="${ENV_NAME:-wan22_videogpa}"
PHYSICAL_GPU_ID="${GPU_ID:-0}"
VIDEOGPA_DIR="${VIDEOGPA_DIR:-${VGM_REPO_ROOT}/VideoGPA}"
MODEL_PATH="${MODEL_PATH:-${VGM_MODEL_ROOT}/wan/Wan2.2-TI2V-5B}"
PROMPT_JSON="${PROMPT_JSON:-${VGM_OUTPUT_ROOT}/dl3dv_miniset/prompts_30_camera_motion.json}"
LORA_PATH="${LORA_PATH:-${VGM_MODEL_ROOT}/videogpa/VideoGPA-Wan2.2TI2V-lora}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${VGM_OUTPUT_ROOT}/evaluation/wan22_compare}"

SEED="${SEED:-42}"
FRAME_NUM="${FRAME_NUM:-81}"
SAMPLING_STEPS="${SAMPLING_STEPS:-20}"
SHIFT="${SHIFT:-5.0}"
GUIDE_SCALE="${GUIDE_SCALE:-5.0}"
FPS="${FPS:-24}"
LORA_WEIGHT="${LORA_WEIGHT:-0.2}"
VARIANTS="${VARIANTS:-both}" # both | baseline | lora
NUM_PROMPTS="${NUM_PROMPTS:-}"

LORA_TAG="$(printf '%s' "${LORA_WEIGHT}" | tr '.' 'p')"
RUN_NAME="${RUN_NAME:-dl3dv30_f${FRAME_NUM}_steps${SAMPLING_STEPS}_seed${SEED}}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"
GEN_ROOT="${RUN_ROOT}/generation"
BASELINE_DIR="${GEN_ROOT}/baseline"
LORA_DIR="${GEN_ROOT}/lora_w${LORA_TAG}"
LOG_DIR="${RUN_ROOT}/logs"
CONFIG_DIR="${RUN_ROOT}/config"

if [ ! -f "${PROMPT_JSON}" ]; then
  printf 'Prompt JSON not found: %s\n' "${PROMPT_JSON}" >&2
  exit 1
fi

if [ ! -f "${VIDEOGPA_DIR}/generate/Wan2.2-TI2V-5B.py" ]; then
  printf 'VideoGPA Wan generate script not found under: %s\n' "${VIDEOGPA_DIR}" >&2
  exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
  printf 'WAN model path not found: %s\n' "${MODEL_PATH}" >&2
  exit 1
fi

mkdir -p "${BASELINE_DIR}" "${LORA_DIR}" "${LOG_DIR}" "${CONFIG_DIR}"
cp "${PROMPT_JSON}" "${CONFIG_DIR}/prompt_json_used.json"

COMMON_ARGS=(
  --model_path "${MODEL_PATH}"
  --prompt_json "${PROMPT_JSON}"
  --gpu_id 0
  --seed "${SEED}"
  --frame_num "${FRAME_NUM}"
  --sampling_steps "${SAMPLING_STEPS}"
  --shift "${SHIFT}"
  --guide_scale "${GUIDE_SCALE}"
  --fps "${FPS}"
)

if [ -n "${NUM_PROMPTS}" ]; then
  COMMON_ARGS+=(--num_prompts "${NUM_PROMPTS}")
fi

write_manifest() {
  {
    printf 'run_name=%s\n' "${RUN_NAME}"
    printf 'prompt_json=%s\n' "${PROMPT_JSON}"
    printf 'model_path=%s\n' "${MODEL_PATH}"
    printf 'lora_path=%s\n' "${LORA_PATH}"
    printf 'baseline_dir=%s\n' "${BASELINE_DIR}"
    printf 'lora_dir=%s\n' "${LORA_DIR}"
    printf 'seed=%s\n' "${SEED}"
    printf 'frame_num=%s\n' "${FRAME_NUM}"
    printf 'sampling_steps=%s\n' "${SAMPLING_STEPS}"
    printf 'shift=%s\n' "${SHIFT}"
    printf 'guide_scale=%s\n' "${GUIDE_SCALE}"
    printf 'fps=%s\n' "${FPS}"
    printf 'lora_weight=%s\n' "${LORA_WEIGHT}"
    printf 'physical_gpu_id=%s\n' "${PHYSICAL_GPU_ID}"
    printf 'variants=%s\n' "${VARIANTS}"
    printf 'num_prompts=%s\n' "${NUM_PROMPTS:-all}"
  } > "${CONFIG_DIR}/generation_manifest.txt"
}

run_variant() {
  local label="$1"
  local out_dir="$2"
  shift 2

  local log_file="${LOG_DIR}/${label}.log"
  local cmd_file="${CONFIG_DIR}/${label}_command.txt"
  local cmd=(
    conda run -n "${ENV_NAME}" python "${VIDEOGPA_DIR}/generate/Wan2.2-TI2V-5B.py"
    "${COMMON_ARGS[@]}"
    --output_dir "${out_dir}"
    "$@"
  )

  {
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${PHYSICAL_GPU_ID}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } > "${cmd_file}"

  log "Starting ${label}; output: ${out_dir}"
  CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU_ID}" stdbuf -oL -eL "${cmd[@]}" 2>&1 | tee "${log_file}"
  log "Finished ${label}; log: ${log_file}"
}

write_manifest

case "${VARIANTS}" in
  both)
    run_variant baseline "${BASELINE_DIR}"
    run_variant "lora_w${LORA_TAG}" "${LORA_DIR}" --lora_path "${LORA_PATH}" --lora_weight "${LORA_WEIGHT}"
    ;;
  baseline)
    run_variant baseline "${BASELINE_DIR}"
    ;;
  lora)
    run_variant "lora_w${LORA_TAG}" "${LORA_DIR}" --lora_path "${LORA_PATH}" --lora_weight "${LORA_WEIGHT}"
    ;;
  *)
    printf 'Unsupported VARIANTS=%s. Use both, baseline, or lora.\n' "${VARIANTS}" >&2
    exit 1
    ;;
esac

log "Generation root: ${GEN_ROOT}"
log "Run manifest: ${CONFIG_DIR}/generation_manifest.txt"
