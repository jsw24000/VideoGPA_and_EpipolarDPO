#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[run_wan22_i2v_smoke] %s\n' "$*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${ENV_NAME:-wan22_videogpa}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/wan22_smoke"
WAN_SRC_DIR="${WAN_SRC_DIR:-${PROJECT_ROOT}/third_party/Wan2.2}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models/wan/Wan2.2-TI2V-5B}"

PROMPT="${PROMPT:-The camera slowly moves forward through the scene while the geometry and object shapes remain stable.}"
SIZE="${SIZE:-1280*704}"
FRAME_NUM="${FRAME_NUM:-17}"
SAMPLE_STEPS="${SAMPLE_STEPS:-6}"
SAMPLE_SHIFT="${SAMPLE_SHIFT:-5.0}"
SAMPLE_GUIDE_SCALE="${SAMPLE_GUIDE_SCALE:-5.0}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-unipc}"
SEED="${SEED:-42}"
OFFLOAD_MODEL="${OFFLOAD_MODEL:-False}"
CONVERT_MODEL_DTYPE="${CONVERT_MODEL_DTYPE:-1}"
T5_CPU="${T5_CPU:-0}"

mkdir -p "${OUTPUT_ROOT}/input" "${OUTPUT_ROOT}/generated"

command -v conda >/dev/null 2>&1 || {
  printf 'conda was not found on PATH.\n' >&2
  exit 1
}

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  printf 'Conda env %s does not exist. Run scripts/setup_wan22_env.sh first.\n' "${ENV_NAME}" >&2
  exit 1
fi

if [ ! -f "${WAN_SRC_DIR}/generate.py" ]; then
  printf 'Wan2.2 generate.py not found at %s\n' "${WAN_SRC_DIR}" >&2
  exit 1
fi

if [ -n "${IMAGE_PATH:-}" ]; then
  INPUT_IMAGE="${IMAGE_PATH}"
else
  INPUT_IMAGE=""
  for candidate in \
    "${MODEL_DIR}/examples/i2v_input.JPG" \
    "${MODEL_DIR}/assets/i2v_input.JPG" \
    "${WAN_SRC_DIR}/examples/i2v_input.JPG" \
    "${WAN_SRC_DIR}/assets/i2v_input.JPG"; do
    if [ -f "${candidate}" ]; then
      INPUT_IMAGE="${candidate}"
      break
    fi
  done
fi

if [ -z "${INPUT_IMAGE}" ] || [ ! -f "${INPUT_IMAGE}" ]; then
  printf 'No valid I2V input image found. Set IMAGE_PATH=/path/to/image.\n' >&2
  exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/runs/${RUN_ID}"
INPUT_DIR="${OUTPUT_ROOT}/input/${RUN_ID}"
GENERATED_DIR="${OUTPUT_ROOT}/generated/${RUN_ID}"
mkdir -p "${RUN_DIR}" "${INPUT_DIR}" "${GENERATED_DIR}"

INPUT_COPY="${INPUT_DIR}/$(basename "${INPUT_IMAGE}")"
cp "${INPUT_IMAGE}" "${INPUT_COPY}"

SAVE_FILE="${GENERATED_DIR}/wan22_ti2v_i2v_smoke.mp4"
RUN_LOG="${RUN_DIR}/run.log"
COMMAND_FILE="${RUN_DIR}/command_used.txt"
MEM_LOG="${RUN_DIR}/gpu_memory.csv"
MEM_SUMMARY="${RUN_DIR}/gpu_memory_summary.txt"

CMD=(
  conda run -n "${ENV_NAME}" python "${WAN_SRC_DIR}/generate.py"
  --task ti2v-5B
  --size "${SIZE}"
  --ckpt_dir "${MODEL_DIR}"
  --image "${INPUT_COPY}"
  --prompt "${PROMPT}"
  --save_file "${SAVE_FILE}"
  --frame_num "${FRAME_NUM}"
  --sample_steps "${SAMPLE_STEPS}"
  --sample_shift "${SAMPLE_SHIFT}"
  --sample_guide_scale "${SAMPLE_GUIDE_SCALE}"
  --sample_solver "${SAMPLE_SOLVER}"
  --base_seed "${SEED}"
  --offload_model "${OFFLOAD_MODEL}"
)

if [ "${CONVERT_MODEL_DTYPE}" = "1" ]; then
  CMD+=(--convert_model_dtype)
fi

if [ "${T5_CPU}" = "1" ]; then
  CMD+=(--t5_cpu)
fi

{
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU_ID}"
  printf '%q ' "${CMD[@]}"
  printf '\n'
} > "${COMMAND_FILE}"
cp "${COMMAND_FILE}" "${OUTPUT_ROOT}/command_used.txt"

printf 'timestamp,index,memory.used_mib,memory.total_mib,name\n' > "${MEM_LOG}"
(
  while true; do
    printf '%s,' "$(date -Is)" >> "${MEM_LOG}"
    nvidia-smi --id="${GPU_ID}" --query-gpu=index,memory.used,memory.total,name --format=csv,noheader,nounits >> "${MEM_LOG}" 2>/dev/null || true
    sleep 5
  done
) &
SAMPLER_PID=$!

cleanup() {
  if [ -n "${SAMPLER_PID:-}" ]; then
    kill "${SAMPLER_PID}" >/dev/null 2>&1 || true
    wait "${SAMPLER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

log "Running official Wan2.2 I2V smoke test on GPU ${GPU_ID}"
set +e
CUDA_VISIBLE_DEVICES="${GPU_ID}" stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee "${RUN_LOG}"
STATUS=${PIPESTATUS[0]}
set -e
cleanup
trap - EXIT

cp "${RUN_LOG}" "${OUTPUT_ROOT}/run.log"

awk -F, '
  NR > 1 {
    value=$3
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
    if (value + 0 > max) max = value + 0
  }
  END {
    printf "max_gpu_memory_mib=%d\nmax_gpu_memory_gib=%.2f\n", max, max / 1024
  }
' "${MEM_LOG}" > "${MEM_SUMMARY}"
cp "${MEM_SUMMARY}" "${OUTPUT_ROOT}/gpu_memory_summary.txt"

if [ "${STATUS}" -ne 0 ]; then
  printf 'Wan2.2 smoke test failed with exit code %s. Log: %s\n' "${STATUS}" "${RUN_LOG}" >&2
  exit "${STATUS}"
fi

if [ ! -s "${SAVE_FILE}" ]; then
  printf 'Wan2.2 command finished, but no non-empty video was written at %s\n' "${SAVE_FILE}" >&2
  exit 1
fi

log "Video saved to ${SAVE_FILE}"
log "Run log saved to ${RUN_LOG}"
log "$(tr '\n' ' ' < "${MEM_SUMMARY}")"
