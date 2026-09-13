#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../../env/require_profile.sh"
vgm_require_profile

CONFIG="${CONFIG:-${VGM_REPO_ROOT}/configs/videogpa/wan22_14b_t2v_formal.yaml}"
RUN_DIR="${RUN_DIR:?RUN_DIR must point to the completed A14B VideoGPA source run}"
GPU_IDS="${GPU_IDS:-0}"
EXPERT_MODE="${EXPERT_MODE:-high}"
REFERENCE_MODE="${REFERENCE_MODE:-shared_base}"
GATE_STEPS="${GATE_STEPS:-1}"
TRAINING_SHIFT="${TRAINING_SHIFT:-5.0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF
PROBE_ID="${PROBE_ID:-$(date +%Y%m%d_%H%M%S)_${EXPERT_MODE}_${REFERENCE_MODE}_${GATE_STEPS}step}"
PROBE_DIR="${PROBE_DIR:-${VGM_OUTPUT_ROOT}/videogpa/wan22_14b_t2v/probes/${PROBE_ID}}"

case "${EXPERT_MODE}" in
  high|low|both) ;;
  *) printf 'EXPERT_MODE must be high, low, or both; got %s\n' "${EXPERT_MODE}" >&2; exit 2 ;;
esac
case "${REFERENCE_MODE}" in
  shared_base|separate) ;;
  *) printf 'REFERENCE_MODE must be shared_base or separate; got %s\n' "${REFERENCE_MODE}" >&2; exit 2 ;;
esac
if ! [[ "${GATE_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'GATE_STEPS must be a positive integer; got %s\n' "${GATE_STEPS}" >&2
  exit 2
fi
if [[ "$(realpath -m "${PROBE_DIR}")" == "$(realpath -m "${RUN_DIR}")" ]]; then
  printf 'PROBE_DIR must be separate from RUN_DIR\n' >&2
  exit 2
fi
if [[ -d "${PROBE_DIR}" ]] && find "${PROBE_DIR}" -mindepth 1 -print -quit | grep -q .; then
  printf 'Refusing non-empty PROBE_DIR: %s\n' "${PROBE_DIR}" >&2
  exit 2
fi
test -f "${RUN_DIR}/manifests/encoded_pairs.json"
mkdir -p "${PROBE_DIR}/logs" "${PROBE_DIR}/reports"

PY_CMD=()
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_CMD=("${PYTHON_BIN}")
else
  PY_CMD=(conda run --no-capture-output -n "${VIDEOGPA_CONDA_ENV:-wan22_videogpa}" python)
fi

TRAINER="${VGM_REPO_ROOT}/VideoGPA/train/Wan2.2-T2V-5B/03_train.py"
TRAIN_ARGS=(
  "${TRAINER}"
  --config "${CONFIG}"
  --run-dir "${RUN_DIR}"
  --metadata_path "${RUN_DIR}/manifests/encoded_pairs.json"
  --output_dir "${PROBE_DIR}"
  --expert-mode "${EXPERT_MODE}"
  --reference-mode "${REFERENCE_MODE}"
  --timestep-mode shifted_scheduler
  --training-shift "${TRAINING_SHIFT}"
  --max_train_steps "${GATE_STEPS}"
  --warmup_steps 0
  --memory-probe
)

{
  printf 'RUN_DIR=%s\n' "${RUN_DIR}"
  printf 'PROBE_DIR=%s\n' "${PROBE_DIR}"
  printf 'GPU_IDS=%s\n' "${GPU_IDS}"
  printf 'EXPERT_MODE=%s\n' "${EXPERT_MODE}"
  printf 'REFERENCE_MODE=%s\n' "${REFERENCE_MODE}"
  printf 'GATE_STEPS=%s\n' "${GATE_STEPS}"
  printf 'TIMESTEP_MODE=shifted_scheduler\n'
  printf 'TRAINING_SHIFT=%s\n' "${TRAINING_SHIFT}"
  printf 'PYTORCH_CUDA_ALLOC_CONF=%s\n' "${PYTORCH_CUDA_ALLOC_CONF}"
} | tee "${PROBE_DIR}/probe_config.txt"

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
if (( ${#GPU_LIST[@]} > 1 )); then
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY_CMD[@]}" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${#GPU_LIST[@]}" \
    "${TRAIN_ARGS[@]}" 2>&1 | tee "${PROBE_DIR}/logs/gate.log"
else
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY_CMD[@]}" "${TRAIN_ARGS[@]}" --device 0 \
    2>&1 | tee "${PROBE_DIR}/logs/gate.log"
fi

printf 'Memory gate PASS: %s\n' "${PROBE_DIR}"
