#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[run_wan22_compare_score] %s\n' "$*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env/require_profile.sh"
vgm_require_profile

ENV_NAME="${ENV_NAME:-wan22_videogpa}"
VIDEOGPA_DIR="${VIDEOGPA_DIR:-${VGM_REPO_ROOT}/VideoGPA}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${VGM_OUTPUT_ROOT}/evaluation/wan22_compare}"

SEED="${SEED:-42}"
FRAME_NUM="${FRAME_NUM:-81}"
SAMPLING_STEPS="${SAMPLING_STEPS:-20}"
LORA_WEIGHT="${LORA_WEIGHT:-0.2}"
LORA_TAG="$(printf '%s' "${LORA_WEIGHT}" | tr '.' 'p')"
RUN_NAME="${RUN_NAME:-dl3dv30_f${FRAME_NUM}_steps${SAMPLING_STEPS}_seed${SEED}}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"
GEN_ROOT="${RUN_ROOT}/generation"
BASELINE_DIR="${BASELINE_DIR:-${GEN_ROOT}/baseline}"
LORA_DIR="${LORA_DIR:-${GEN_ROOT}/lora_w${LORA_TAG}}"
SCORE_ROOT="${RUN_ROOT}/scores"
LOG_DIR="${RUN_ROOT}/logs"

SCORE_DEVICES="${SCORE_DEVICES:-0}"
SCORE_NUM_FRAMES="${SCORE_NUM_FRAMES:-10}"
SCORE_BACKBONE="${SCORE_BACKBONE:-vggt}"
SCORE_MODEL_NAME="${SCORE_MODEL_NAME:-${VGM_MODEL_ROOT}/vggt/VGGT-1B}"
SCORE_DESCRIPTOR_TYPE="${SCORE_DESCRIPTOR_TYPE:-sift}"
INSTALL_SCORE_DEPS="${INSTALL_SCORE_DEPS:-0}"

mkdir -p "${SCORE_ROOT}" "${LOG_DIR}"

if [ ! -d "${BASELINE_DIR}" ]; then
  printf 'Baseline generation dir not found: %s\n' "${BASELINE_DIR}" >&2
  exit 1
fi

if [ ! -d "${LORA_DIR}" ]; then
  printf 'LoRA generation dir not found: %s\n' "${LORA_DIR}" >&2
  exit 1
fi

if [ ! -f "${VIDEOGPA_DIR}/replicate_scorer.py" ]; then
  printf 'VideoGPA scorer not found under: %s\n' "${VIDEOGPA_DIR}" >&2
  exit 1
fi

check_score_deps() {
  SCORE_DESCRIPTOR_TYPE="${SCORE_DESCRIPTOR_TYPE}" conda run -n "${ENV_NAME}" python -c "import importlib
import os
missing = []
mods = ['lpips', 'piq', 'kornia', 'plyfile', 'decord']
if os.environ.get('SCORE_DESCRIPTOR_TYPE', '').lower() == 'lightglue':
    mods.append('lightglue')
for mod in mods:
    try:
        importlib.import_module(mod)
    except Exception as exc:
        missing.append(f'{mod}: {type(exc).__name__}: {exc}')
if missing:
    print('\n'.join(missing))
    raise SystemExit(1)
print('score_deps_ok')"
}

if ! check_score_deps; then
  if [ "${INSTALL_SCORE_DEPS}" = "1" ]; then
    log "Installing missing lightweight scorer dependencies"
    conda run -n "${ENV_NAME}" python -m pip install lpips piq kornia "numpy==1.26.4" "plyfile==1.0.3"
    if [ "${SCORE_DESCRIPTOR_TYPE}" = "lightglue" ]; then
      conda run -n "${ENV_NAME}" python -m pip install "git+https://github.com/cvg/LightGlue.git"
    fi
    check_score_deps
  else
    cat >&2 <<EOF
Missing scorer dependencies.
Install them with:

  conda run -n ${ENV_NAME} python -m pip install lpips piq kornia 'numpy==1.26.4' 'plyfile==1.0.3'

or rerun this script with:

  INSTALL_SCORE_DEPS=1 bash scripts/run_wan22_compare_score.sh

If you explicitly set SCORE_DESCRIPTOR_TYPE=lightglue, install LightGlue with:

  conda run -n ${ENV_NAME} python -m pip install 'git+https://github.com/cvg/LightGlue.git'
EOF
    exit 1
  fi
fi

score_variant() {
  local label="$1"
  local base_dir="$2"
  local csv_file="${SCORE_ROOT}/${label}_scores.csv"
  local json_file="${SCORE_ROOT}/${label}_scores.json"
  local log_file="${LOG_DIR}/score_${label}.log"

  log "Scoring ${label}; videos: ${base_dir}"
  (
    cd "${VIDEOGPA_DIR}"
    SCORE_DEVICES="${SCORE_DEVICES}" \
    SCORE_BASE_DIR="${base_dir}" \
    SCORE_BACKBONE="${SCORE_BACKBONE}" \
    SCORE_MODEL_NAME="${SCORE_MODEL_NAME}" \
    SCORE_DESCRIPTOR_TYPE="${SCORE_DESCRIPTOR_TYPE}" \
    SCORE_NUM_FRAMES="${SCORE_NUM_FRAMES}" \
    SCORE_OUTPUT_CSV="${csv_file}" \
    SCORE_OUTPUT_JSON="${json_file}" \
    conda run -n "${ENV_NAME}" python replicate_scorer.py
  ) 2>&1 | tee "${log_file}"
}

score_variant baseline "${BASELINE_DIR}"
score_variant "lora_w${LORA_TAG}" "${LORA_DIR}"

conda run -n "${ENV_NAME}" python "${SCRIPT_DIR}/summarize_wan22_compare_scores.py" \
  --baseline_csv "${SCORE_ROOT}/baseline_scores.csv" \
  --lora_csv "${SCORE_ROOT}/lora_w${LORA_TAG}_scores.csv" \
  --output_csv "${SCORE_ROOT}/compare_summary.csv" \
  --output_md "${SCORE_ROOT}/compare_summary.md"

log "Score root: ${SCORE_ROOT}"
