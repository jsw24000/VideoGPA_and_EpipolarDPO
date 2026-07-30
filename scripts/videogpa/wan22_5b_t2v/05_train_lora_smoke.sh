#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../../env/require_profile.sh"
vgm_require_profile
CONFIG="${CONFIG:-${VGM_REPO_ROOT}/configs/videogpa/wan22_5b_t2v_smoke.yaml}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
GPU_ID="${GPU_ID:-0}"

PY_CMD=()
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_CMD=("${PYTHON_BIN}")
else
  CONDA_ENV="${VIDEOGPA_CONDA_ENV:-wan22_videogpa}"
  PY_CMD=(conda run -n "${CONDA_ENV}" python)
fi

"${PY_CMD[@]}" "${VGM_REPO_ROOT}/VideoGPA/train/Wan2.2-T2V-5B/03_train.py" \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --metadata_path "${RUN_DIR}/manifests/encoded_pairs.json" \
  --output_dir "${RUN_DIR}" \
  --device "${GPU_ID}"
