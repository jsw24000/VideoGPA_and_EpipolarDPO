#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T2V_SCRIPT_DIR="${SCRIPT_DIR}/../wan22_5b_t2v"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../../env/require_profile.sh"
vgm_require_profile
CONFIG="${CONFIG:-${VGM_REPO_ROOT}/configs/videogpa/wan22_5b_i2v_formal.yaml}"

PY_CMD=()
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_CMD=("${PYTHON_BIN}")
else
  CONDA_ENV="${VIDEOGPA_CONDA_ENV:-wan22_videogpa}"
  PY_CMD=(conda run -n "${CONDA_ENV}" python)
fi

"${PY_CMD[@]}" -m py_compile \
  "${T2V_SCRIPT_DIR}/common.py" \
  "${T2V_SCRIPT_DIR}/merge_shards.py" \
  "${SCRIPT_DIR}/01_make_train_subset.py" \
  "${VGM_REPO_ROOT}/VideoGPA/generate/Wan2.2-I2V-5B.py"

"${PY_CMD[@]}" "${SCRIPT_DIR}/01_make_train_subset.py" --help >/dev/null
"${PY_CMD[@]}" "${T2V_SCRIPT_DIR}/merge_shards.py" --help >/dev/null
if "${PY_CMD[@]}" -c "import torch, peft, PIL" >/dev/null 2>&1; then
  "${PY_CMD[@]}" "${VGM_REPO_ROOT}/VideoGPA/generate/Wan2.2-I2V-5B.py" --help >/dev/null
else
  printf 'static checks: skipped WAN I2V help probe because torch/peft/PIL is not importable in this Python.\n'
fi

if command -v shellcheck >/dev/null 2>&1; then
  shellcheck \
    "${SCRIPT_DIR}/02_generate_candidates.sh" \
    "${SCRIPT_DIR}/run_formal_generation.sh" \
    "${SCRIPT_DIR}/static_checks.sh"
fi

test -f "${CONFIG}"
test -f "${VGM_REPO_ROOT}/configs/videogpa/wan22_5b_i2v_formal.yaml"
test -f "${VGM_MANIFEST_ROOT}/videogpa_protocol/train_i2v.json"
test -d "${VGM_FIRST_FRAMES_ROOT}"
test -d "${VGM_REPO_ROOT}/VideoGPA/Wan2.2"
test -d "${VGM_OUTPUT_ROOT}"

"${PY_CMD[@]}" -m vgm_common.config --config "${CONFIG}" --print output_root | grep -F "/videogpa/wan22_5b_i2v/formal" >/dev/null

printf 'static checks passed\n'
