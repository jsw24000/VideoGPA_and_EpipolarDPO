#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../../env/require_profile.sh"
vgm_require_profile
CONFIG="${CONFIG:?CONFIG is required}"

PY_CMD=()
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_CMD=("${PYTHON_BIN}")
else
  CONDA_ENV="${VIDEOGPA_CONDA_ENV:-wan22_videogpa}"
  PY_CMD=(conda run -n "${CONDA_ENV}" python)
fi

"${PY_CMD[@]}" -m py_compile \
  "${SCRIPT_DIR}/common.py" \
  "${SCRIPT_DIR}/00_preflight.py" \
  "${SCRIPT_DIR}/01_validate_source.py" \
  "${SCRIPT_DIR}/02_score_epipolar.py" \
  "${SCRIPT_DIR}/03_select_pairs.py" \
  "${SCRIPT_DIR}/merge_shards.py" \
  "${VGM_REPO_ROOT}/VideoGPA/train/dataset.py" \
  "${VGM_REPO_ROOT}/VideoGPA/train/loss.py" \
  "${VGM_REPO_ROOT}/VideoGPA/train/Wan2.2-T2V-5B/02_encode.py" \
  "${VGM_REPO_ROOT}/VideoGPA/train/Wan2.2-T2V-5B/03_train.py"

"${PY_CMD[@]}" "${SCRIPT_DIR}/00_preflight.py" --help >/dev/null
"${PY_CMD[@]}" "${SCRIPT_DIR}/01_validate_source.py" --help >/dev/null
"${PY_CMD[@]}" "${SCRIPT_DIR}/02_score_epipolar.py" --help >/dev/null
"${PY_CMD[@]}" "${SCRIPT_DIR}/03_select_pairs.py" --help >/dev/null
"${PY_CMD[@]}" "${SCRIPT_DIR}/merge_shards.py" --help >/dev/null

bash -n \
  "${SCRIPT_DIR}/02_score_epipolar.sh" \
  "${SCRIPT_DIR}/04_encode_latents.sh" \
  "${SCRIPT_DIR}/05_train_lora.sh" \
  "${SCRIPT_DIR}/run_formal.sh" \
  "${SCRIPT_DIR}/static_checks.sh"

test -f "${CONFIG}"
test -d "${VGM_OUTPUT_ROOT}"
"${PY_CMD[@]}" -m vgm_common.config --config "${CONFIG}" --print output_root | grep -F "/epipolar_dpo/" >/dev/null
"${PY_CMD[@]}" -m vgm_common.config --config "${CONFIG}" --print source_run | grep -F "/videogpa/" >/dev/null

printf 'epipolar_dpo static checks passed\n'
