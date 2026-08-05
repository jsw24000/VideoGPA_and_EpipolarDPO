#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../../env/require_profile.sh"
vgm_require_profile
CONFIG="${CONFIG:-${VGM_REPO_ROOT}/configs/videogpa/wan22_5b_t2v_smoke.yaml}"

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
  "${SCRIPT_DIR}/01_make_smoke_subset.py" \
  "${SCRIPT_DIR}/01_make_train_subset.py" \
  "${SCRIPT_DIR}/score_preferences.py" \
  "${SCRIPT_DIR}/merge_shards.py" \
  "${SCRIPT_DIR}/write_smoke_summary.py" \
  "${VGM_REPO_ROOT}/VideoGPA/generate/Wan2.2-T2V-5B.py" \
  "${VGM_REPO_ROOT}/VideoGPA/train/Wan2.2-T2V-5B/02_encode.py" \
  "${VGM_REPO_ROOT}/VideoGPA/train/Wan2.2-T2V-5B/03_train.py"

"${PY_CMD[@]}" "${SCRIPT_DIR}/00_preflight.py" --help >/dev/null
"${PY_CMD[@]}" "${SCRIPT_DIR}/01_make_smoke_subset.py" --help >/dev/null
"${PY_CMD[@]}" "${SCRIPT_DIR}/01_make_train_subset.py" --help >/dev/null
"${PY_CMD[@]}" "${SCRIPT_DIR}/score_preferences.py" --help >/dev/null
"${PY_CMD[@]}" "${SCRIPT_DIR}/merge_shards.py" --help >/dev/null
"${PY_CMD[@]}" "${SCRIPT_DIR}/write_smoke_summary.py" --help >/dev/null
if "${PY_CMD[@]}" -c "import torch" >/dev/null 2>&1; then
  "${PY_CMD[@]}" "${VGM_REPO_ROOT}/VideoGPA/generate/Wan2.2-T2V-5B.py" --help >/dev/null
  "${PY_CMD[@]}" "${VGM_REPO_ROOT}/VideoGPA/train/Wan2.2-T2V-5B/02_encode.py" --help >/dev/null
  "${PY_CMD[@]}" "${VGM_REPO_ROOT}/VideoGPA/train/Wan2.2-T2V-5B/03_train.py" --help >/dev/null
else
  printf 'static checks: skipped heavy WAN help probes because torch is not importable in this Python.\n'
fi

if command -v shellcheck >/dev/null 2>&1; then
  shellcheck \
    "${SCRIPT_DIR}/02_generate_candidates.sh" \
    "${SCRIPT_DIR}/03_score_preferences.sh" \
    "${SCRIPT_DIR}/04_encode_latents.sh" \
    "${SCRIPT_DIR}/05_train_lora_smoke.sh" \
    "${SCRIPT_DIR}/06_generate_comparison.sh" \
    "${SCRIPT_DIR}/run_formal.sh" \
    "${SCRIPT_DIR}/run_smoke.sh" \
    "${SCRIPT_DIR}/static_checks.sh"
fi

test -f "${CONFIG}"
test -f "${VGM_REPO_ROOT}/configs/videogpa/wan22_5b_t2v_formal.yaml"
test -f "${VGM_MANIFEST_ROOT}/videogpa_protocol/train_t2v.json"
test -d "${VGM_REPO_ROOT}/VideoGPA/Wan2.2"
test -d "${VGM_OUTPUT_ROOT}"

printf 'static checks passed\n'
