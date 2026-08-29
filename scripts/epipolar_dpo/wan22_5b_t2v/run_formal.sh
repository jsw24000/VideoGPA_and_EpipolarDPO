#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EPIPOLAR_DPO_DEFAULT_CONFIG="${EPIPOLAR_DPO_DEFAULT_CONFIG:-${VGM_REPO_ROOT:-${SCRIPT_DIR}/../../..}/configs/epipolar_dpo/wan22_5b_t2v_formal.yaml}"
exec bash "${SCRIPT_DIR}/../wan22_5b/run_formal.sh" "$@"
