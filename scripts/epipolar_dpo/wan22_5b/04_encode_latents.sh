#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../../env/require_profile.sh"
vgm_require_profile

CONFIG="${CONFIG:?CONFIG is required}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"

env CONFIG="${CONFIG}" RUN_DIR="${RUN_DIR}" bash "${VGM_REPO_ROOT}/scripts/videogpa/wan22_5b_t2v/04_encode_latents.sh"
