#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../env/require_profile.sh"
vgm_require_profile

python "${SCRIPT_DIR}/build_all_conditions.py" "$@"
