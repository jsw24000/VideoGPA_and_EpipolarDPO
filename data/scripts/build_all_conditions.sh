#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
printf 'WARNING: data/scripts/build_all_conditions.sh is deprecated; use scripts/data/build_all_conditions.sh.\n' >&2
bash "${SCRIPT_DIR}/../../scripts/data/build_all_conditions.sh" "$@"
