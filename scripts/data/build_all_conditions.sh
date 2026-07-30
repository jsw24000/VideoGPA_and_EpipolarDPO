#!/usr/bin/env bash
set -euo pipefail

python "$(dirname "$0")/build_all_conditions.py" "$@"
