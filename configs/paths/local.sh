# shellcheck shell=bash

_vgm_profile_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VGM_REPO_ROOT="$(cd "${_vgm_profile_dir}/../.." && pwd)"
VGM_PROFILE="local"
VGM_ROOT="${VGM_REPO_ROOT}"
VGM_DL3DV_ROOT="${VGM_REPO_ROOT}/data"
VGM_MODEL_ROOT="${VGM_REPO_ROOT}/models"
VGM_OUTPUT_ROOT="${VGM_REPO_ROOT}/outputs"
unset _vgm_profile_dir
