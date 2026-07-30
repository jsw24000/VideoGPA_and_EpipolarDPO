#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  printf 'This script must be sourced, not executed.\n' >&2
  printf 'Use: source scripts/env/activate_profile.sh local\n' >&2
  printf '  or: source scripts/env/activate_profile.sh cluster_zk\n' >&2
  exit 2
fi

_vgm_activate_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_vgm_repo_root="$(cd "${_vgm_activate_dir}/../.." && pwd)"
_vgm_profile="${1:-}"

if [[ -z "${_vgm_profile}" ]]; then
  printf 'Usage: source scripts/env/activate_profile.sh <local|cluster_zk>\n' >&2
  unset _vgm_activate_dir _vgm_repo_root _vgm_profile
  return 2
fi

_vgm_profile_file="${_vgm_repo_root}/configs/paths/${_vgm_profile}.sh"
if [[ ! -f "${_vgm_profile_file}" ]]; then
  printf 'Unknown VGM path profile: %s\n' "${_vgm_profile}" >&2
  printf 'Expected profile file at: %s\n' "${_vgm_profile_file}" >&2
  unset _vgm_activate_dir _vgm_repo_root _vgm_profile _vgm_profile_file
  return 2
fi

unset VGM_PROFILE
unset VGM_ROOT
unset VGM_REPO_ROOT
unset VGM_DL3DV_ROOT
unset VGM_MODEL_ROOT
unset VGM_OUTPUT_ROOT
unset VGM_ARCHIVES_ROOT
unset VGM_EXTRACTED_ROOT
unset VGM_MANIFEST_ROOT
unset VGM_FIRST_FRAMES_ROOT
unset VGM_VALIDATION_ROOT

# shellcheck source=/dev/null
source "${_vgm_profile_file}"

VGM_ARCHIVES_ROOT="${VGM_DL3DV_ROOT}/archives"
VGM_EXTRACTED_ROOT="${VGM_DL3DV_ROOT}/extracted"
VGM_MANIFEST_ROOT="${VGM_DL3DV_ROOT}/manifests"
VGM_FIRST_FRAMES_ROOT="${VGM_DL3DV_ROOT}/first_frames"
VGM_VALIDATION_ROOT="${VGM_DL3DV_ROOT}/validation"

export VGM_PROFILE
export VGM_ROOT
export VGM_REPO_ROOT
export VGM_DL3DV_ROOT
export VGM_MODEL_ROOT
export VGM_OUTPUT_ROOT
export VGM_ARCHIVES_ROOT
export VGM_EXTRACTED_ROOT
export VGM_MANIFEST_ROOT
export VGM_FIRST_FRAMES_ROOT
export VGM_VALIDATION_ROOT

case ":${PYTHONPATH:-}:" in
  *":${VGM_REPO_ROOT}:"*) ;;
  *) export PYTHONPATH="${VGM_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

printf 'Activated VGM profile: %s\n' "${VGM_PROFILE}"
printf '  VGM_REPO_ROOT=%s\n' "${VGM_REPO_ROOT}"
printf '  VGM_DL3DV_ROOT=%s\n' "${VGM_DL3DV_ROOT}"
printf '  VGM_MODEL_ROOT=%s\n' "${VGM_MODEL_ROOT}"
printf '  VGM_OUTPUT_ROOT=%s\n' "${VGM_OUTPUT_ROOT}"

if [[ "${VGM_PROFILE}" == "local" ]]; then
  for _vgm_path in "${VGM_DL3DV_ROOT}" "${VGM_MODEL_ROOT}" "${VGM_OUTPUT_ROOT}"; do
    if [[ ! -e "${_vgm_path}" ]]; then
      printf 'Notice: local path does not exist yet: %s\n' "${_vgm_path}" >&2
    fi
  done
elif [[ "${VGM_PROFILE}" == "cluster_zk" ]]; then
  printf 'Cluster paths are resolved only; this activation does not create remote data/model/output directories.\n'
fi

unset _vgm_activate_dir _vgm_repo_root _vgm_profile _vgm_profile_file _vgm_path
