# shellcheck shell=bash

vgm_require_profile() {
  local missing=0
  for key in VGM_PROFILE VGM_ROOT VGM_REPO_ROOT VGM_DL3DV_ROOT VGM_MODEL_ROOT VGM_OUTPUT_ROOT VGM_MANIFEST_ROOT VGM_FIRST_FRAMES_ROOT VGM_VALIDATION_ROOT; do
    if [[ -z "${!key:-}" ]]; then
      printf 'Missing %s. Activate a path profile first:\n' "${key}" >&2
      printf '  source scripts/env/activate_profile.sh local\n' >&2
      printf '  source scripts/env/activate_profile.sh cluster_zk\n' >&2
      missing=1
    fi
  done
  if [[ "${missing}" != "0" ]]; then
    return 2
  fi
  case ":${PYTHONPATH:-}:" in
    *":${VGM_REPO_ROOT}:"*) ;;
    *) export PYTHONPATH="${VGM_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
  esac
}
