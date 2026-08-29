#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../../env/require_profile.sh"
vgm_require_profile

CONFIG="${EPIPOLAR_DPO_DEFAULT_CONFIG:-${VGM_REPO_ROOT}/configs/epipolar_dpo/wan22_5b_t2v_formal.yaml}"
RUN_ID=""
RESUME=0
FORCE_STAGE=""
STOP_AFTER=""
GPU_ID="${GPU_ID:-0}"
GPU_IDS="${GPU_IDS:-${GPU_ID}}"
ORIGINAL_ARGS=("$@")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      if [[ "$2" = /* ]]; then
        CONFIG="$2"
      else
        CONFIG="${VGM_REPO_ROOT}/$2"
      fi
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --force-stage)
      FORCE_STAGE="$2"
      shift 2
      ;;
    --stop-after)
      STOP_AFTER="$2"
      shift 2
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

PY_CMD=()
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_CMD=("${PYTHON_BIN}")
else
  CONDA_ENV="${VIDEOGPA_CONDA_ENV:-wan22_videogpa}"
  PY_CMD=(conda run -n "${CONDA_ENV}" python)
fi

if [[ -z "${RUN_ID}" ]]; then
  SHORT_HASH="$(git -C "${VGM_REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || printf 'nogit')"
  RUN_ID="$(date +%Y%m%d_%H%M%S)_${SHORT_HASH}"
fi

OUTPUT_ROOT="$("${PY_CMD[@]}" -m vgm_common.config --config "${CONFIG}" --print output_root)"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
if [[ -d "${RUN_DIR}" && "${RESUME}" != "1" ]] && find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  printf '[epipolar_run_formal] refusing to reuse non-empty RUN_DIR without --resume: %s\n' "${RUN_DIR}" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}/"{config,preflight,manifests,encoded,checkpoints,logs,reports,evaluation}

export RUN_DIR
export CONFIG
export GPU_ID
export GPU_IDS

write_record() {
  local path="$1"
  shift
  if [[ "${RESUME}" == "1" && -e "${path}" ]]; then
    path="${path%.*}.resume_$(date +%Y%m%d_%H%M%S).${path##*.}"
  fi
  "$@" > "${path}"
}

write_record "${RUN_DIR}/command.txt" bash -c 'printf "bash"; printf " %q" "$0" "$@"; printf "\n"' "$0" "${ORIGINAL_ARGS[@]}"
write_record "${RUN_DIR}/environment.txt" bash -c '
  printf "VGM_PROFILE=%s\n" "${VGM_PROFILE}"
  printf "VGM_REPO_ROOT=%s\n" "${VGM_REPO_ROOT}"
  printf "VGM_DL3DV_ROOT=%s\n" "${VGM_DL3DV_ROOT}"
  printf "VGM_MODEL_ROOT=%s\n" "${VGM_MODEL_ROOT}"
  printf "VGM_OUTPUT_ROOT=%s\n" "${VGM_OUTPUT_ROOT}"
  printf "VGM_FIRST_FRAMES_ROOT=%s\n" "${VGM_FIRST_FRAMES_ROOT}"
  printf "GPU_ID=%s\n" "${GPU_ID}"
  printf "GPU_IDS=%s\n" "${GPU_IDS}"
  printf "CUDA_VISIBLE_DEVICES=%s\n" "${CUDA_VISIBLE_DEVICES:-}"
  printf "PYTHONPATH=%s\n" "${PYTHONPATH:-}"
  printf "EPIPOLAR_DPO_MAX_GROUPS=%s\n" "${EPIPOLAR_DPO_MAX_GROUPS:-}"
  printf "EPIPOLAR_DPO_MAX_TRAIN_STEPS=%s\n" "${EPIPOLAR_DPO_MAX_TRAIN_STEPS:-}"
  printf "EPIPOLAR_DPO_TRAIN_WARMUP_STEPS=%s\n" "${EPIPOLAR_DPO_TRAIN_WARMUP_STEPS:-}"
'
write_record "${RUN_DIR}/git_state.txt" bash -c '
  git -C "${VGM_REPO_ROOT}" rev-parse HEAD 2>/dev/null || true
  git -C "${VGM_REPO_ROOT}" status --short 2>/dev/null || true
'

state_update() {
  local stage="$1"
  local status="$2"
  "${PY_CMD[@]}" -c "import json, pathlib, sys, datetime; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()) if p.exists() else {'run_id': sys.argv[2], 'run_type': 'formal', 'method': 'epipolar_dpo', 'stages': {}}; d['updated_at']=datetime.datetime.now().isoformat(); d['stages'][sys.argv[3]]=sys.argv[4]; p.write_text(json.dumps(d, indent=2), encoding='utf-8')" "${RUN_DIR}/run_state.json" "${RUN_ID}" "${stage}" "${status}"
}

FORCE_ACTIVE=0
run_stage() {
  local stage="$1"
  local log_name="$2"
  shift
  shift
  if [[ -n "${FORCE_STAGE}" && "${FORCE_STAGE}" == "${stage}" ]]; then
    FORCE_ACTIVE=1
  fi
  local done_marker="${RUN_DIR}/${stage}.DONE"
  if [[ -f "${done_marker}" && "${FORCE_ACTIVE}" != "1" ]]; then
    printf '[epipolar_run_formal] skip done stage %s\n' "${stage}"
    return
  fi
  printf '[epipolar_run_formal] start stage %s\n' "${stage}"
  state_update "${stage}" "running"
  set +e
  {
    printf '[epipolar_run_formal] command:'
    printf ' %q' "$@"
    printf '\n'
    "$@"
  } 2>&1 | tee "${RUN_DIR}/logs/${log_name}"
  local status="${PIPESTATUS[0]}"
  set -e
  if [[ "${status}" != "0" ]]; then
    state_update "${stage}" "failed"
    printf '[epipolar_run_formal] stage %s failed with %s\n' "${stage}" "${status}" >&2
    exit "${status}"
  fi
  date > "${done_marker}"
  state_update "${stage}" "done"
  printf '[epipolar_run_formal] done stage %s\n' "${stage}"
  if [[ -n "${STOP_AFTER}" && "${STOP_AFTER}" == "${stage}" ]]; then
    printf '[epipolar_run_formal] stopped after %s\n' "${stage}"
    exit 0
  fi
}

has_complete_checkpoint() {
  find "${RUN_DIR}/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'step_*' \
    -exec test -f '{}/trainer_state.json' ';' -exec test -f '{}/adapter_config.json' ';' -print -quit 2>/dev/null | grep -q .
}

TRAIN_ARGS=()
if [[ "${RESUME}" == "1" ]] && has_complete_checkpoint; then
  TRAIN_ARGS+=(--resume)
fi

FORCE_SCORING=0
if [[ "${FORCE_STAGE}" == "epipolar_scoring" ]]; then
  FORCE_SCORING=1
fi
FORCE_ENCODING=0
if [[ "${FORCE_STAGE}" == "encoding" ]]; then
  FORCE_ENCODING=1
fi

MAX_GROUP_ARGS=()
if [[ -n "${EPIPOLAR_DPO_MAX_GROUPS:-}" ]]; then
  MAX_GROUP_ARGS+=(--max-groups "${EPIPOLAR_DPO_MAX_GROUPS}")
fi

printf '#!/usr/bin/env bash\nset -euo pipefail\nsource %q %q\nunset CUDA_VISIBLE_DEVICES\nVIDEOGPA_CONDA_ENV=%q GPU_ID=%q GPU_IDS=%q EPIPOLAR_DPO_MAX_GROUPS=%q EPIPOLAR_DPO_MAX_TRAIN_STEPS=%q EPIPOLAR_DPO_TRAIN_WARMUP_STEPS=%q bash %q --config %q --run-id %q --resume\n' \
  "${VGM_REPO_ROOT}/scripts/env/activate_profile.sh" \
  "${VGM_PROFILE}" \
  "${VIDEOGPA_CONDA_ENV:-wan22_videogpa}" \
  "${GPU_ID}" \
  "${GPU_IDS}" \
  "${EPIPOLAR_DPO_MAX_GROUPS:-}" \
  "${EPIPOLAR_DPO_MAX_TRAIN_STEPS:-}" \
  "${EPIPOLAR_DPO_TRAIN_WARMUP_STEPS:-}" \
  "${SCRIPT_DIR}/run_formal.sh" \
  "${CONFIG}" \
  "${RUN_ID}" > "${RUN_DIR}/reproduce.sh"
chmod +x "${RUN_DIR}/reproduce.sh"

run_stage preflight preflight.log "${PY_CMD[@]}" "${SCRIPT_DIR}/00_preflight.py" --config "${CONFIG}" --run-dir "${RUN_DIR}"
run_stage static_checks static_checks.log bash "${SCRIPT_DIR}/static_checks.sh"
run_stage source_validation source_validation.log "${PY_CMD[@]}" "${SCRIPT_DIR}/01_validate_source.py" --config "${CONFIG}" --run-dir "${RUN_DIR}" "${MAX_GROUP_ARGS[@]}"
run_stage epipolar_scoring epipolar_scoring.log env FORCE="${FORCE_SCORING}" bash "${SCRIPT_DIR}/02_score_epipolar.sh" "${MAX_GROUP_ARGS[@]}"
run_stage pair_selection pair_selection.log "${PY_CMD[@]}" "${SCRIPT_DIR}/03_select_pairs.py" --config "${CONFIG}" --run-dir "${RUN_DIR}"
run_stage encoding encoding.log env FORCE="${FORCE_ENCODING}" bash "${SCRIPT_DIR}/04_encode_latents.sh"
run_stage training training.log bash "${SCRIPT_DIR}/05_train_lora.sh" "${TRAIN_ARGS[@]}"

printf '[epipolar_run_formal] complete: %s\n' "${RUN_DIR}"
