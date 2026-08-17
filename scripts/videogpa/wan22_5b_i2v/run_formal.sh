#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T2V_SCRIPT_DIR="${SCRIPT_DIR}/../wan22_5b_t2v"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../../env/require_profile.sh"
vgm_require_profile
CONFIG="${VGM_REPO_ROOT}/configs/videogpa/wan22_5b_i2v_formal.yaml"
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
  printf '[run_formal_i2v] refusing to reuse non-empty RUN_DIR without --resume: %s\n' "${RUN_DIR}" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}/"{config,preflight,manifests,candidates,encoded,checkpoints,comparisons,logs,reports,samples,evaluation}

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
'
write_record "${RUN_DIR}/git_state.txt" bash -c '
  git -C "${VGM_REPO_ROOT}" rev-parse HEAD 2>/dev/null || true
  git -C "${VGM_REPO_ROOT}" status --short 2>/dev/null || true
'

state_update() {
  local stage="$1"
  local status="$2"
  "${PY_CMD[@]}" -c "import json, pathlib, sys, datetime; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()) if p.exists() else {'run_id': sys.argv[2], 'run_type': 'formal', 'stages': {}}; d['updated_at']=datetime.datetime.now().isoformat(); d['stages'][sys.argv[3]]=sys.argv[4]; p.write_text(json.dumps(d, indent=2), encoding='utf-8')" "${RUN_DIR}/run_state.json" "${RUN_ID}" "${stage}" "${status}"
}

run_stage() {
  local stage="$1"
  local log_name="$2"
  shift
  shift
  local done_marker="${RUN_DIR}/${stage}.DONE"
  if [[ -f "${done_marker}" && "${FORCE_STAGE}" != "${stage}" ]]; then
    printf '[run_formal_i2v] skip done stage %s\n' "${stage}"
    return
  fi
  printf '[run_formal_i2v] start stage %s\n' "${stage}"
  state_update "${stage}" "running"
  set +e
  {
    printf '[run_formal_i2v] command:'
    printf ' %q' "$@"
    printf '\n'
    "$@"
  } 2>&1 | tee "${RUN_DIR}/logs/${log_name}"
  local status="${PIPESTATUS[0]}"
  set -e
  if [[ "${status}" != "0" ]]; then
    state_update "${stage}" "failed"
    printf '[run_formal_i2v] stage %s failed with %s\n' "${stage}" "${status}" >&2
    exit "${status}"
  fi
  date > "${done_marker}"
  state_update "${stage}" "done"
  printf '[run_formal_i2v] done stage %s\n' "${stage}"
  if [[ -n "${STOP_AFTER}" && "${STOP_AFTER}" == "${stage}" ]]; then
    printf '[run_formal_i2v] stopped after %s\n' "${stage}"
    exit 0
  fi
}

has_complete_checkpoint() {
  find "${RUN_DIR}/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'step_*' \
    \( -path '*/step_*' \) -exec test -f '{}/trainer_state.json' ';' -exec test -f '{}/adapter_config.json' ';' -print -quit 2>/dev/null | grep -q .
}

FORCE_ENV=0
if [[ "${FORCE_STAGE}" == "generation_candidates" || "${FORCE_STAGE}" == "encoding" ]]; then
  FORCE_ENV=1
fi

TRAIN_ARGS=()
if [[ "${RESUME}" == "1" ]] && has_complete_checkpoint; then
  TRAIN_ARGS+=(--resume)
fi

printf '#!/usr/bin/env bash\nset -euo pipefail\nsource %q %q\nunset CUDA_VISIBLE_DEVICES\nVIDEOGPA_CONDA_ENV=%q GPU_ID=%q GPU_IDS=%q bash %q --config %q --run-id %q --resume\n' \
  "${VGM_REPO_ROOT}/scripts/env/activate_profile.sh" \
  "${VGM_PROFILE}" \
  "${VIDEOGPA_CONDA_ENV:-wan22_videogpa}" \
  "${GPU_ID}" \
  "${GPU_IDS}" \
  "${SCRIPT_DIR}/run_formal.sh" \
  "${CONFIG}" \
  "${RUN_ID}" > "${RUN_DIR}/reproduce.sh"
chmod +x "${RUN_DIR}/reproduce.sh"

run_stage preflight preflight.log "${PY_CMD[@]}" "${T2V_SCRIPT_DIR}/00_preflight.py" --config "${CONFIG}" --run-dir "${RUN_DIR}"
run_stage static_checks static_checks.log bash "${SCRIPT_DIR}/static_checks.sh"
run_stage subset subset.log "${PY_CMD[@]}" "${SCRIPT_DIR}/01_make_train_subset.py" --config "${CONFIG}" --run-dir "${RUN_DIR}"
run_stage generation_candidates generation.log env FORCE="${FORCE_ENV}" MODE=formal bash "${SCRIPT_DIR}/02_generate_candidates.sh"
run_stage scoring scoring.log env DISABLE_DEBUG_FALLBACK=1 bash "${T2V_SCRIPT_DIR}/03_score_preferences.sh"
run_stage encoding encoding.log env FORCE="${FORCE_ENV}" bash "${T2V_SCRIPT_DIR}/04_encode_latents.sh"
run_stage training training.log bash "${T2V_SCRIPT_DIR}/05_train_lora_smoke.sh" "${TRAIN_ARGS[@]}"

printf '[run_formal_i2v] complete: %s\n' "${RUN_DIR}"
