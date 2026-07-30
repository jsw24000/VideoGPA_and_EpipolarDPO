#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/videogpa/wan22_5b_t2v_smoke.yaml"
RUN_ID=""
RESUME=0
FORCE_STAGE=""
STOP_AFTER=""
GPU_ID="${GPU_ID:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
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
  SHORT_HASH="$(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || printf 'nogit')"
  RUN_ID="$(date +%Y%m%d_%H%M%S)_${SHORT_HASH}"
fi

OUTPUT_ROOT="${VIDEOGPA_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/videogpa/wan2.2-5b/t2v/smoke}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}/"{config,preflight,manifests,candidates,encoded,checkpoints,comparisons,logs,reports}

export PROJECT_ROOT
export RUN_DIR
export CONFIG
export GPU_ID

state_update() {
  local stage="$1"
  local status="$2"
  "${PY_CMD[@]}" -c "import json, pathlib, sys, datetime; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()) if p.exists() else {'run_id': sys.argv[2], 'stages': {}}; d['updated_at']=datetime.datetime.now().isoformat(); d['stages'][sys.argv[3]]=sys.argv[4]; p.write_text(json.dumps(d, indent=2), encoding='utf-8')" "${RUN_DIR}/run_state.json" "${RUN_ID}" "${stage}" "${status}"
}

run_stage() {
  local stage="$1"
  local log_name="$2"
  shift
  shift
  local done_marker="${RUN_DIR}/${stage}.DONE"
  if [[ -f "${done_marker}" && "${FORCE_STAGE}" != "${stage}" ]]; then
    printf '[run_smoke] skip done stage %s\n' "${stage}"
    return
  fi
  printf '[run_smoke] start stage %s\n' "${stage}"
  state_update "${stage}" "running"
  set +e
  {
    printf '[run_smoke] command:'
    printf ' %q' "$@"
    printf '\n'
    "$@"
  } 2>&1 | tee "${RUN_DIR}/logs/${log_name}"
  local status="${PIPESTATUS[0]}"
  set -e
  if [[ "${status}" != "0" ]]; then
    state_update "${stage}" "failed"
    printf '[run_smoke] stage %s failed with %s\n' "${stage}" "${status}" >&2
    exit "${status}"
  fi
  date > "${done_marker}"
  state_update "${stage}" "done"
  printf '[run_smoke] done stage %s\n' "${stage}"
  if [[ -n "${STOP_AFTER}" && "${STOP_AFTER}" == "${stage}" ]]; then
    printf '[run_smoke] stopped after %s\n' "${stage}"
    exit 0
  fi
}

run_scoring_stage() {
  local stage="scoring"
  local log_name="scoring.log"
  local done_marker="${RUN_DIR}/${stage}.DONE"
  if [[ -f "${done_marker}" && "${FORCE_STAGE}" != "${stage}" ]]; then
    printf '[run_smoke] skip done stage %s\n' "${stage}"
    return
  fi
  printf '[run_smoke] start stage %s\n' "${stage}"
  state_update "${stage}" "running"
  set +e
  {
    printf '[run_smoke] command:'
    printf ' %q' env DISABLE_DEBUG_FALLBACK=1 bash "${SCRIPT_DIR}/03_score_preferences.sh"
    printf '\n'
    env DISABLE_DEBUG_FALLBACK=1 bash "${SCRIPT_DIR}/03_score_preferences.sh"
    local status="$?"
    if [[ "${status}" != "0" ]] && grep -q "INSUFFICIENT_PAIRS" "${RUN_DIR}/manifests/preference_pairs.json" 2>/dev/null; then
      local fallback_size
      fallback_size="$("${PY_CMD[@]}" -c "import sys, yaml; data=yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8')) or {}; print(int(data.get('data', {}).get('fallback_subset_size', 8)))" "${CONFIG}")"
      printf '[run_smoke] strict 4-prompt scoring produced fewer than 2 pairs; extending train 8K subset to %s prompts before debug fallback\n' "${fallback_size}"
      printf '[run_smoke] command:'
      printf ' %q' "${PY_CMD[@]}" "${SCRIPT_DIR}/01_make_smoke_subset.py" --config "${CONFIG}" --run-dir "${RUN_DIR}" --subset-size "${fallback_size}"
      printf '\n'
      "${PY_CMD[@]}" "${SCRIPT_DIR}/01_make_smoke_subset.py" --config "${CONFIG}" --run-dir "${RUN_DIR}" --subset-size "${fallback_size}"
      status="$?"
      if [[ "${status}" != "0" ]]; then
        exit "${status}"
      fi
      printf '[run_smoke] command:'
      printf ' %q' env FORCE=1 MODE=formal bash "${SCRIPT_DIR}/02_generate_candidates.sh"
      printf '\n'
      env FORCE=1 MODE=formal bash "${SCRIPT_DIR}/02_generate_candidates.sh"
      status="$?"
      if [[ "${status}" != "0" ]]; then
        exit "${status}"
      fi
      printf '[run_smoke] command:'
      printf ' %q' bash "${SCRIPT_DIR}/03_score_preferences.sh"
      printf '\n'
      bash "${SCRIPT_DIR}/03_score_preferences.sh"
      status="$?"
    fi
    exit "${status}"
  } 2>&1 | tee "${RUN_DIR}/logs/${log_name}"
  local status="${PIPESTATUS[0]}"
  set -e
  if [[ "${status}" != "0" ]]; then
    state_update "${stage}" "failed"
    printf '[run_smoke] stage %s failed with %s\n' "${stage}" "${status}" >&2
    exit "${status}"
  fi
  date > "${done_marker}"
  state_update "${stage}" "done"
  printf '[run_smoke] done stage %s\n' "${stage}"
  if [[ -n "${STOP_AFTER}" && "${STOP_AFTER}" == "${stage}" ]]; then
    printf '[run_smoke] stopped after %s\n' "${stage}"
    exit 0
  fi
}

FORCE_ENV=0
if [[ "${FORCE_STAGE}" == "generation_micro" || "${FORCE_STAGE}" == "generation_candidates" || "${FORCE_STAGE}" == "encoding" ]]; then
  FORCE_ENV=1
fi

printf '#!/usr/bin/env bash\nset -euo pipefail\nVIDEOGPA_CONDA_ENV=%q bash %q --config %q --run-id %q --resume\n' \
  "${VIDEOGPA_CONDA_ENV:-wan22_videogpa}" \
  "${SCRIPT_DIR}/run_smoke.sh" \
  "${CONFIG}" \
  "${RUN_ID}" > "${RUN_DIR}/reproduce.sh"
chmod +x "${RUN_DIR}/reproduce.sh"

run_stage preflight preflight.log "${PY_CMD[@]}" "${SCRIPT_DIR}/00_preflight.py" --config "${CONFIG}" --run-dir "${RUN_DIR}"
run_stage static_checks static_checks.log bash "${SCRIPT_DIR}/static_checks.sh"
run_stage subset subset.log "${PY_CMD[@]}" "${SCRIPT_DIR}/01_make_smoke_subset.py" --config "${CONFIG}" --run-dir "${RUN_DIR}"
run_stage generation_micro generation_micro.log env FORCE="${FORCE_ENV}" MODE=micro bash "${SCRIPT_DIR}/02_generate_candidates.sh"
run_stage generation_candidates generation.log env FORCE="${FORCE_ENV}" MODE=formal bash "${SCRIPT_DIR}/02_generate_candidates.sh"
run_scoring_stage
run_stage encoding encoding.log env FORCE="${FORCE_ENV}" bash "${SCRIPT_DIR}/04_encode_latents.sh"
run_stage training training.log bash "${SCRIPT_DIR}/05_train_lora_smoke.sh"
run_stage comparison comparison.log bash "${SCRIPT_DIR}/06_generate_comparison.sh"
run_stage summary summary.log "${PY_CMD[@]}" "${SCRIPT_DIR}/write_smoke_summary.py" --config "${CONFIG}" --run-dir "${RUN_DIR}"

printf '[run_smoke] complete: %s\n' "${RUN_DIR}"
