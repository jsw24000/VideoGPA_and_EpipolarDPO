#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/videogpa/wan22_5b_t2v_formal.yaml"
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

CONFIG_OUTPUT_ROOT="$("${PY_CMD[@]}" -c "import pathlib, sys, yaml; cfg=yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8')) or {}; root=pathlib.Path(cfg.get('project', {}).get('project_root') if cfg.get('project', {}).get('project_root') not in (None, 'auto') else sys.argv[2]).resolve(); out=pathlib.Path(cfg.get('paths', {}).get('output_root', 'outputs/videogpa/wan2.2-5b/t2v/formal')).expanduser(); print((out if out.is_absolute() else root/out).resolve())" "${CONFIG}" "${PROJECT_ROOT}")"
OUTPUT_ROOT="${VIDEOGPA_OUTPUT_ROOT:-${CONFIG_OUTPUT_ROOT}}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}/"{config,preflight,manifests,candidates,encoded,checkpoints,comparisons,logs,reports}

export PROJECT_ROOT
export RUN_DIR
export CONFIG
export GPU_ID

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
    printf '[run_formal] skip done stage %s\n' "${stage}"
    return
  fi
  printf '[run_formal] start stage %s\n' "${stage}"
  state_update "${stage}" "running"
  set +e
  {
    printf '[run_formal] command:'
    printf ' %q' "$@"
    printf '\n'
    "$@"
  } 2>&1 | tee "${RUN_DIR}/logs/${log_name}"
  local status="${PIPESTATUS[0]}"
  set -e
  if [[ "${status}" != "0" ]]; then
    state_update "${stage}" "failed"
    printf '[run_formal] stage %s failed with %s\n' "${stage}" "${status}" >&2
    exit "${status}"
  fi
  date > "${done_marker}"
  state_update "${stage}" "done"
  printf '[run_formal] done stage %s\n' "${stage}"
  if [[ -n "${STOP_AFTER}" && "${STOP_AFTER}" == "${stage}" ]]; then
    printf '[run_formal] stopped after %s\n' "${stage}"
    exit 0
  fi
}

FORCE_ENV=0
if [[ "${FORCE_STAGE}" == "generation_candidates" || "${FORCE_STAGE}" == "encoding" ]]; then
  FORCE_ENV=1
fi

printf '#!/usr/bin/env bash\nset -euo pipefail\nVIDEOGPA_CONDA_ENV=%q GPU_ID=%q bash %q --config %q --run-id %q --resume\n' \
  "${VIDEOGPA_CONDA_ENV:-wan22_videogpa}" \
  "${GPU_ID}" \
  "${SCRIPT_DIR}/run_formal.sh" \
  "${CONFIG}" \
  "${RUN_ID}" > "${RUN_DIR}/reproduce.sh"
chmod +x "${RUN_DIR}/reproduce.sh"

run_stage preflight preflight.log "${PY_CMD[@]}" "${SCRIPT_DIR}/00_preflight.py" --config "${CONFIG}" --run-dir "${RUN_DIR}"
run_stage static_checks static_checks.log bash "${SCRIPT_DIR}/static_checks.sh"
run_stage subset subset.log "${PY_CMD[@]}" "${SCRIPT_DIR}/01_make_train_subset.py" --config "${CONFIG}" --run-dir "${RUN_DIR}"
run_stage generation_candidates generation.log env FORCE="${FORCE_ENV}" MODE=formal bash "${SCRIPT_DIR}/02_generate_candidates.sh"
run_stage scoring scoring.log env DISABLE_DEBUG_FALLBACK=1 bash "${SCRIPT_DIR}/03_score_preferences.sh"
run_stage encoding encoding.log env FORCE="${FORCE_ENV}" bash "${SCRIPT_DIR}/04_encode_latents.sh"
run_stage training training.log bash "${SCRIPT_DIR}/05_train_lora_smoke.sh"

printf '[run_formal] complete: %s\n' "${RUN_DIR}"
