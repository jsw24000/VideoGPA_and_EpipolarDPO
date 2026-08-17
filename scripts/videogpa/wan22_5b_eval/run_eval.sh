#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T2V_SCRIPT_DIR="${SCRIPT_DIR}/../wan22_5b_t2v"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../../env/require_profile.sh"
vgm_require_profile

RUN_DIR=""
TASK="${TASK:-t2v}"
EVAL_NAME="${EVAL_NAME:-dl3dv1k_seed456}"
SEED="${SEED:-456}"
LIMIT="${LIMIT:-all}"
GPU_ID="${GPU_ID:-0}"
GPU_IDS="${GPU_IDS:-${GPU_ID}}"
SCORE_DEVICES="${SCORE_DEVICES:-${GPU_IDS}}"
EVAL_LORA_VARIANTS="${EVAL_LORA_VARIANTS:-}"
SKIP_GENERATION=0
SKIP_SCORE=0
MANIFEST_ONLY=0
FORCE_MANIFEST=0
FORCE_GENERATION=0
CONFIG_T2V="${VGM_REPO_ROOT}/configs/videogpa/wan22_5b_t2v_formal.yaml"
CONFIG_I2V="${VGM_REPO_ROOT}/configs/videogpa/wan22_5b_i2v_formal.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    --eval-name)
      EVAL_NAME="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --variants)
      EVAL_LORA_VARIANTS="$2"
      shift 2
      ;;
    --skip-generation)
      SKIP_GENERATION=1
      shift
      ;;
    --skip-score)
      SKIP_SCORE=1
      shift
      ;;
    --manifest-only)
      MANIFEST_ONLY=1
      shift
      ;;
    --force-manifest)
      FORCE_MANIFEST=1
      shift
      ;;
    --force-generation)
      FORCE_GENERATION=1
      shift
      ;;
    --config-t2v)
      CONFIG_T2V="$2"
      shift 2
      ;;
    --config-i2v)
      CONFIG_I2V="$2"
      shift 2
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${RUN_DIR}" ]]; then
  printf '--run-dir is required\n' >&2
  exit 2
fi
RUN_DIR="$(cd "${RUN_DIR}" && pwd)"
case "${TASK}" in
  t2v|i2v|both) ;;
  *)
    printf 'Unsupported --task=%s; use t2v, i2v, or both\n' "${TASK}" >&2
    exit 2
    ;;
esac

PY_CMD=()
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_CMD=("${PYTHON_BIN}")
else
  CONDA_ENV="${VIDEOGPA_CONDA_ENV:-wan22_videogpa}"
  PY_CMD=(conda run --no-capture-output -n "${CONDA_ENV}" python)
fi

EVAL_DIR="${RUN_DIR}/evaluation/${EVAL_NAME}"
MANIFEST_DIR="${EVAL_DIR}/manifests"
GEN_DIR="${EVAL_DIR}/generation"
SCORE_DIR="${EVAL_DIR}/scores"
LOG_DIR="${EVAL_DIR}/logs"
CONFIG_DIR="${EVAL_DIR}/config"
MANIFEST_PATH="${MANIFEST_DIR}/eval_1k_seed${SEED}.json"
mkdir -p "${MANIFEST_DIR}" "${GEN_DIR}" "${SCORE_DIR}" "${LOG_DIR}" "${CONFIG_DIR}"

if [[ -z "${EVAL_LORA_VARIANTS}" && -d "${RUN_DIR}/checkpoints/step_010000" ]]; then
  EVAL_LORA_VARIANTS="videogpa_step_010000=${RUN_DIR}/checkpoints/step_010000:0.2"
fi

{
  printf 'VGM_PROFILE=%s\n' "${VGM_PROFILE}"
  printf 'VGM_REPO_ROOT=%s\n' "${VGM_REPO_ROOT}"
  printf 'VGM_DL3DV_ROOT=%s\n' "${VGM_DL3DV_ROOT}"
  printf 'VGM_MODEL_ROOT=%s\n' "${VGM_MODEL_ROOT}"
  printf 'VGM_OUTPUT_ROOT=%s\n' "${VGM_OUTPUT_ROOT}"
  printf 'RUN_DIR=%s\n' "${RUN_DIR}"
  printf 'EVAL_DIR=%s\n' "${EVAL_DIR}"
  printf 'TASK=%s\n' "${TASK}"
  printf 'SEED=%s\n' "${SEED}"
  printf 'LIMIT=%s\n' "${LIMIT}"
  printf 'GPU_IDS=%s\n' "${GPU_IDS}"
  printf 'SCORE_DEVICES=%s\n' "${SCORE_DEVICES}"
  printf 'EVAL_LORA_VARIANTS=%s\n' "${EVAL_LORA_VARIANTS}"
} > "${CONFIG_DIR}/environment.txt"

if [[ ! -f "${MANIFEST_PATH}" || "${FORCE_MANIFEST}" == "1" ]]; then
  "${PY_CMD[@]}" "${SCRIPT_DIR}/make_eval_manifest.py" \
    --output "${MANIFEST_PATH}" \
    --seed "${SEED}" \
    --limit "${LIMIT}" | tee "${LOG_DIR}/make_eval_manifest.log"
else
  printf '[run_eval] using existing manifest: %s\n' "${MANIFEST_PATH}"
fi

if [[ "${MANIFEST_ONLY}" == "1" ]]; then
  printf '[run_eval] manifest only complete: %s\n' "${MANIFEST_PATH}"
  exit 0
fi

task_manifest_path() {
  printf '%s/%s_eval_1k_seed%s.json\n' "${MANIFEST_DIR}" "$1" "${SEED}"
}

ensure_task_manifest() {
  local task="$1"
  local task_manifest
  task_manifest="$(task_manifest_path "${task}")"
  if [[ ! -f "${task_manifest}" || "${FORCE_MANIFEST}" == "1" ]]; then
    "${PY_CMD[@]}" "${SCRIPT_DIR}/make_task_manifest.py" \
      --input "${MANIFEST_PATH}" \
      --output "${task_manifest}" \
      --task "${task}" | tee "${LOG_DIR}/make_task_manifest_${task}.log"
  else
    printf '[run_eval] using existing %s task manifest: %s\n' "${task}" "${task_manifest}"
  fi
}

resolve_da3_model() {
  if [[ -n "${SCORE_MODEL_NAME:-}" ]]; then
    printf '%s\n' "${SCORE_MODEL_NAME}"
    return
  fi
  for candidate in \
    "${VGM_MODEL_ROOT}/da3/DA3-LARGE" \
    "${VGM_MODEL_ROOT}/da3/DA3-Large" \
    "${VGM_MODEL_ROOT}/da3/DA3-large"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  printf 'DA3 model not found under %s/da3. Set SCORE_MODEL_NAME explicitly.\n' "${VGM_MODEL_ROOT}" >&2
  return 1
}

variant_names() {
  printf 'baseline\n'
  if [[ -n "${EVAL_LORA_VARIANTS}" ]]; then
    IFS=',' read -r -a specs <<< "${EVAL_LORA_VARIANTS}"
    for spec in "${specs[@]}"; do
      printf '%s\n' "${spec%%=*}"
    done
  fi
}

variant_lora_path() {
  local name="$1"
  if [[ "${name}" == "baseline" || -z "${EVAL_LORA_VARIANTS}" ]]; then
    return
  fi
  IFS=',' read -r -a specs <<< "${EVAL_LORA_VARIANTS}"
  for spec in "${specs[@]}"; do
    if [[ "${spec%%=*}" == "${name}" ]]; then
      local rhs="${spec#*=}"
      printf '%s\n' "${rhs%:*}"
      return
    fi
  done
}

variant_lora_weight() {
  local name="$1"
  if [[ "${name}" == "baseline" || -z "${EVAL_LORA_VARIANTS}" ]]; then
    return
  fi
  IFS=',' read -r -a specs <<< "${EVAL_LORA_VARIANTS}"
  for spec in "${specs[@]}"; do
    if [[ "${spec%%=*}" == "${name}" ]]; then
      local rhs="${spec#*=}"
      printf '%s\n' "${rhs##*:}"
      return
    fi
  done
}

generator_for_task() {
  case "$1" in
    t2v) printf '%s\n' "${VGM_REPO_ROOT}/VideoGPA/generate/Wan2.2-T2V-5B.py" ;;
    i2v) printf '%s\n' "${VGM_REPO_ROOT}/VideoGPA/generate/Wan2.2-I2V-5B.py" ;;
  esac
}

config_for_task() {
  case "$1" in
    t2v) printf '%s\n' "${CONFIG_T2V}" ;;
    i2v) printf '%s\n' "${CONFIG_I2V}" ;;
  esac
}

generate_variant() {
  local task="$1"
  local variant="$2"
  local lora_path="$3"
  local lora_weight="$4"
  local generator
  local config
  local task_manifest
  generator="$(generator_for_task "${task}")"
  config="$(config_for_task "${task}")"
  ensure_task_manifest "${task}"
  task_manifest="$(task_manifest_path "${task}")"
  local out_dir="${GEN_DIR}/${task}/${variant}"
  local done_marker="${EVAL_DIR}/generation_${task}_${variant}.DONE"
  local force_args=()
  local lora_args=()

  if [[ -f "${done_marker}" && "${FORCE_GENERATION}" != "1" ]]; then
    printf '[run_eval] skip done generation %s/%s\n' "${task}" "${variant}"
    return
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    printf '[run_eval] CUDA_VISIBLE_DEVICES=%s is set; unset it before physical-GPU sharded generation.\n' "${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
  fi
  if [[ "${FORCE_GENERATION}" == "1" ]]; then
    force_args+=(--force)
  fi
  if [[ -n "${lora_path}" ]]; then
    if [[ ! -d "${lora_path}" ]]; then
      printf '[run_eval] LoRA path not found for %s: %s\n' "${variant}" "${lora_path}" >&2
      exit 1
    fi
    lora_args+=(--lora_path "${lora_path}" --lora_weight "${lora_weight}")
  fi

  mkdir -p "${out_dir}"
  IFS=',' read -r -a gpu_list <<< "${GPU_IDS}"
  if (( ${#gpu_list[@]} > 1 )); then
    local shard_manifests=()
    local pids=()
    for shard_index in "${!gpu_list[@]}"; do
      local gpu="${gpu_list[${shard_index}]}"
      local shard_manifest="${MANIFEST_DIR}/${task}_${variant}.shard_${shard_index}.json"
      local shard_log="${LOG_DIR}/generate_${task}_${variant}.shard_${shard_index}.log"
      shard_manifests+=("${shard_manifest}")
      printf '[run_eval] generate %s/%s shard %s/%s on GPU %s\n' \
        "${task}" "${variant}" "${shard_index}" "${#gpu_list[@]}" "${gpu}"
      (
        "${PY_CMD[@]}" "${generator}" \
          --config "${config}" \
          --run-dir "${EVAL_DIR}" \
          --input_json "${task_manifest}" \
          --output_dir "${out_dir}" \
          --candidate_groups_json "${shard_manifest}" \
          --gpu_id "${gpu}" \
          --candidate_seeds "${SEED}" \
          --candidates_per_prompt 1 \
          --shard_index "${shard_index}" \
          --num_shards "${#gpu_list[@]}" \
          "${lora_args[@]}" \
          "${force_args[@]}"
      ) >"${shard_log}" 2>&1 &
      pids+=("$!")
    done
    local status=0
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        status=1
      fi
    done
    if [[ "${status}" != "0" ]]; then
      printf '[run_eval] generation failed for %s/%s; see %s/generate_%s_%s.shard_*.log\n' \
        "${task}" "${variant}" "${LOG_DIR}" "${task}" "${variant}" >&2
      exit "${status}"
    fi
    "${PY_CMD[@]}" "${T2V_SCRIPT_DIR}/merge_shards.py" groups \
      --output "${MANIFEST_DIR}/${task}_${variant}.candidate_groups.json" \
      --order-json "${task_manifest}" \
      "${shard_manifests[@]}"
  else
    "${PY_CMD[@]}" "${generator}" \
      --config "${config}" \
      --run-dir "${EVAL_DIR}" \
      --input_json "${task_manifest}" \
      --output_dir "${out_dir}" \
      --candidate_groups_json "${MANIFEST_DIR}/${task}_${variant}.candidate_groups.json" \
      --gpu_id "${GPU_ID}" \
      --candidate_seeds "${SEED}" \
      --candidates_per_prompt 1 \
      "${lora_args[@]}" \
      "${force_args[@]}" 2>&1 | tee "${LOG_DIR}/generate_${task}_${variant}.log"
  fi
  date > "${done_marker}"
}

score_variant() {
  local task="$1"
  local variant="$2"
  local base_dir="${GEN_DIR}/${task}/${variant}"
  local out_dir="${SCORE_DIR}/da3/${task}"
  local done_marker="${EVAL_DIR}/score_da3_${task}_${variant}.DONE"
  local da3_model
  da3_model="$(resolve_da3_model)"
  if [[ -f "${done_marker}" ]]; then
    printf '[run_eval] skip done score %s/%s\n' "${task}" "${variant}"
    return
  fi
  if [[ ! -d "${base_dir}" ]]; then
    printf '[run_eval] generation dir missing for score: %s\n' "${base_dir}" >&2
    exit 1
  fi
  mkdir -p "${out_dir}"
  (
    cd "${VGM_REPO_ROOT}/VideoGPA"
    SCORE_DEVICES="${SCORE_DEVICES}" \
    SCORE_BASE_DIR="${base_dir}" \
    SCORE_BACKBONE="da3" \
    SCORE_MODEL_NAME="${da3_model}" \
    SCORE_DESCRIPTOR_TYPE="${SCORE_DESCRIPTOR_TYPE:-lightglue}" \
    SCORE_NUM_FRAMES="${SCORE_NUM_FRAMES:-10}" \
    SCORE_CONF_THRES="${SCORE_CONF_THRES:-0}" \
    SCORE_RESUME=1 \
    SCORE_SEED_FILTER="${SEED}" \
    SCORE_OUTPUT_CSV="${out_dir}/${variant}_scores.csv" \
    SCORE_OUTPUT_JSON="${out_dir}/${variant}_scores.json" \
    "${PY_CMD[@]}" replicate_scorer.py
  ) 2>&1 | tee "${LOG_DIR}/score_da3_${task}_${variant}.log"
  date > "${done_marker}"
}

task_list=()
if [[ "${TASK}" == "both" ]]; then
  task_list=(t2v i2v)
else
  task_list=("${TASK}")
fi

if [[ "${SKIP_GENERATION}" != "1" ]]; then
  for task in "${task_list[@]}"; do
    while IFS= read -r variant; do
      generate_variant "${task}" "${variant}" "$(variant_lora_path "${variant}")" "$(variant_lora_weight "${variant}")"
    done < <(variant_names)
  done
fi

if [[ "${SKIP_SCORE}" != "1" ]]; then
  for task in "${task_list[@]}"; do
    while IFS= read -r variant; do
      score_variant "${task}" "${variant}"
    done < <(variant_names)
    while IFS= read -r variant; do
      if [[ "${variant}" == "baseline" ]]; then
        continue
      fi
      "${PY_CMD[@]}" "${VGM_REPO_ROOT}/scripts/summarize_wan22_compare_scores.py" \
        --baseline_csv "${SCORE_DIR}/da3/${task}/baseline_scores.csv" \
        --lora_csv "${SCORE_DIR}/da3/${task}/${variant}_scores.csv" \
        --output_csv "${SCORE_DIR}/da3/${task}/${variant}_vs_baseline_summary.csv" \
        --output_md "${SCORE_DIR}/da3/${task}/${variant}_vs_baseline_summary.md"
    done < <(variant_names)
  done
fi

printf '[run_eval] complete: %s\n' "${EVAL_DIR}"
