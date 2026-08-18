#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T2V_SCRIPT_DIR="${SCRIPT_DIR}/../wan22_5b_t2v"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../../env/require_profile.sh"
vgm_require_profile

RUN_DIR=""
TASK="${TASK:-}"
EVAL_NAME="${EVAL_NAME:-dl3dv1k_seed456}"
SEED="${SEED:-456}"
LIMIT="${LIMIT:-100}"
GPU_ID="${GPU_ID:-0}"
GPU_IDS="${GPU_IDS:-${GPU_ID}}"
SCORE_DEVICES="${SCORE_DEVICES:-${GPU_IDS}}"
EVAL_VARIANT="${EVAL_VARIANT:-${EVAL_LORA_VARIANT:-}}"
RUN_BASELINE="${EVAL_RUN_BASELINE:-1}"
BASELINE_ONLY=0
SKIP_GENERATION=0
SKIP_SCORE=0
MANIFEST_ONLY=0
FORCE_MANIFEST=0
FORCE_GENERATION=0
CONFIG_T2V="${VGM_REPO_ROOT}/configs/videogpa/wan22_5b_t2v_formal.yaml"
CONFIG_I2V="${VGM_REPO_ROOT}/configs/videogpa/wan22_5b_i2v_formal.yaml"
CONFIG_14B_T2V="${VGM_REPO_ROOT}/configs/videogpa/wan22_14b_t2v_formal.yaml"
CONFIG_14B_I2V="${VGM_REPO_ROOT}/configs/videogpa/wan22_14b_i2v_formal.yaml"

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
    --variant|--variants)
      EVAL_VARIANT="$2"
      shift 2
      ;;
    --skip-baseline)
      RUN_BASELINE=0
      shift
      ;;
    --baseline-only)
      RUN_BASELINE=1
      BASELINE_ONLY=1
      shift
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
    --config-14b-t2v)
      CONFIG_14B_T2V="$2"
      shift 2
      ;;
    --config-14b-i2v)
      CONFIG_14B_I2V="$2"
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
TASK_MANIFEST_PATH="${MANIFEST_DIR}/task_eval_1k_seed${SEED}.json"

RUN_CONFIG=""
RUN_CONFIG_TASK=""
RUN_CONFIG_METHOD=""
RUN_CONFIG_SCALE=""
if [[ -f "${RUN_DIR}/config_resolved.yaml" ]]; then
  RUN_CONFIG="${RUN_DIR}/config_resolved.yaml"
  IFS=$'\t' read -r RUN_CONFIG_TASK RUN_CONFIG_METHOD RUN_CONFIG_SCALE < <(
    "${PY_CMD[@]}" -c 'import sys, yaml; data=yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}; project=data.get("project") or {}; print("\t".join(str(project.get(key, "")) for key in ("task", "method", "model_scale")))' "${RUN_CONFIG}"
  )
fi

if [[ -z "${TASK}" ]]; then
  TASK="${RUN_CONFIG_TASK}"
fi
case "${TASK}" in
  t2v|i2v) ;;
  *)
    printf 'Could not infer a single task from %s; pass --task t2v or --task i2v.\n' "${RUN_CONFIG:-RUN_DIR}" >&2
    exit 2
    ;;
esac
if [[ -n "${RUN_CONFIG_TASK}" && "${RUN_CONFIG_TASK}" != "${TASK}" ]]; then
  printf 'Task mismatch: RUN_DIR config is %s but --task is %s.\n' "${RUN_CONFIG_TASK}" "${TASK}" >&2
  exit 2
fi
case "${RUN_BASELINE}" in
  1|true|TRUE|yes|YES) RUN_BASELINE=1 ;;
  0|false|FALSE|no|NO) RUN_BASELINE=0 ;;
  *)
    printf 'EVAL_RUN_BASELINE must be 0/1 or true/false, got %s\n' "${RUN_BASELINE}" >&2
    exit 2
    ;;
esac
if [[ "${EVAL_VARIANT}" == *,* ]]; then
  printf 'Only one fine-tuned variant is allowed per RUN_DIR; got: %s\n' "${EVAL_VARIANT}" >&2
  exit 2
fi

mkdir -p "${MANIFEST_DIR}" "${GEN_DIR}" "${SCORE_DIR}" "${LOG_DIR}" "${CONFIG_DIR}"

adapter_complete() {
  local adapter_dir="$1"
  [[ -f "${adapter_dir}/adapter_config.json" ]] || return 1
  [[ -f "${adapter_dir}/adapter_model.safetensors" || -f "${adapter_dir}/adapter_model.bin" ]]
}

checkpoint_complete() {
  local ckpt="$1"
  [[ -d "${ckpt}" ]] || return 1
  if adapter_complete "${ckpt}"; then
    return 0
  fi
  adapter_complete "${ckpt}/low_noise_model" && adapter_complete "${ckpt}/high_noise_model"
}

discover_latest_lora_variant() {
  local method="${EVAL_METHOD_NAME:-${RUN_CONFIG_METHOD:-videogpa}}"
  method="$(printf '%s' "${method}" | tr '[:upper:] -' '[:lower:]__')"
  local best_step=-1
  local best_dir=""
  local ckpt
  for ckpt in "${RUN_DIR}"/checkpoints/step_*; do
    [[ -d "${ckpt}" ]] || continue
    checkpoint_complete "${ckpt}" || continue
    local name="${ckpt##*/}"
    local step="${name#step_}"
    if [[ "${step}" =~ ^[0-9]+$ ]] && (( 10#${step} > best_step )); then
      best_step=$((10#${step}))
      best_dir="${ckpt}"
    fi
  done
  if [[ -n "${best_dir}" ]]; then
    printf '%s_step_%06d=%s:0.2\n' "${method}" "${best_step}" "${best_dir}"
  fi
}

if [[ -z "${EVAL_VARIANT}" && "${BASELINE_ONLY}" != "1" ]]; then
  EVAL_VARIANT="$(discover_latest_lora_variant || true)"
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
  printf 'RUN_BASELINE=%s\n' "${RUN_BASELINE}"
  printf 'BASELINE_ONLY=%s\n' "${BASELINE_ONLY}"
  printf 'EVAL_VARIANT=%s\n' "${EVAL_VARIANT}"
  printf 'RUN_CONFIG=%s\n' "${RUN_CONFIG}"
  printf 'RUN_CONFIG_TASK=%s\n' "${RUN_CONFIG_TASK}"
  printf 'RUN_CONFIG_METHOD=%s\n' "${RUN_CONFIG_METHOD}"
  printf 'RUN_CONFIG_SCALE=%s\n' "${RUN_CONFIG_SCALE}"
} > "${CONFIG_DIR}/environment.txt"

if [[ "${FORCE_MANIFEST}" == "1" ]] && find "${GEN_DIR}" -type f -name "seed_${SEED}.mp4" -print -quit 2>/dev/null | grep -q .; then
  printf 'Refusing --force-manifest because generated seed_%s videos already exist under %s. Use a new --eval-name or remove the old eval directory first.\n' \
    "${SEED}" "${GEN_DIR}" >&2
  exit 2
fi

if [[ ! -f "${MANIFEST_PATH}" || "${FORCE_MANIFEST}" == "1" ]]; then
  "${PY_CMD[@]}" "${SCRIPT_DIR}/make_eval_manifest.py" \
    --output "${MANIFEST_PATH}" \
    --seed "${SEED}" \
    --limit "${LIMIT}" | tee "${LOG_DIR}/make_eval_manifest.log"
else
  printf '[run_eval] using existing manifest: %s\n' "${MANIFEST_PATH}"
fi

EXPECTED_SAMPLES="${LIMIT}"
if [[ "${EXPECTED_SAMPLES}" == "all" ]]; then
  EXPECTED_SAMPLES=1000
fi
MANIFEST_SAMPLES="$("${PY_CMD[@]}" -c 'import json, sys; data=json.load(open(sys.argv[1], encoding="utf-8")); print(data.get("num_samples", ""))' "${MANIFEST_PATH}")"
MANIFEST_SEED="$("${PY_CMD[@]}" -c 'import json, sys; data=json.load(open(sys.argv[1], encoding="utf-8")); print(data.get("seed", ""))' "${MANIFEST_PATH}")"
if [[ "${MANIFEST_SAMPLES}" != "${EXPECTED_SAMPLES}" || "${MANIFEST_SEED}" != "${SEED}" ]]; then
  printf 'Existing eval manifest does not match this run: samples=%s seed=%s, expected samples=%s seed=%s. Remove %s or choose a new --eval-name.\n' \
    "${MANIFEST_SAMPLES}" "${MANIFEST_SEED}" "${EXPECTED_SAMPLES}" "${SEED}" "${EVAL_DIR}" >&2
  exit 2
fi

ensure_task_manifest() {
  if [[ ! -f "${TASK_MANIFEST_PATH}" || "${FORCE_MANIFEST}" == "1" ]]; then
    "${PY_CMD[@]}" "${SCRIPT_DIR}/make_task_manifest.py" \
      --input "${MANIFEST_PATH}" \
      --output "${TASK_MANIFEST_PATH}" \
      --task "${TASK}" | tee "${LOG_DIR}/make_task_manifest.log"
  else
    local task_manifest_info
    task_manifest_info="$("${PY_CMD[@]}" -c 'import json, sys; data=json.load(open(sys.argv[1], encoding="utf-8")); print("{}\t{}".format(data.get("task", "?"), data.get("num_samples", "?")))' "${TASK_MANIFEST_PATH}")"
    if [[ "${task_manifest_info}" != "${TASK}"$'\t'"${EXPECTED_SAMPLES}" ]]; then
      printf 'Existing task manifest does not match task=%s samples=%s: %s. Remove the eval directory or use --force-manifest before generation exists.\n' \
        "${TASK}" "${EXPECTED_SAMPLES}" "${TASK_MANIFEST_PATH}" >&2
      exit 2
    fi
    printf '[run_eval] using task manifest: %s\n' "${TASK_MANIFEST_PATH}"
  fi
}

if [[ "${MANIFEST_ONLY}" == "1" ]]; then
  ensure_task_manifest
  printf '[run_eval] manifest only complete: %s and %s\n' "${MANIFEST_PATH}" "${TASK_MANIFEST_PATH}"
  exit 0
fi

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
  if [[ "${RUN_BASELINE}" == "1" ]]; then
    printf 'baseline\n'
  fi
  if [[ "${BASELINE_ONLY}" != "1" && -n "${EVAL_VARIANT}" ]]; then
    printf '%s\n' "${EVAL_VARIANT%%=*}"
  fi
}

variant_lora_path() {
  local name="$1"
  if [[ "${name}" == "baseline" || -z "${EVAL_VARIANT}" ]]; then
    return
  fi
  local rhs="${EVAL_VARIANT#*=}"
  if [[ "${rhs}" == *:* ]]; then
    printf '%s\n' "${rhs%:*}"
  else
    printf '%s\n' "${rhs}"
  fi
}

variant_lora_weight() {
  local name="$1"
  if [[ "${name}" == "baseline" || -z "${EVAL_VARIANT}" ]]; then
    return
  fi
  local rhs="${EVAL_VARIANT#*=}"
  if [[ "${rhs}" == *:* ]]; then
    printf '%s\n' "${rhs##*:}"
  else
    printf '0.2\n'
  fi
}

assert_variant_video_count() {
  local variant="$1"
  local base_dir="$2"
  local actual
  if [[ ! -d "${base_dir}" ]]; then
    printf 'Generation directory missing for %s: %s\n' "${variant}" "${base_dir}" >&2
    exit 1
  fi
  actual="$(find "${base_dir}" -type f -name "seed_${SEED}.mp4" -size +0c 2>/dev/null | wc -l)"
  if [[ "${actual}" != "${EXPECTED_SAMPLES}" ]]; then
    printf 'Expected exactly %s seed_%s videos for %s, found %s under %s. Refusing to score stale or incomplete output.\n' \
      "${EXPECTED_SAMPLES}" "${SEED}" "${variant}" "${actual}" "${base_dir}" >&2
    exit 1
  fi
}

config_is_a14b() {
  grep -Eq 'dual_expert_a14b|t2v-A14B|i2v-A14B' "$1"
}

config_for_task() {
  if [[ -n "${RUN_CONFIG}" && "${RUN_CONFIG_TASK}" == "$1" ]]; then
    printf '%s\n' "${RUN_CONFIG}"
    return
  fi
  case "$1" in
    t2v)
      if [[ "${RUN_DIR}" == *"wan22_14b_t2v"* ]]; then
        printf '%s\n' "${CONFIG_14B_T2V}"
      else
        printf '%s\n' "${CONFIG_T2V}"
      fi
      ;;
    i2v)
      if [[ "${RUN_DIR}" == *"wan22_14b_i2v"* ]]; then
        printf '%s\n' "${CONFIG_14B_I2V}"
      else
        printf '%s\n' "${CONFIG_I2V}"
      fi
      ;;
  esac
}

generator_for_task_config() {
  local task="$1"
  local config="$2"
  if config_is_a14b "${config}"; then
    printf '%s\n' "${VGM_REPO_ROOT}/VideoGPA/generate/Wan2.2-A14B.py"
    return
  fi
  case "${task}" in
    t2v) printf '%s\n' "${VGM_REPO_ROOT}/VideoGPA/generate/Wan2.2-T2V-5B.py" ;;
    i2v) printf '%s\n' "${VGM_REPO_ROOT}/VideoGPA/generate/Wan2.2-I2V-5B.py" ;;
  esac
}

generate_variant() {
  local variant="$1"
  local lora_path="$2"
  local lora_weight="$3"
  local generator
  local config
  config="$(config_for_task "${TASK}")"
  generator="$(generator_for_task_config "${TASK}" "${config}")"
  ensure_task_manifest
  local out_dir="${GEN_DIR}/${variant}"
  local done_marker="${EVAL_DIR}/generation_${variant}.DONE"
  local force_args=()
  local lora_args=()
  local is_a14b=0
  if [[ "${generator}" == *"Wan2.2-A14B.py" ]]; then
    is_a14b=1
  fi

  if [[ -f "${done_marker}" && "${FORCE_GENERATION}" != "1" ]]; then
    printf '[run_eval] skip done generation %s\n' "${variant}"
    assert_variant_video_count "${variant}" "${out_dir}"
    return
  fi
  if [[ "${is_a14b}" != "1" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
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
  if [[ "${is_a14b}" == "1" ]]; then
    if (( ${#gpu_list[@]} > 1 )); then
      local extra_args=()
      if [[ "${DIT_FSDP:-1}" == "1" ]]; then
        extra_args+=(--dit_fsdp)
      fi
      if [[ "${T5_FSDP:-1}" == "1" ]]; then
        extra_args+=(--t5_fsdp)
      fi
      if [[ "${USE_SP:-1}" == "1" ]]; then
        extra_args+=(--ulysses_size "${ULYSSES_SIZE:-${#gpu_list[@]}}")
      fi
      printf '[run_eval] distributed A14B generation for %s: GPU_IDS=%s DIT_FSDP=%s T5_FSDP=%s USE_SP=%s ULYSSES_SIZE=%s\n' \
        "${variant}" "${GPU_IDS}" "${DIT_FSDP:-1}" "${T5_FSDP:-1}" "${USE_SP:-1}" "${ULYSSES_SIZE:-${#gpu_list[@]}}"
      CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY_CMD[@]}" -m torch.distributed.run \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${#gpu_list[@]}" \
        "${generator}" \
        --config "${config}" \
        --run-dir "${EVAL_DIR}" \
        --input_json "${TASK_MANIFEST_PATH}" \
        --output_dir "${out_dir}" \
        --candidate_groups_json "${MANIFEST_DIR}/${variant}.candidate_groups.json" \
        --gpu_id 0 \
        --candidate_seeds "${SEED}" \
        --candidates_per_prompt 1 \
        "${extra_args[@]}" \
        "${lora_args[@]}" \
        "${force_args[@]}" 2>&1 | tee "${LOG_DIR}/generate_${variant}.log"
    else
      "${PY_CMD[@]}" "${generator}" \
        --config "${config}" \
        --run-dir "${EVAL_DIR}" \
        --input_json "${TASK_MANIFEST_PATH}" \
        --output_dir "${out_dir}" \
        --candidate_groups_json "${MANIFEST_DIR}/${variant}.candidate_groups.json" \
        --gpu_id "${GPU_ID}" \
        --candidate_seeds "${SEED}" \
        --candidates_per_prompt 1 \
        "${lora_args[@]}" \
        "${force_args[@]}" 2>&1 | tee "${LOG_DIR}/generate_${variant}.log"
    fi
  elif (( ${#gpu_list[@]} > 1 )); then
    local shard_manifests=()
    local pids=()
    for shard_index in "${!gpu_list[@]}"; do
      local gpu="${gpu_list[${shard_index}]}"
      local shard_manifest="${MANIFEST_DIR}/${variant}.shard_${shard_index}.json"
      local shard_log="${LOG_DIR}/generate_${variant}.shard_${shard_index}.log"
      shard_manifests+=("${shard_manifest}")
      printf '[run_eval] generate %s shard %s/%s on GPU %s\n' \
        "${variant}" "${shard_index}" "${#gpu_list[@]}" "${gpu}"
      (
        "${PY_CMD[@]}" "${generator}" \
          --config "${config}" \
          --run-dir "${EVAL_DIR}" \
          --input_json "${TASK_MANIFEST_PATH}" \
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
      printf '[run_eval] generation failed for %s; see %s/generate_%s.shard_*.log\n' \
        "${variant}" "${LOG_DIR}" "${variant}" >&2
      exit "${status}"
    fi
    "${PY_CMD[@]}" "${T2V_SCRIPT_DIR}/merge_shards.py" groups \
      --output "${MANIFEST_DIR}/${variant}.candidate_groups.json" \
      --order-json "${TASK_MANIFEST_PATH}" \
      "${shard_manifests[@]}"
  else
    "${PY_CMD[@]}" "${generator}" \
      --config "${config}" \
      --run-dir "${EVAL_DIR}" \
      --input_json "${TASK_MANIFEST_PATH}" \
      --output_dir "${out_dir}" \
      --candidate_groups_json "${MANIFEST_DIR}/${variant}.candidate_groups.json" \
      --gpu_id "${GPU_ID}" \
      --candidate_seeds "${SEED}" \
      --candidates_per_prompt 1 \
      "${lora_args[@]}" \
      "${force_args[@]}" 2>&1 | tee "${LOG_DIR}/generate_${variant}.log"
  fi
  assert_variant_video_count "${variant}" "${out_dir}"
  date > "${done_marker}"
}

score_variant() {
  local variant="$1"
  local base_dir="${GEN_DIR}/${variant}"
  local out_dir="${SCORE_DIR}/da3"
  local done_marker="${EVAL_DIR}/score_da3_${variant}.DONE"
  local da3_model
  da3_model="$(resolve_da3_model)"
  if [[ -f "${done_marker}" ]]; then
    printf '[run_eval] skip done score %s\n' "${variant}"
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
  ) 2>&1 | tee "${LOG_DIR}/score_da3_${variant}.log"
  date > "${done_marker}"
}

if [[ "${BASELINE_ONLY}" != "1" && -z "${EVAL_VARIANT}" ]]; then
  printf 'No complete checkpoint found under %s/checkpoints. Pass --variant NAME=PATH[:WEIGHT] or use --baseline-only.\n' "${RUN_DIR}" >&2
  exit 2
fi
if [[ "${RUN_BASELINE}" != "1" && "${BASELINE_ONLY}" == "1" ]]; then
  printf -- '--baseline-only cannot be combined with --skip-baseline.\n' >&2
  exit 2
fi
if [[ -n "${EVAL_VARIANT}" && "${EVAL_VARIANT}" != *=* ]]; then
  printf 'Invalid --variant; expected NAME=CHECKPOINT_PATH[:WEIGHT], got %s\n' "${EVAL_VARIANT}" >&2
  exit 2
fi
if [[ -n "${EVAL_VARIANT}" ]]; then
  VARIANT_NAME="${EVAL_VARIANT%%=*}"
  if [[ ! "${VARIANT_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || "${VARIANT_NAME}" == "baseline" ]]; then
    printf 'Invalid variant name %s; use letters, digits, dot, underscore, or hyphen, and do not use baseline.\n' "${VARIANT_NAME}" >&2
    exit 2
  fi
fi

if [[ "${SKIP_GENERATION}" != "1" ]]; then
  while IFS= read -r variant; do
    generate_variant "${variant}" "$(variant_lora_path "${variant}")" "$(variant_lora_weight "${variant}")"
  done < <(variant_names)
fi

if [[ "${SKIP_SCORE}" != "1" ]]; then
  while IFS= read -r variant; do
    assert_variant_video_count "${variant}" "${GEN_DIR}/${variant}"
    score_variant "${variant}"
  done < <(variant_names)
  while IFS= read -r variant; do
    if [[ "${variant}" == "baseline" ]]; then
      continue
    fi
    if [[ -f "${SCORE_DIR}/da3/baseline_scores.csv" && -f "${SCORE_DIR}/da3/${variant}_scores.csv" ]]; then
      "${PY_CMD[@]}" "${VGM_REPO_ROOT}/scripts/summarize_wan22_compare_scores.py" \
        --baseline_csv "${SCORE_DIR}/da3/baseline_scores.csv" \
        --lora_csv "${SCORE_DIR}/da3/${variant}_scores.csv" \
        --output_csv "${SCORE_DIR}/da3/${variant}_vs_baseline_summary.csv" \
        --output_md "${SCORE_DIR}/da3/${variant}_vs_baseline_summary.md"
    fi
  done < <(variant_names)
fi

printf '[run_eval] complete: %s\n' "${EVAL_DIR}"
