#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '\n[setup_wan22_env] %s\n' "$*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env/require_profile.sh"
vgm_require_profile
ENV_NAME="${ENV_NAME:-wan22_videogpa}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
OUTPUT_DIR="${OUTPUT_DIR:-${VGM_OUTPUT_ROOT}/wan22_smoke}"
WAN_SRC_DIR="${WAN_SRC_DIR:-${VGM_REPO_ROOT}/third_party/Wan2.2}"
WAN_REPO_URL="${WAN_REPO_URL:-https://github.com/Wan-Video/Wan2.2.git}"

TORCH_VERSION="${TORCH_VERSION:-2.8.0+cu128}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0+cu128}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.8.0+cu128}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

NUMPY_VERSION="${NUMPY_VERSION:-1.26.4}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.51.3}"
DIFFUSERS_VERSION="${DIFFUSERS_VERSION:-0.33.1}"
ACCELERATE_VERSION="${ACCELERATE_VERSION:-1.6.0}"
PEFT_VERSION="${PEFT_VERSION:-0.15.2}"
SAFETENSORS_VERSION="${SAFETENSORS_VERSION:-0.5.3}"
HUGGINGFACE_HUB_VERSION="${HUGGINGFACE_HUB_VERSION:-0.34.4}"
SENTENCEPIECE_VERSION="${SENTENCEPIECE_VERSION:-0.2.0}"
DECORD_VERSION="${DECORD_VERSION:-0.6.0}"
LIBROSA_VERSION="${LIBROSA_VERSION:-0.10.2.post1}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3.post1}"
OPENCV_HEADLESS_VERSION="${OPENCV_HEADLESS_VERSION:-4.10.0.84}"
LPIPS_VERSION="${LPIPS_VERSION:-0.1.4}"

mkdir -p "${OUTPUT_DIR}"

wan_source_looks_complete() {
  [ -f "${WAN_SRC_DIR}/generate.py" ] &&
    [ -f "${WAN_SRC_DIR}/requirements.txt" ] &&
    [ -f "${WAN_SRC_DIR}/wan/textimage2video.py" ] &&
    [ -f "${WAN_SRC_DIR}/wan/configs/wan_ti2v_5B.py" ]
}

command -v conda >/dev/null 2>&1 || {
  printf 'conda was not found on PATH.\n' >&2
  exit 1
}

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

SYSTEM_INFO_FILE="${OUTPUT_DIR}/system_info.txt"
log "Collecting hardware and toolchain info at ${SYSTEM_INFO_FILE}"
{
  printf 'Collected at: %s\n\n' "$(date -Is)"

  printf '## nvidia-smi\n'
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    printf 'nvidia-smi not found on PATH\n'
  fi
  printf '\n'

  printf '## nvcc --version\n'
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
  else
    printf 'nvcc not found on PATH\n'
  fi
  printf '\n'

  printf '## gcc --version\n'
  if command -v gcc >/dev/null 2>&1; then
    gcc --version
  else
    printf 'gcc not found on PATH\n'
  fi
  printf '\n'

  printf '## conda --version\n'
  conda --version
  printf '\n'

  printf '## python --version\n'
  python --version 2>&1
  printf '\n'

  printf '## uname -a\n'
  uname -a
} > "${SYSTEM_INFO_FILE}"

if [ -d "${WAN_SRC_DIR}/.git" ]; then
  log "Reusing existing Wan2.2 source at ${WAN_SRC_DIR}"
elif wan_source_looks_complete; then
  log "Reusing existing non-git Wan2.2 source at ${WAN_SRC_DIR}"
elif [ -e "${WAN_SRC_DIR}" ]; then
  printf 'Wan2.2 source directory exists but does not look complete: %s\n' "${WAN_SRC_DIR}" >&2
  printf 'Expected generate.py, requirements.txt, wan/textimage2video.py, and wan/configs/wan_ti2v_5B.py.\n' >&2
  printf 'Move the incomplete directory aside, or set WAN_SRC_DIR to a valid Wan2.2 source checkout.\n' >&2
  exit 1
else
  log "Cloning official Wan2.2 source into ${WAN_SRC_DIR}"
  mkdir -p "$(dirname "${WAN_SRC_DIR}")"
  git clone "${WAN_REPO_URL}" "${WAN_SRC_DIR}"
fi

if [ -d "${VGM_REPO_ROOT}/VideoGPA" ]; then
  if [ -e "${VGM_REPO_ROOT}/VideoGPA/Wan2.2" ] && [ ! -L "${VGM_REPO_ROOT}/VideoGPA/Wan2.2" ]; then
    log "VideoGPA/Wan2.2 exists and is not a symlink; leaving it untouched."
  else
    log "Ensuring VideoGPA/Wan2.2 points to ../third_party/Wan2.2"
    ln -sfn ../third_party/Wan2.2 "${VGM_REPO_ROOT}/VideoGPA/Wan2.2"
  fi
fi

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  log "Reusing existing Conda env: ${ENV_NAME}"
else
  log "Creating Conda env ${ENV_NAME} with Python ${PYTHON_VERSION}"
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

PIP=(conda run -n "${ENV_NAME}" python -m pip)
PY=(conda run -n "${ENV_NAME}" python)

log "Upgrading build helpers"
"${PIP[@]}" install --upgrade pip setuptools wheel packaging ninja

log "Installing PyTorch CUDA 12.8 wheels"
"${PIP[@]}" install \
  --index-url "${PYTORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}"

PINNED_REQ="${OUTPUT_DIR}/requirements.wan22.pinned.txt"
FILTERED_REQ="${OUTPUT_DIR}/requirements.wan22.filtered.txt"

cat > "${PINNED_REQ}" <<EOF
numpy==${NUMPY_VERSION}
transformers==${TRANSFORMERS_VERSION}
diffusers==${DIFFUSERS_VERSION}
accelerate==${ACCELERATE_VERSION}
peft==${PEFT_VERSION}
safetensors==${SAFETENSORS_VERSION}
huggingface-hub==${HUGGINGFACE_HUB_VERSION}
sentencepiece==${SENTENCEPIECE_VERSION}
decord==${DECORD_VERSION}
librosa==${LIBROSA_VERSION}
opencv-python-headless==${OPENCV_HEADLESS_VERSION}
lpips==${LPIPS_VERSION}
EOF

log "Filtering Wan2.2 requirements to avoid reinstalling torch/flash-attn/opencv-python"
awk '
  /^[[:space:]]*($|#)/ { next }
  {
    line=$0
    key=tolower(line)
    sub(/^[[:space:]]*/, "", key)
    if (key ~ /^(torch|torchvision|torchaudio|flash[_-]?attn|opencv-python|opencv-contrib-python)([[:space:]<>=!~].*)?$/) next
    print line
  }
' "${WAN_SRC_DIR}/requirements.txt" > "${FILTERED_REQ}"

log "Removing GUI OpenCV wheels if present"
"${PIP[@]}" uninstall -y opencv-python opencv-contrib-python >/dev/null

log "Installing pinned WAN2.2 and VideoGPA LoRA foundation packages"
"${PIP[@]}" install -r "${PINNED_REQ}"

log "Installing filtered official Wan2.2 requirements"
"${PIP[@]}" install -r "${FILTERED_REQ}"

if [ "${SKIP_FLASH_ATTN:-0}" = "1" ]; then
  log "SKIP_FLASH_ATTN=1, leaving flash-attn uninstalled."
else
  log "Installing flash-attn last with --no-build-isolation"
  MAX_JOBS="${MAX_JOBS:-8}" "${PIP[@]}" install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation
fi

log "Quick import probe"
"${PY[@]}" - <<'PY'
import importlib
mods = ["torch", "torchvision", "transformers", "diffusers", "peft", "accelerate", "safetensors", "huggingface_hub", "flash_attn", "numpy"]
for mod in mods:
    importlib.import_module(mod)
print("imports_ok")
PY

log "Done. Activate with: conda activate ${ENV_NAME}"
