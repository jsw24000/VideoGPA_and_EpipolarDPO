# WAN2.2 TI2V-5B Environment Setup

This project keeps WAN2.2 source code and model weights separate.

- Conda env: `wan22_videogpa`
- Python: `3.10`
- WAN2.2 source: `third_party/Wan2.2`
- WAN2.2 model weights: `models/wan/Wan2.2-TI2V-5B`
- VideoGPA WAN LoRA path: `VideoGPA/checkpoints/VideoGPA-Wan2.2TI2V-lora`
- VideoGPA WAN source link: `VideoGPA/Wan2.2 -> ../third_party/Wan2.2`

## Version Choices

The machine has NVIDIA driver `570.211.01`, system CUDA toolkit `12.8`, GCC `11.4.0`, and two RTX PRO 6000 Blackwell GPUs with about 96 GiB each. The setup uses PyTorch CUDA 12.8 wheels instead of matching the system toolkit exactly.

Pinned foundation versions:

- `torch==2.8.0+cu128`
- `torchvision==0.23.0+cu128`
- `torchaudio==2.8.0+cu128`
- `numpy==1.26.4`
- `transformers==4.51.3`
- `diffusers==0.33.1`
- `accelerate==1.6.0`
- `peft==0.15.2`
- `safetensors==0.5.3`
- `huggingface-hub==0.34.4`
- `sentencepiece==0.2.0`
- `decord==0.6.0`
- `librosa==0.10.2.post1`
- `flash-attn==2.8.3.post1`

`transformers==4.51.3` follows the official Wan2.2 requirement upper bound. `numpy` is held to 1.x to avoid NumPy 2.x compatibility issues. `flash-attn` is installed last with `--no-build-isolation` after PyTorch is present.

## Setup

```bash
bash scripts/setup_wan22_env.sh
conda activate wan22_videogpa
python scripts/check_wan22_env.py
```

The setup script refreshes `outputs/wan22_smoke/system_info.txt` and creates a filtered copy of the official WAN requirements in `outputs/wan22_smoke/requirements.wan22.filtered.txt`, excluding `torch`, `torchvision`, `torchaudio`, and `flash_attn` so the selected CUDA wheel is not overwritten.

## Smoke Test

Default smoke command:

```bash
bash scripts/run_wan22_i2v_smoke.sh
```

Useful overrides:

```bash
GPU_ID=1 bash scripts/run_wan22_i2v_smoke.sh
OFFLOAD_MODEL=True T5_CPU=1 bash scripts/run_wan22_i2v_smoke.sh
FRAME_NUM=17 SAMPLE_STEPS=6 bash scripts/run_wan22_i2v_smoke.sh
```

The smoke test uses the official `third_party/Wan2.2/generate.py` entrypoint with task `ti2v-5B`, model path `models/wan/Wan2.2-TI2V-5B`, official resolution `1280*704`, an example I2V input image, 17 frames, and 6 sampling steps. This is a functional smoke test, not a baseline-quality generation run.

Outputs are written under:

- `outputs/wan22_smoke/system_info.txt`
- `outputs/wan22_smoke/environment_versions.txt`
- `outputs/wan22_smoke/command_used.txt`
- `outputs/wan22_smoke/run.log`
- `outputs/wan22_smoke/input/<timestamp>/`
- `outputs/wan22_smoke/generated/<timestamp>/`
- `outputs/wan22_smoke/runs/<timestamp>/gpu_memory_summary.txt`

## Current Result

Smoke status: PASS on 2026-07-25 13:50 CST using the official `third_party/Wan2.2/generate.py` TI2V entrypoint.

Peak GPU memory: `53205 MiB` / `51.96 GiB` on GPU 0.

Generated video: `outputs/wan22_smoke/generated/20260725_134918/wan22_ti2v_i2v_smoke.mp4`

The smoke command used `--task ti2v-5B`, `--size 1280*704`, `--frame_num 17`, `--sample_steps 6`, `--sample_solver unipc`, `--base_seed 42`, and `--convert_model_dtype`. `ffprobe` reads the output as H.264, 17 frames, 24 fps, `800x1088`, 2.86 MB. The log finished cleanly with no CUDA, shape, dtype, or checkpoint errors, and no VideoGPA LoRA was loaded.

## Common Issues

- `flash_attn` import fails: rerun setup after confirming PyTorch imports successfully, `nvcc --version` works, GCC is compatible, and enough RAM is available. Do not modify system CUDA to fix this.
- CUDA unavailable: confirm `nvidia-smi` works, activate `wan22_videogpa`, and check that PyTorch reports `torch.version.cuda == 12.8`.
- bf16 unsupported: use a GPU with bf16 support. The current RTX PRO 6000 Blackwell GPUs support bf16.
- T5 causes memory pressure: rerun with `OFFLOAD_MODEL=True T5_CPU=1`.
- Model path error: verify `models/wan/Wan2.2-TI2V-5B` contains the T5 weight, VAE weight, config files, and safetensors shards.
- Safetensors shard incomplete: verify all `diffusion_pytorch_model-*.safetensors` files and `diffusion_pytorch_model.safetensors.index.json` are present.

## Later VideoGPA LoRA Integration

Do not load or merge VideoGPA LoRA during this smoke test. The environment includes `peft`, `diffusers`, `transformers`, `accelerate`, and `safetensors`, so the next step can use `VideoGPA/checkpoints/VideoGPA-Wan2.2TI2V-lora` with the same WAN2.2 source tree. Baseline and LoRA runs should use the same prompt, input image, seed, frame count, resolution, solver, sampling steps, shift, and guidance scale.
