# WAN2.2 5B T2V VideoGPA Smoke Scripts

This is the runnable smoke harness for the pure T2V sibling of the official
VideoGPA `Wan2.2-TI2V-5B` chain. Activate a path profile first; the smoke YAML
then resolves `experiment.output_subdir` under `VGM_OUTPUT_ROOT`:

```text
$VGM_OUTPUT_ROOT/videogpa/wan22_5b_t2v/smoke/<run_id>/
```

It never writes generated videos, latents, or checkpoints into `VideoGPA/data`,
`VideoGPA/checkpoints`, project `data`, or `models`.

## Quick Checks

Run only preflight and static checks:

```bash
source scripts/env/activate_profile.sh local
VIDEOGPA_CONDA_ENV=wan22_videogpa \
bash scripts/videogpa/wan22_5b_t2v/run_smoke.sh --stop-after static_checks
```

Run only preflight:

```bash
source scripts/env/activate_profile.sh local
VIDEOGPA_CONDA_ENV=wan22_videogpa \
bash scripts/videogpa/wan22_5b_t2v/run_smoke.sh --stop-after preflight
```

## Full Smoke

Run from any working directory after activating a profile:

```bash
source scripts/env/activate_profile.sh local
VIDEOGPA_CONDA_ENV=wan22_videogpa bash scripts/videogpa/wan22_5b_t2v/run_smoke.sh
```

Useful options:

```bash
bash scripts/videogpa/wan22_5b_t2v/run_smoke.sh --resume --run-id <run_id>
bash scripts/videogpa/wan22_5b_t2v/run_smoke.sh --stop-after preflight
bash scripts/videogpa/wan22_5b_t2v/run_smoke.sh --force-stage generation_candidates --run-id <run_id>
```

Important path environment variables come from the active profile:

```bash
VGM_REPO_ROOT=<repo>
VGM_DL3DV_ROOT=<dl3dv root>
VGM_MODEL_ROOT=<model root>
VGM_OUTPUT_ROOT=<output root>
HF_HOME=/path/to/hf/cache
```

## Stage Notes

- `preflight`: writes `preflight/preflight_report.md`, `reports/official_diff.md`, and `config/environment.txt`.
- `static_checks`: runs `py_compile`, `--help` checks, shellcheck if present, and path checks.
- `subset`: deterministically samples 4 unique 8K train scenes from `train_t2v.json`.
- `generation_micro`: one prompt, one seed, same text-only branch, for plumbing.
- `generation_candidates`: 4 prompts x 3 seeds, base model only.
- `scoring`: first applies official thresholds with debug fallback disabled. If fewer than 2 pairs remain, it extends only the 8K train subset to `data.fallback_subset_size` (8 by default), regenerates missing candidates, then allows `DEBUG_ONLY_NOT_COMPARABLE` fallback pairs if still needed.
- `encoding`: writes text-only conditions and winner/loser latents, with no `image_latent`.
- `training`: runs 5-step DPO LoRA smoke and validates adapter reload.
- `comparison`: generates base/LoRA videos with the same prompt and seed.

For strict scoring without debug fallback:

```bash
RUN_DIR=$VGM_OUTPUT_ROOT/videogpa/wan22_5b_t2v/smoke/<run_id> \
DISABLE_DEBUG_FALLBACK=1 \
bash scripts/videogpa/wan22_5b_t2v/03_score_preferences.sh
```

## Formal Reproduction

Formal config:

```text
configs/videogpa/wan22_5b_t2v_formal.yaml
```

Formal output root:

```text
$VGM_OUTPUT_ROOT/videogpa/wan22_5b_t2v/formal/<run_id>/
```

The formal selector reads only:

```text
$VGM_MANIFEST_ROOT/videogpa_protocol/train_t2v.json
$VGM_MANIFEST_ROOT/master_all.jsonl
$VGM_MANIFEST_ROOT/caption_index.jsonl
```

It selects all train T2V scenes from buckets `8K`, `9K`, `10K`, and `11K` by default. It does not read `test_t2v.json`, `test_i2v.json`, `train_i2v.json`, or first-frame files.

Run only formal preflight/static checks:

```bash
source scripts/env/activate_profile.sh local
VIDEOGPA_CONDA_ENV=wan22_videogpa \
bash scripts/videogpa/wan22_5b_t2v/run_formal.sh --stop-after static_checks
```

Run formal through candidate generation only:

```bash
source scripts/env/activate_profile.sh local
GPU_ID=1 \
VIDEOGPA_CONDA_ENV=wan22_videogpa \
bash scripts/videogpa/wan22_5b_t2v/run_formal.sh --stop-after generation_candidates
```

Run full formal chain:

```bash
source scripts/env/activate_profile.sh local
GPU_ID=1 \
VIDEOGPA_CONDA_ENV=wan22_videogpa \
bash scripts/videogpa/wan22_5b_t2v/run_formal.sh
```

Resume a formal run:

```bash
source scripts/env/activate_profile.sh local
GPU_ID=1 \
VIDEOGPA_CONDA_ENV=wan22_videogpa \
bash scripts/videogpa/wan22_5b_t2v/run_formal.sh --resume --run-id <run_id>
```

Formal stages are `preflight`, `static_checks`, `subset`, `generation_candidates`, `scoring`, `encoding`, and `training`. Unlike smoke, formal skips `generation_micro`, skips base/LoRA comparison generation, and disables debug fallback pairs.

Expected scale with the default formal config:

```text
3147 train prompts x 3 seeds = 9441 candidate videos
81 frames, 1280x704, 50 sampling steps per video
```

This is intentionally expensive. The default formal harness is single-GPU sequential; shard generation/scoring/encoding manually for practical wall-clock time.

Environment requirements:

```text
Python 3.10
torch / torchvision with CUDA
diffusers
transformers 4.51.3-compatible dependency set
peft
accelerate
decord
lpips
easydict
ffmpeg and ffprobe on PATH
Wan2.2-TI2V-5B weights under VGM_MODEL_ROOT
VGGT-1B weights under VGM_MODEL_ROOT
```

With the current `transformers==4.51.3`, `huggingface-hub` must satisfy:

```text
huggingface-hub>=0.30.0,<1.0
```

If preflight/static checks fail with `huggingface-hub==1.x`, repair the environment before running generation:

```bash
conda run -n wan22_videogpa python -m pip install 'huggingface-hub>=0.30.0,<1.0'
```

LoRA checkpoints are written under:

```text
$VGM_OUTPUT_ROOT/videogpa/wan22_5b_t2v/formal/<run_id>/checkpoints/step_001000/
$VGM_OUTPUT_ROOT/videogpa/wan22_5b_t2v/formal/<run_id>/checkpoints/step_002000/
...
$VGM_OUTPUT_ROOT/videogpa/wan22_5b_t2v/formal/<run_id>/checkpoints/step_010000/
```

Each run contains top-level `config_resolved.yaml`, `command.txt`, `environment.txt`, `git_state.txt`, and checkpoint artifacts under `checkpoints/`.
