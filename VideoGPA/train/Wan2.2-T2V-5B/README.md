# WAN2.2-T2V-5B VideoGPA Smoke Chain

This directory is a low-intrusion pure T2V sibling of `VideoGPA/train/Wan2.2-TI2V-5B`.

It uses the same `Wan2.2-TI2V-5B` base checkpoint, but enters the model's text-only branch and trains VideoGPA DPO LoRA without a first-frame condition. Prompts come from VLM natural captions in `data/manifests/videogpa_protocol/train_t2v.json`; first frames and `image_latent` are not used.

## What Changed From TI2V

- Removed `image_path` / `image_prompt` reads.
- Removed PIL first-frame loading.
- Removed VAE image-latent encoding.
- Removed `image_latent` from condition files.
- Removed clean first temporal latent replacement during training.
- Removed TI2V mask/timestep construction for the first latent frame.

## What Stayed The Same

- Same WAN T5 text encoder and WAN VAE latent layout.
- Same `WanModel` base checkpoint.
- Same flow-matching target.
- Same frozen reference model and policy/reference DPO comparison.
- Same `VideoGPA/train/loss.py` DPO loss.
- Same WAN LoRA target modules `q/k/v/o`, rank `64`, alpha `128.0`.
- Same AdamW and cosine warmup scheduler family.
- Same VGGT consistency scorer semantics.

## Smoke Versus Formal Training

The smoke config uses 4 train 8K prompts, 3 candidate seeds per prompt, and 5 optimizer steps. It is for plumbing and checkpoint validation only. It is not comparable to paper-scale training.

Formal training should keep `test_t2v.json` out of the pipeline, point to the full train T2V manifest, increase generation/training scale, and write to a separate formal output root such as `outputs/videogpa/wan2.2-5b/t2v/formal`.

The smoke chain first scores candidates with official thresholds and debug fallback disabled. If fewer than 2 preference pairs are available, the harness extends only the 8K train subset to 8 prompts, regenerates missing candidates, and only then permits fallback pairs marked `DEBUG_ONLY_NOT_COMPARABLE`.

## Formal 8K-11K Expansion

For formal VideoGPA T2V training, create a separate formal YAML rather than editing the smoke config in place. Keep:

- `paths.train_manifest: data/manifests/videogpa_protocol/train_t2v.json`
- `project.task: t2v`
- no first-frame or image fields
- `scoring.smoke_fallback_if_no_pairs: false`
- output root under `outputs/videogpa/wan2.2-5b/t2v/formal`

To expand beyond the current 8K smoke bucket, change the subset/export stage in a formal harness to include the intended train buckets `8K`, `9K`, `10K`, and `11K`. Do not include `1K`, `test_t2v.json`, or `test_i2v.json`.

This repo now includes a full formal config and harness:

```bash
GPU_ID=1 VIDEOGPA_CONDA_ENV=wan22_videogpa \
bash scripts/videogpa/wan22_5b_t2v/run_formal.sh \
  --config configs/videogpa/wan22_5b_t2v_formal.yaml
```

The resulting LoRA adapters are saved under:

```text
outputs/videogpa/wan2.2-5b/t2v/formal/<run_id>/checkpoints/step_*/
```

## Portable Paths

Use environment variables instead of editing source:

- `PROJECT_ROOT`
- `WAN22_5B_MODEL_PATH`
- `VGGT_MODEL_PATH`
- `VIDEOGPA_ROOT`
- `VIDEOGPA_OUTPUT_ROOT`
- `HF_HOME`

The default smoke config is `configs/videogpa/wan22_5b_t2v_smoke.yaml`.

Run the smoke chain from the project root:

```bash
bash scripts/videogpa/wan22_5b_t2v/run_smoke.sh
```

On this project the runnable environment is expected to be:

```bash
VIDEOGPA_CONDA_ENV=wan22_videogpa bash scripts/videogpa/wan22_5b_t2v/run_smoke.sh
```

Run only preflight/static checks:

```bash
VIDEOGPA_CONDA_ENV=wan22_videogpa \
bash scripts/videogpa/wan22_5b_t2v/run_smoke.sh --stop-after static_checks
```
