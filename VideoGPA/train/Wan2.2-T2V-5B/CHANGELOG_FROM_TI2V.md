# Changelog From WAN2.2-TI2V-5B

## Reference

- Original generation reference: `VideoGPA/generate/Wan2.2-TI2V-5B.py`
- Original encoder reference: `VideoGPA/train/Wan2.2-TI2V-5B/02_encode.py`
- Original trainer reference: `VideoGPA/train/Wan2.2-TI2V-5B/03_train.py`
- WAN modality branch reference: `VideoGPA/Wan2.2/wan/textimage2video.py`

## New Files

- `VideoGPA/generate/Wan2.2-T2V-5B.py`
- `VideoGPA/train/Wan2.2-T2V-5B/02_encode.py`
- `VideoGPA/train/Wan2.2-T2V-5B/03_train.py`

## Generation

- Original TI2V logic read `image_path`, opened a PIL image, and called `engine.generate(..., img=image, ...)`.
- T2V logic rejects image fields and calls `engine.generate(..., img=None, ...)`.
- Reason: local WAN `WanTI2V.generate()` dispatches to `t2v()` only when `img is None`.
- Method impact: only conditioning modality changes. Base checkpoint, sampler, prompt encoding, seed handling, and LoRA loading semantics remain WAN-native.
- Smoke wrapper additions: deterministic multi-seed candidate grouping, ffprobe frame validation, invalid-video quarantine, run-local manifests, and portable model-path resolution. These do not change the VideoGPA method.

## Encoding

- Original TI2V logic saved condition files with `encoder_hidden_states` and `image_latent`.
- T2V logic saves condition files with `encoder_hidden_states` only.
- Original TI2V logic loaded and resized first-frame images.
- T2V logic never reads first-frame image paths.
- Winner and loser video latents are still encoded through WAN VAE with the same normalized `[3, F, H, W]` video tensor format.
- Method impact: only image conditioning is removed.
- Compatibility addition: the encoded manifest keeps the official `groups -> videos -> latent_path/condition_path/consistency_score/motion_norm` shape required by `VideoGPA/train/dataset.py`.

## Training

- Original TI2V logic replaced the first temporal noisy latent with `image_latent`.
- T2V logic does not replace any temporal latent.
- Original TI2V logic built a mask-derived timestep tensor with a clean first latent frame.
- T2V logic passes a sampled 1D timestep tensor to `WanModel.forward()`, which expands it across the full sequence.
- Winner and loser still share the same timestep, noise realization, and text condition.
- The DPO loss, frozen reference model, LoRA target modules, rank, alpha, optimizer, scheduler family, and flow-matching target remain aligned with the original WAN VideoGPA script.
- Method impact: conditioning modality changes; preference optimization definition remains DPO.
- Checkpoint addition: each smoke checkpoint stores PEFT adapter files, optimizer/scheduler state, trainer state, and `resolved_config.yaml` for reload/reproduction.

## Smoke-Orchestration Difference

- The official WAN script uses PyTorch Lightning and W&B orchestration.
- The local runnable env lacks `pytorch_lightning` and `wandb`, so the T2V smoke trainer uses a direct PyTorch loop.
- This affects smoke orchestration only, not DPO math, model calls, LoRA adapter format, or checkpoint reload validation.
- Safety addition: `run_smoke.sh` writes stage DONE markers, `run_state.json`, fixed logs under `logs/`, and `reproduce.sh`; it does not delete user data or write generated artifacts outside `outputs/videogpa/...`.
