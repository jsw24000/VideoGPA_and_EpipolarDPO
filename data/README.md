# DL3DV Condition Data Pack

This directory stores only portable metadata, scripts, configs, schemas, logs, manifests, and reports. Large DL3DV scene archives, extracted images, videos, and first frames must live outside the project tree under an asset root such as `/data1/3DVGM_data`.

## Official Protocol

The official split is:

- Train: `8K`, `9K`, `10K`, `11K`
- Test: `1K`

Natural language captions come only from the local VideoGPA files:

- `VideoGPA/dl3dv_video_captions/captions_8K.json`
- `VideoGPA/dl3dv_video_captions/captions_9K.json`
- `VideoGPA/dl3dv_video_captions/captions_10K.json`
- `VideoGPA/dl3dv_video_captions/captions_11K.json`
- `VideoGPA/dl3dv_video_captions/captions_1K.json`

No extra VLM is required. The stored CogVLM2-Video captions are the only natural scene descriptions used here.

## T2V and I2V Conditions

T2V train and test use the natural VLM caption exactly, except for stripping leading and trailing whitespace. T2V prompts never append scripted camera motion and never use `transforms.json`.

I2V train uses first frame plus the official static-scene scripted camera prompt imported from `VideoGPA/data_prep/generate_i2v_prompts.py`. The scene-local random generator is seeded from `2026 + stable scene hash`, so reruns and migrations reproduce the same camera motion.

I2V test uses first frame plus the same natural 1K VLM caption used by T2V test. Test I2V does not use the scripted camera prompt, although the master manifest stores a scripted backup field for auditing.

Caption keys use `images_8`. The downloader first looks for that directory in each HF zip. If the actual 960P zip exposes exactly one `images_*` directory, it can use that unique directory and records `requested_image_dir`, `resolved_image_dir`, and `image_dir_fallback_used` in `first_frames.jsonl`.

## Prepared Data Status

Current asset root:

```text
/home/data1/3dsm/3DVGM_data
```

The condition pack is complete and validated:

- Validation status: `pass`
- Errors: `0`
- Warnings: `0`
- Missing first frames: `0`
- Total scenes: `4147`
- Train scenes: `3147`
- Test scenes: `1000`

Per-subset coverage:

| subset | split | scenes | first frames | raw zip cache |
| --- | --- | ---: | ---: | ---: |
| `1K` | test | 1000 | 1000 | 1000 |
| `8K` | train | 840 | 840 | 840 |
| `9K` | train | 900 | 900 | 900 |
| `10K` | train | 909 | 909 | 909 |
| `11K` | train | 498 | 498 | 498 |

Large files are stored outside git. In the current local workspace the first frames have been copied under project `data/`, but `data/first_frames/` is ignored by git because it is several GiB:

- First frames for local training: `/home/3dsm/Desktop/3DVGM/data/first_frames/`
- Original external first-frame copy: `/home/data1/3dsm/3DVGM_data/first_frames/`
- Preserved original DL3DV zip cache: `/home/data1/3dsm/3DVGM_data/download_cache/`
- Temporary staging: `/home/data1/3dsm/3DVGM_data/staging/`

Project-side portable metadata is stored here:

- Caption index: `data/manifests/caption_index.jsonl`
- First-frame index: `data/manifests/first_frames.jsonl`
- Canonical all records: `data/manifests/master_all.jsonl`
- Canonical train records: `data/manifests/master_train.jsonl`
- Canonical test records: `data/manifests/master_test.jsonl`
- VideoGPA JSON exports: `data/manifests/videogpa_protocol/`
- Shared JSONL exports for downstream adapters: `data/manifests/shared_protocol/`
- Validation report: `data/reports/final_validation.json`
- Human-readable summary: `data/reports/dataset_summary.md`

Prompt/export counts:

| output | records | condition |
| --- | ---: | --- |
| `train_i2v.json` | 3147 | first frame + official scripted static-scene camera prompt |
| `train_t2v.json` | 3147 | natural VideoGPA VLM caption only |
| `test_i2v.json` | 1000 | first frame + natural 1K VideoGPA VLM caption |
| `test_t2v.json` | 1000 | natural 1K VideoGPA VLM caption only |

This matches the local VideoGPA protocol used for raw condition construction:

- T2V train/test prompts are the natural CogVLM2-Video captions from `VideoGPA/dl3dv_video_captions/`.
- T2V prompts do not append scripted camera motion.
- I2V train prompts are generated from `VideoGPA/data_prep/generate_i2v_prompts.py`, reusing the official static-scene prefix, motion primitives, 2/3-piece random composition, and `then` / `followed by` connector rules.
- I2V test prompts use the same natural `1K` caption as T2V test, not the scripted camera prompt.
- Canonical manifests store first-frame paths relative to the asset root, so the pack can be moved to another server by changing `asset_root`.

This pack contains only raw condition data. Candidate videos, preference scores, winner/loser pairs, and LoRA training artifacts are not included.

## Storage

Project `data/` contains small files only:

- `configs/`
- `scripts/`
- `schemas/`
- `manifests/`
- `reports/`
- `logs/`
- `tests/`

External asset root contains large or derived assets:

- `dl3dv_raw_960p/`
- `first_frames/`
- `download_cache/`
- `staging/`

Run storage resolution first:

```bash
python data/scripts/resolve_storage.py --scratch-root /data1/3DVGM_data
```

On a new server, regenerate `data/configs/storage.local.yaml` or pass the new `--asset-root` when exporting and validating. Canonical manifests only store paths relative to `asset_root`.

## Commands

Small smoke run:

```bash
python data/scripts/build_all_conditions.py --scratch-root /data1/3DVGM_data --splits 1K --limit 3 --seed 2026 --resume
```

Validate that smoke scope explicitly:

```bash
python data/scripts/validate_condition_pack.py --asset-root /data1/3DVGM_data --splits 1K --limit 3
```

Full run:

```bash
python data/scripts/build_all_conditions.py --scratch-root /data1/3DVGM_data --seed 2026 --resume
```

Validate an existing pack:

```bash
python data/scripts/validate_condition_pack.py --asset-root /data1/3DVGM_data
```

Export prompts after moving first frames to a new server:

```bash
python data/scripts/export_prompt_jsons.py --asset-root /new/asset/root/3DVGM_data
```

Dry-run cleanup:

```bash
python data/scripts/cleanup_raw_data.py --asset-root /data1/3DVGM_data
```

Real cleanup:

```bash
python data/scripts/cleanup_raw_data.py --asset-root /data1/3DVGM_data --confirm-cleanup
```

## Downstream Use

VideoGPA and Epipolar-DPO should both read the same canonical condition manifests in `data/manifests/`. Candidate videos, preference scores, winner/loser pairs, and LoRA training data are intentionally outside this raw condition-data task.

## Git Layout Notes

The root `3DVGM` repository records `VideoGPA/`, `Epipolar-DPO/`, and `third_party/Wan2.2/` as submodules. This avoids mixing nested Git histories into the root repository and keeps large upstream projects reproducible by commit pointer.

Large local artifacts are intentionally ignored by git:

- `data/first_frames/`
- `models/`
- `outputs/`
- checkpoint and weight files such as `*.safetensors`, `*.pth`, `*.pt`, `*.ckpt`
- host-local `data/configs/storage.local.yaml`

Local compatibility edits currently present inside the `VideoGPA/` working tree are also saved as a portable patch:

```bash
git -C VideoGPA apply ../patches/VideoGPA-local-changes.patch
```
