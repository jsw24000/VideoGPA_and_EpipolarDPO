# Path Migration Audit

Generated during the local/cluster path profile migration.

## Runtime Files Untracked

The following tracked runtime areas contained local absolute paths or generated
metadata and were removed from Git index without deleting local files:

- `data/manifests/**`
- `data/logs/**`
- `data/reports/*.json`
- `data/reports/*.jsonl`
- `data/reports/*.md`

## Repo-Owned Code And Config

| File:Line | Path Meaning | Root | Ownership | Action |
| --- | --- | --- | --- | --- |
| `configs/videogpa/wan22_5b_t2v_smoke.yaml:3` | output subdir | `VGM_OUTPUT_ROOT` | repo config | Replaced old `paths.output_root` with `experiment.output_subdir`. |
| `configs/videogpa/wan22_5b_t2v_smoke.yaml:17` | WAN model | `VGM_MODEL_ROOT` | repo config | Replaced model path/auto with `model.model_relpath`. |
| `configs/videogpa/wan22_5b_t2v_smoke.yaml:21` | train manifest | `VGM_DL3DV_ROOT` | repo config | Replaced `data/manifests/...` with `data.manifest_relpath`. |
| `configs/videogpa/wan22_5b_t2v_smoke.yaml:23` | first-frame root | `VGM_DL3DV_ROOT` | repo config | Added `data.first_frames_relroot`. |
| `configs/videogpa/wan22_5b_t2v_formal.yaml:3` | output subdir | `VGM_OUTPUT_ROOT` | repo config | Same relative schema for formal. |
| `configs/videogpa/wan22_5b_t2v_formal.yaml:17` | WAN/VGGT models | `VGM_MODEL_ROOT` | repo config | Same relative schema for formal. |
| `configs/videogpa/wan22_5b_t2v_formal.yaml:21` | train manifest | `VGM_DL3DV_ROOT` | repo config | Same relative schema for formal. |
| `scripts/env/activate_profile.sh:28` | stale profile vars | profile env | repo wrapper | Unsets old profile variables before sourcing the selected profile. |
| `scripts/env/activate_profile.sh:43` | derived DL3DV roots | `VGM_DL3DV_ROOT` | repo wrapper | Exports archives/extracted/manifests/first_frames/validation roots. |
| `scripts/env/check_paths.py:48` | root validation | all `VGM_*` roots | repo wrapper | Prints resolved roots and validates local/cluster profile shape. |
| `vgm_common/paths.py:141` | relative path safety | root-specific | repo module | Rejects committed absolute paths and `../` escape from configured roots. |
| `vgm_common/config.py:99` | YAML resolving | model/data/output roots | repo module | Resolves relative YAML fields into absolute runtime paths. |
| `vgm_common/config.py:169` | resolved config output | run dir | repo module | Writes `config_resolved.yaml` under outputs only. |
| `scripts/check_wan22_env.py:14` | WAN smoke env report | `VGM_OUTPUT_ROOT` | repo wrapper | Writes environment report under profile output root. |
| `scripts/check_wan22_env.py:15` | WAN model check | `VGM_MODEL_ROOT` | repo wrapper | Reads model tree from profile model root. |
| `scripts/run_wan22_compare_generate.sh:16` | WAN model | `VGM_MODEL_ROOT` | repo wrapper | Uses profile model root. |
| `scripts/run_wan22_compare_generate.sh:19` | comparison outputs | `VGM_OUTPUT_ROOT` | repo wrapper | Uses `evaluation/wan22_compare` output subdir. |
| `scripts/run_wan22_compare_score.sh:33` | VGGT scorer model | `VGM_MODEL_ROOT` | repo wrapper | Uses profile model root. |
| `scripts/run_wan22_i2v_smoke.sh:14` | I2V smoke output | `VGM_OUTPUT_ROOT` | repo wrapper | Uses `videogpa/wan22_5b_i2v/smoke`. |
| `scripts/videogpa/wan22_5b_t2v/run_smoke.sh:62` | output root | `VGM_OUTPUT_ROOT` | repo wrapper | Resolves YAML output subdir via `vgm_common.config`. |
| `scripts/videogpa/wan22_5b_t2v/run_smoke.sh:76` | run environment | profile env | repo wrapper | Saves `environment.txt` in the run dir. |
| `scripts/videogpa/wan22_5b_t2v/run_formal.sh:62` | output root | `VGM_OUTPUT_ROOT` | repo wrapper | Same as smoke for formal YAML. |
| `scripts/videogpa/wan22_5b_t2v/00_preflight.py:232` | master/caption manifests | `VGM_MANIFEST_ROOT` | repo wrapper | Reads manifests from profile manifest root. |
| `scripts/videogpa/wan22_5b_t2v/01_make_smoke_subset.py:42` | master manifest | `VGM_MANIFEST_ROOT` | repo wrapper | Reads `master_all.jsonl` from profile manifest root. |
| `scripts/videogpa/wan22_5b_t2v/score_preferences.py:145` | HF cache | `VGM_MODEL_ROOT` | repo wrapper | Keeps model cache under profile model root. |
| `scripts/data/dl3dv_conditions/common.py:129` | storage layout | `VGM_DL3DV_ROOT` | repo script | Replaces scratch-root guessing with profile-derived layout. |
| `scripts/data/download_dl3dv_first_frames.py:220` | caption index | `VGM_MANIFEST_ROOT` | repo script | Reads caption index from profile manifest root. |
| `scripts/data/download_dl3dv_first_frames.py:344` | download report | `VGM_VALIDATION_ROOT` | repo script | Writes validation reports outside committed metadata. |
| `scripts/data/export_prompt_jsons.py:80` | shared protocol | `VGM_MANIFEST_ROOT` | repo script | Writes prompt JSONs/JSONL under profile manifests. |
| `scripts/data/validate_condition_pack.py:150` | manifest validation | `VGM_MANIFEST_ROOT` | repo script | Validates relative manifest fields and resolves against DL3DV root. |
| `scripts/data/validate_condition_pack.py:282` | validation outputs | `VGM_VALIDATION_ROOT` | repo script | Writes final reports under validation root. |
| `VideoGPA/generate/Wan2.2-T2V-5B.py:102` | WAN model/config | resolver output | repo-added adapter | Uses central config resolver; generation semantics unchanged. |
| `VideoGPA/train/Wan2.2-T2V-5B/02_encode.py:110` | WAN model/config | resolver output | repo-added adapter | Uses central config resolver; encoding semantics unchanged. |
| `VideoGPA/train/Wan2.2-T2V-5B/03_train.py:144` | WAN model/config | resolver output | repo-added adapter | Uses central config resolver; training semantics unchanged. |

## Official Or Third-Party Code

Most official code under `VideoGPA`, `Epipolar-DPO`, and `third_party` was left
unchanged. The only updated files under `VideoGPA` are the repo-added WAN2.2 T2V
bridge files:

- `VideoGPA/generate/Wan2.2-T2V-5B.py`
- `VideoGPA/train/Wan2.2-T2V-5B/02_encode.py`
- `VideoGPA/train/Wan2.2-T2V-5B/03_train.py`

Those changes only route model/config paths through `vgm_common.config`.

## Remaining Hardcoded Paths

- `configs/paths/cluster_zk.sh` intentionally contains the cluster absolute
  root for the `cluster_zk` profile.
- Documentation contains literal example commands for the cluster path.
- Original upstream examples in `third_party`, `VideoGPA`, and `Epipolar-DPO`
  may still contain example paths; they are not on the repo wrapper path.
