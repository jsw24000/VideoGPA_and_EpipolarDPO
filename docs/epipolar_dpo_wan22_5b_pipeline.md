# WAN2.2 5B Epipolar-DPO Sibling Pipeline

This pipeline reuses completed VideoGPA raw candidates as read-only sources and
creates independent Epipolar-DPO runs under `VGM_OUTPUT_ROOT`.

## Source Runs

- T2V source: `$VGM_OUTPUT_ROOT/videogpa/wan22_5b_t2v/formal/wan22_5b_t2v_formal_001`
- I2V source: `$VGM_OUTPUT_ROOT/videogpa/wan22_5b_i2v/formal/wan22_5b_i2v_formal_001`
- Source manifest: `manifests/candidate_groups.json`
- Expected source shape: 3147 groups, 3 candidates per group, 9441 MP4 files.
- Candidate `video_path` values remain relative to the source run, for example
  `candidates/<group_id>/seed_1001.mp4`.

Source runs are treated as read-only. Epipolar stages must not write scores,
pairs, tensors, logs, checkpoints, or DONE markers into these source folders.

## Output Runs

- T2V output root: `$VGM_OUTPUT_ROOT/epipolar_dpo/wan22_5b_t2v/formal`
- I2V output root: `$VGM_OUTPUT_ROOT/epipolar_dpo/wan22_5b_i2v/formal`

Each run stores:

- `config_resolved.yaml`
- `command.txt`
- `environment.txt`
- `git_state.txt`
- `reproduce.sh`
- `run_state.json`
- `manifests/source_validation.json`
- `manifests/scored_candidates.json`
- `manifests/preference_pairs.json`
- `manifests/pair_summary.json`
- `manifests/encoded_pairs.json`
- `encoded/`
- `checkpoints/`
- `logs/`
- `reports/`

## Stages

The formal wrappers run:

1. `preflight`
2. `static_checks`
3. `source_validation`
4. `epipolar_scoring`
5. `pair_selection`
6. `encoding`
7. `training`

There is no generation stage. Missing formal source candidates are a hard
failure.

## Scoring And Pairing

Epipolar scoring uses the upstream repository implementation:

- `Epipolar-DPO/metrics/video_evaluation/epipolar.py`
- metric field: `epipolar_consistency`
- metric mode: `min`
- default descriptor: `sift`
- default frame sampling: every 15 frames
- aggregation: mean Sampson distance over consecutive sampled frame pairs

Motion filtering uses:

- `Epipolar-DPO/metrics/video_evaluation/dynamics.py`
- field: `motion_dynamics`
- default upper threshold: `0.9`, matching upstream `MOTION_THRESHOLD`

Pair selection is frozen into `manifests/preference_pairs.json`. For each group,
valid candidates are filtered by metric validity and motion, sorted by
`epipolar_consistency` ascending, and paired as best winner vs worst loser.

## Encoding And Training

Encoding reuses `VideoGPA/train/Wan2.2-T2V-5B/02_encode.py`:

- source MP4s are decoded and re-encoded through the WAN VAE.
- latent provenance is recorded as `posthoc_mp4_vae`.
- T2V condition schema is `encoder_hidden_states`.
- I2V condition schema is `encoder_hidden_states + image_latent`.

Training reuses the local WAN2.2 native trainer:

- policy: raw WAN2.2 5B plus trainable LoRA
- reference: frozen raw WAN2.2 5B without LoRA
- winner/loser coupling: same timestep and same Gaussian noise
- I2V semantics: `single_ti2v_5b` clean-first-frame `image_latent` path
- loss strategy: `epipolar_dpo`
- Flow-DPO formula: `-logsigmoid(-0.5 * beta * (win_diff - lose_diff))`
- default beta: `500.0`

The trainer logs:

- `world_size`
- `batch_size_per_gpu`
- `gradient_accumulation_steps`
- `effective_global_pair_batch`

## Commands

Formal T2V:

```bash
source scripts/env/activate_profile.sh cluster_zk
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/epipolar_dpo/wan22_5b_t2v/run_formal.sh
```

Formal I2V:

```bash
source scripts/env/activate_profile.sh cluster_zk
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/epipolar_dpo/wan22_5b_i2v/run_formal.sh
```

Remote smoke from source validation through the first training checkpoint:

```bash
source scripts/env/activate_profile.sh cluster_zk
EPIPOLAR_DPO_MAX_GROUPS=16 EPIPOLAR_DPO_MAX_TRAIN_STEPS=1 \
GPU_IDS=0,1,2,3,4,5,6,7 \
bash scripts/epipolar_dpo/wan22_5b_t2v/run_formal.sh --run-id epipolar_t2v_smoke_001
```

```bash
source scripts/env/activate_profile.sh cluster_zk
EPIPOLAR_DPO_MAX_GROUPS=16 EPIPOLAR_DPO_MAX_TRAIN_STEPS=1 \
GPU_IDS=0,1,2,3,4,5,6,7 \
bash scripts/epipolar_dpo/wan22_5b_i2v/run_formal.sh --run-id epipolar_i2v_smoke_001
```

Resume or rerun a stage:

```bash
bash scripts/epipolar_dpo/wan22_5b_t2v/run_formal.sh --resume --run-id <run_id>
bash scripts/epipolar_dpo/wan22_5b_t2v/run_formal.sh --resume --run-id <run_id> --force-stage epipolar_scoring
bash scripts/epipolar_dpo/wan22_5b_t2v/run_formal.sh --stop-after source_validation
```
