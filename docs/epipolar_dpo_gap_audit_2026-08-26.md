# Epipolar-DPO Gap Audit for WAN2.2 5B Candidates

Date: 2026-08-26

Scope: read-only code/README audit of the local repository plus the current
official `KupynOrest/epipolar-dpo` GitHub repository at HEAD `3eba627`. No
candidate videos, manifests, checkpoints, or output directories were modified.

## Bottom Line

The existing WAN2.2 5B T2V/I2V candidate generation stage saves MP4 videos and
candidate-group metadata, but it does not save the final denoised latent, T5
text condition embedding, or I2V image condition embedding/latent at generation
time.

Those training tensors are created later by the VideoGPA encoding stage:

- video latents are reconstructed by VAE-encoding decoded MP4 frames;
- text conditions are re-encoded with the WAN T5 encoder and saved as
  `encoder_hidden_states`;
- 5B I2V first-frame conditioning is VAE-encoded and saved as `image_latent`.

Therefore the existing candidate MP4s can be reused for Epipolar scoring, but
the raw candidate-generation output cannot be passed directly into the official
Epipolar-DPO `model_training/reward_lora/dataset.py` without either regenerating
with latent/condition capture or building a conversion/encoding adapter.

## Evidence From Current 3DVGM Code

### WAN2.2 5B Candidate Generation

T2V candidate generation calls `engine.generate(...)`, saves the returned video
tensor with ffmpeg, validates it with ffprobe, and writes only per-video metadata
such as `generation_id`, `seed`, `video_path`, `frame_num`, `size`, `fps`,
`lora_loaded`, and `ffprobe`.

Relevant files:

- `VideoGPA/generate/Wan2.2-T2V-5B.py`
- `VideoGPA/generate/Wan2.2-I2V-5B.py`
- `scripts/videogpa/wan22_5b_t2v/02_generate_candidates.sh`
- `scripts/videogpa/wan22_5b_i2v/02_generate_candidates.sh`

The I2V manifest also records `image_path`, `image_prompt`, `camera_motion`,
`task: i2v`, and `image_conditioned: true`, but no tensor paths.

An explicit search for `latent_path`, `condition_path`,
`encoder_hidden_states`, `image_latent`, `image_embedding`,
`prompt_embedding`, `torch.save`, `np.save`, `.pt`, and `.safetensors` in the
two 5B generation entrypoints found no generation-time tensor save sites.

### WAN2.2 Native Sampler Internals

The vendored WAN2.2 `WanTI2V` implementation does have an internal final latent
(`x0`) before VAE decoding in both T2V and I2V branches, but the public return
value is only `videos[0]`.

This means exact generation latents are not recoverable from the current saved
MP4s. Post-hoc VAE encoding can create usable training latents from the decoded
video, but that is not the same artifact as the final latent present at sampling
time before video compression.

### VideoGPA Encoding Stage

The local VideoGPA encoding stage fills the tensor gap for its own trainer:

- `VideoGPA/train/Wan2.2-T2V-5B/02_encode.py` reads `preference_pairs.json`,
  VAE-encodes winner/loser MP4s, saves `encoded/winners/*.pt` and
  `encoded/losers/*.pt`, and writes `latent_path` entries.
- It re-runs T5 and saves a shared condition file with
  `encoder_hidden_states`.
- For 5B I2V it resolves the first frame, VAE-encodes it, and saves
  `image_latent` into the condition file.
- `VideoGPA/train/dataset.py` consumes `encoder_hidden_states`, optional
  `image_embeds`, optional `image_latent`, and optional `i2v_y`.

This is compatible with the local VideoGPA WAN2.2 trainer, not with the official
Epipolar-DPO dataset format as-is.

## Evidence From Official Epipolar-DPO

Official repository and paper state a four-step workflow:

1. generate multiple videos per prompt while saving video outputs and latent
   representations;
2. evaluate/annotate generated videos with 3D consistency metrics;
3. train a reward LoRA with Flow-DPO using the annotated dataset;
4. generate/evaluate with the trained LoRA.

The official generation code expects its pipeline to return
`(video_frames, latents, condition)`, then writes:

- `videos/*.mp4`
- `latents/*.pt`
- `condition/*.pt`
- metadata entries with `video_path`, `latent_path`, `condition_path`,
  `original_video_path`, `dataset_source`, caption fields, seed/fps, etc.

The official `wan/pipeline.py` returns:

```python
condition = {"prompt_embedding": prompt_emb_posi, "image_embedding": image_emb}
return frames, latents, condition
```

The official `model_training/reward_lora/dataset.py` expects a flat list of
metadata entries. It filters out entries without:

- `original_video_path`
- `latent_path`
- `condition_path`
- an allowed `dataset_source` (`DL3DV-10K` or `RealEstate10K`)
- the configured metric, usually `epipolar_consistency`

It groups entries by `original_video_path`, selects best/worst within each
group, loads each `latent_path`, then loads condition files containing
`prompt_embedding` and optional `image_embedding`.

The official metric path computes `epipolar_consistency` from sampled frame
pairs using feature matching, a fundamental matrix, and Sampson distance; the
paper frames this as reliable intra-prompt ranking for DPO rather than a
globally smooth differentiable reward.

## Field Check

| Field requested | Current WAN2.2 5B candidate generation | Later local VideoGPA encoding | Official Epipolar-DPO expectation |
| --- | --- | --- | --- |
| Final latent | Not saved. `WanTI2V` has internal `x0`, but returns only decoded video. | Reconstructed from MP4 by VAE encode as winner/loser latents. | Metadata entries must point to `latent_path`; official generation saves the pipeline latent directly. |
| T5/text condition embedding | Not saved. | Saved as `encoder_hidden_states` in `encoded/conditions/cond_*.pt`. | Condition file should contain `prompt_embedding` (or legacy `prompt_emb`). |
| I2V image embedding | Not saved. | 5B TI2V saves first-frame VAE `image_latent`; A14B path can save `i2v_y`. | Condition file may contain `image_embedding` dict, e.g. DiffSynth `clip_feature`/`y` for image-conditioned Wan. |

## Components Already Present and Reusable

- WAN2.2 5B T2V/I2V formal candidate generation wrappers and configs.
- Deterministic group/seed manifests for 3147 prompts x 3 seeds in the formal
  5B configs.
- VideoGPA scoring pipeline over `candidate_groups.json`, including
  `consistency_score` and `motion_norm` pair filtering.
- A separate `VideoGPA/metrics/epipolar.py` implementation that can compute an
  `Epipolar`/Sampson metric with SIFT or LightGlue.
- `VideoGPA/replicate_scorer.py` can emit `epipolar` and `sampson_error` for a
  flat benchmark-style video directory.
- WAN2.2 5B DPO LoRA training machinery exists locally and can train from
  encoded pairs, including the 5B I2V clean-first-latent path.
- Path-profile infrastructure (`VGM_*`) can support a sibling EpipolarDPO output
  namespace without disturbing VideoGPA outputs.

## Missing Pieces for Epipolar-DPO Fine-Tuning

1. Generation-time latent and condition capture.
   - Needed if the goal is to follow official Epipolar-DPO exactly.
   - Requires exposing `x0`, T5 context, and I2V first-frame condition from
     `WanTI2V.generate/t2v/i2v`, then saving tensor paths in the candidate
     manifest.
   - Alternative: accept post-hoc VAE latents from MP4s via an encoder adapter,
     but record that these are reconstructed/compressed-video latents, not exact
     sampling latents.

2. Official-format Epipolar annotated metadata.
   - Current manifests are nested `groups -> videos`.
   - Official dataset expects a flat list grouped by `original_video_path` and
     containing `dataset_source`, `video_path`, `latent_path`, `condition_path`,
     `epipolar_consistency`, and `motion_dynamics`.
   - A converter is needed, or the official dataset needs to be adapted to the
     local `candidate_groups/encoded_pairs` schema.

3. Epipolar scoring adapter for existing candidates.
   - Current formal score script ranks by VideoGPA `consistency_score`.
   - Need a stage that reads `candidate_groups.json`, computes Sampson
     epipolar error for each MP4, writes `epipolar_consistency`, carries detailed
     per-video diagnostics, and uses `metric_mode: min`.
   - Existing `VideoGPA/metrics/epipolar.py` and official
     `metrics/video_evaluation/epipolar.py` are useful, but no production
     adapter currently connects them to the WAN2.2 5B candidate manifests.

4. Motion filtering compatible with Epipolar-DPO.
   - Current pipeline filters with VideoGPA `motion_norm >= 0.001`.
   - Official dataset code uses `motion_dynamics` and a `MOTION_THRESHOLD = 0.9`
     static/dynamics filter.
   - Need to decide whether to compute official `motion_dynamics`, map
     `motion_norm` to a documented local filter, or disable/patch the official
     filter with explicit provenance.

5. Condition-key adapter.
   - Current local condition files use `encoder_hidden_states` and
     `image_latent`.
   - Official dataset/train code expects `prompt_embedding` and optional
     `image_embedding`.
   - Need either condition conversion files or a dataset/trainer shim that maps
     local keys to official keys.

6. I2V condition semantics decision.
   - For local WAN2.2 5B TI2V, image conditioning is a first-frame VAE latent
     path (`image_latent`) used by the local trainer.
   - Official DiffSynth-style Wan image conditioning stores an `image_embedding`
     dict. If using official `FlowDPOTrainer`, the local `image_latent` is not
     enough without adapting the model input path.

7. Separate EpipolarDPO config/harness/output namespace.
   - Current formal configs are under `configs/videogpa/*` and output under
     `videogpa/...`.
   - Need sibling configs/scripts such as `configs/epipolar_dpo/...` and
     output subdirs such as `epipolar_dpo/wan22_5b_{t2v,i2v}/...` to reuse
     candidate MP4s while keeping scoring, annotated manifests, encoding, and
     LoRA checkpoints isolated.

8. Validation gates.
   - Before training, the EpipolarDPO stage should validate: candidate count,
     ffprobe frame counts, score coverage, finite epipolar metrics, motion
     filter counts, pair counts by reason, tensor-file existence, condition-key
     compatibility, tensor shapes/dtypes, and no accidental reuse of debug
     fallback pairs.

## Recommended Next Implementation Order

1. Build a read-only Epipolar scoring adapter over existing `candidate_groups.json`.
2. Emit an Epipolar annotated manifest plus pair/filter summary, without touching
   the candidate MP4 directory.
3. Decide latent strategy:
   - exact: regenerate or patch generation for `x0`/condition capture;
   - pragmatic: post-hoc VAE encode existing MP4s and mark provenance.
4. Add condition-key conversion or dataset shim.
5. Add an isolated EpipolarDPO train config/harness that consumes the annotated
   manifest and writes to a sibling output namespace.

## Source Links

- Official GitHub: https://github.com/KupynOrest/epipolar-dpo
- Official project page: https://epipolar-dpo.github.io/
- Paper: https://arxiv.org/abs/2510.21615
