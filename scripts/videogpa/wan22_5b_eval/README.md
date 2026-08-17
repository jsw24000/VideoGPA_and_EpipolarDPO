# WAN2.2 DL3DV Evaluation

This runner evaluates one WAN2.2 training run at a time. It infers T2V or I2V from `RUN_DIR/config_resolved.yaml`, deterministically selects 100 scenes from DL3DV-1K with seed 456, and evaluates at most one fine-tuned checkpoint from that run. The WAN baseline is enabled by default and can be skipped.

The same canonical subset is used across model sizes, tasks, and methods:

- split: DL3DV-1K test
- prompt: CogVLM2 natural caption
- T2V input: text only
- I2V input: the same text plus the same-scene first frame
- samples per variant: 100 by default
- baseline plus fine-tuned checkpoint: 200 generated videos total
- fine-tuned checkpoint with `--skip-baseline`: 100 generated videos total
- evaluator: DA3-Large, 10 uniformly sampled frames, confidence threshold 0, LightGlue

## Default Run

The default command evaluates the baseline and the latest complete checkpoint under the run directory. `--task` and `--limit` are not needed.

```bash
source scripts/env/activate_profile.sh cluster_zk
cd "${VGM_REPO_ROOT}"
unset CUDA_VISIBLE_DEVICES

export RUN_DIR="${VGM_OUTPUT_ROOT}/videogpa/wan22_5b_t2v/formal/wan22_5b_t2v_formal_001"
export GPU_IDS=0,1,2,3
export SCORE_DEVICES=0,1,2,3
export VIDEOGPA_CONDA_ENV=wan22_videogpa

bash scripts/videogpa/wan22_5b_eval/run_eval.sh --run-dir "${RUN_DIR}"
```

Evaluate only the fine-tuned checkpoint and omit the baseline:

```bash
bash scripts/videogpa/wan22_5b_eval/run_eval.sh \
  --run-dir "${RUN_DIR}" \
  --skip-baseline
```

Override the automatically discovered checkpoint when needed:

```bash
bash scripts/videogpa/wan22_5b_eval/run_eval.sh \
  --run-dir "${RUN_DIR}" \
  --variant "videogpa_step_010000=${RUN_DIR}/checkpoints/step_010000:0.2"
```

Generate and score only the baseline:

```bash
bash scripts/videogpa/wan22_5b_eval/run_eval.sh \
  --run-dir "${RUN_DIR}" \
  --baseline-only
```

## Output Layout

```text
$RUN_DIR/evaluation/dl3dv1k_seed456/
  manifests/
    eval_1k_seed456.json
    task_eval_1k_seed456.json
    baseline.candidate_groups.json
    <method>_step_<step>.candidate_groups.json
  generation/
    baseline/<group_id>/seed_456.mp4
    <method>_step_<step>/<group_id>/seed_456.mp4
  scores/da3/
    baseline_scores.csv
    baseline_scores.json
    <method>_step_<step>_scores.csv
    <method>_step_<step>_scores.json
    <method>_step_<step>_vs_baseline_summary.md
  logs/
  config/environment.txt
```

The runner rejects multiple comma-separated variants and refuses scoring unless each selected variant contains exactly the manifest's number of `seed_456.mp4` files. This prevents stale 1K outputs from being mixed with the 100-scene subset.
