# WAN2.2 5B DL3DV-1K Evaluation

This runner builds a canonical DL3DV-1K evaluation manifest and uses it for all WAN2.2 5B T2V/I2V variants. The manifest stores natural CogVLM2 captions, one fixed seed, and same-scene first-frame relative paths for I2V. It then derives task-specific input manifests: the T2V input contains no first-frame fields, while the I2V input keeps same-scene first-frame relative paths. It does not copy first-frame files and does not write local absolute paths.

Default protocol:

- split: DL3DV-1K test
- prompt: CogVLM2 natural caption
- T2V input: text only
- I2V input: same text plus same-scene first frame
- generation: 81 frames, 1280x704, 50 steps, shift 5.0, guide scale 5.0, FPS 24
- primary LoRA weight: 0.2
- evaluator: DA3-Large, 10 uniformly sampled frames, confidence threshold 0, LightGlue epipolar matcher
- metrics: PSNR, SSIM, LPIPS, MVCS, 3DCS, Epipolar/Sampson error

Run the current final VideoGPA T2V checkpoint against the WAN baseline:

```bash
source scripts/env/activate_profile.sh cluster_zk
cd "${VGM_REPO_ROOT}"
unset CUDA_VISIBLE_DEVICES

export RUN_DIR="${VGM_OUTPUT_ROOT}/videogpa/wan22_5b_t2v/formal/wan22_5b_t2v_formal_001"
export GPU_IDS=0,1,2,3
export SCORE_DEVICES=0,1,2,3
export VIDEOGPA_CONDA_ENV=wan22_videogpa

bash scripts/videogpa/wan22_5b_eval/run_eval.sh \
  --run-dir "${RUN_DIR}" \
  --task t2v \
  --seed 456
```

Add another method such as Epipolar-DPO by passing comma-separated variants:

```bash
EVAL_LORA_VARIANTS="videogpa_step_010000=${RUN_DIR}/checkpoints/step_010000:0.2,epipolar_dpo_step_xxxxxx=/path/to/epipolar_dpo/checkpoint:0.2" \
bash scripts/videogpa/wan22_5b_eval/run_eval.sh --run-dir "${RUN_DIR}" --task t2v --seed 456
```

Outputs are written under:

```text
$RUN_DIR/evaluation/dl3dv1k_seed456/
  manifests/eval_1k_seed456.json
  manifests/t2v_eval_1k_seed456.json
  manifests/i2v_eval_1k_seed456.json
  generation/<task>/<variant>/<group_id>/seed_456.mp4
  scores/da3/<task>/<variant>_scores.csv
  scores/da3/<task>/<variant>_scores.json
  scores/da3/<task>/<variant>_vs_baseline_summary.md
  logs/
```
