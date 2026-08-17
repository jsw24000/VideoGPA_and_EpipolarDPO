# WAN2.2 5B I2V VideoGPA Formal Scripts

This directory mirrors the WAN2.2 5B T2V formal chain, but uses `train_i2v.json` samples with a first-frame image and camera-motion prompt.

Cluster output resolves through the active path profile:

```bash
source scripts/env/activate_profile.sh cluster_zk
python -m vgm_common.config --config configs/videogpa/wan22_5b_i2v_formal.yaml --print output_root
```

Expected cluster root:

```text
$VGM_OUTPUT_ROOT/videogpa/wan22_5b_i2v/formal
```

Run candidate generation on physical GPUs 4-7:

```bash
source scripts/env/activate_profile.sh cluster_zk
cd "${VGM_REPO_ROOT}"
unset CUDA_VISIBLE_DEVICES
export VIDEOGPA_CONDA_ENV=wan22_videogpa
export GPU_IDS=4,5,6,7
bash scripts/videogpa/wan22_5b_i2v/run_formal_generation.sh --run-id wan22_5b_i2v_formal_001
```

Run the full VideoGPA chain:

```bash
source scripts/env/activate_profile.sh cluster_zk
cd "${VGM_REPO_ROOT}"
unset CUDA_VISIBLE_DEVICES
export VIDEOGPA_CONDA_ENV=wan22_videogpa
export GPU_IDS=4,5,6,7
bash scripts/videogpa/wan22_5b_i2v/run_formal.sh --run-id wan22_5b_i2v_formal_001 --resume
```

Resume the same run:

```bash
bash scripts/videogpa/wan22_5b_i2v/run_formal_generation.sh --run-id wan22_5b_i2v_formal_001 --resume
```

The generator follows the T2V shard convention and passes physical `--gpu_id` values to the Python entrypoint. Keep `CUDA_VISIBLE_DEVICES` unset for this runner; otherwise `--gpu_id 4` may refer to a hidden device.

Formal stages are `preflight`, `static_checks`, `subset`, `generation_candidates`, `scoring`, `encoding`, and `training`. `run_formal_generation.sh` remains the safe generation-only entry for isolated candidate creation; `run_formal.sh` continues from the same run directory and reuses existing valid MP4s unless `--force-stage generation_candidates` is explicit.
