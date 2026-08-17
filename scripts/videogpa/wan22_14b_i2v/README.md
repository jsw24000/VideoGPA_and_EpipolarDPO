# WAN2.2 14B I2V VideoGPA Formal Scripts

Formal output root:

```text
$VGM_OUTPUT_ROOT/videogpa/wan22_14b_i2v/formal/<run_id>/
```

Expected model path:

```text
$VGM_MODEL_ROOT/wan/Wan2.2-I2V-A14B
```

Run preflight/static checks:

```bash
source scripts/env/activate_profile.sh cluster_zk
cd "${VGM_REPO_ROOT}"
export VIDEOGPA_CONDA_ENV=wan22_videogpa
export GPU_IDS=4,5,6,7
bash scripts/videogpa/wan22_14b_i2v/run_formal.sh --stop-after static_checks
```

`GPU_IDS` is treated as one A14B distributed worker group for generation, not as one shard per GPU. Candidate generation uses `Wan2.2-A14B.py`, which dispatches to `wan.WanI2V` and resolves first-frame paths through the active profile.
