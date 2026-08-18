# WAN2.2 14B T2V VideoGPA Formal Scripts

Formal output root:

```text
$VGM_OUTPUT_ROOT/videogpa/wan22_14b_t2v/formal/<run_id>/
```

Expected model path:

```text
$VGM_MODEL_ROOT/wan/Wan2.2-T2V-A14B
```

Run preflight/static checks:

```bash
source scripts/env/activate_profile.sh cluster_zk
cd "${VGM_REPO_ROOT}"
export VIDEOGPA_CONDA_ENV=wan22_videogpa
export GPU_IDS=0,1,2,3
bash scripts/videogpa/wan22_14b_t2v/run_formal.sh --stop-after static_checks
```

Candidate generation uses `Wan2.2-A14B.py`, which dispatches to `wan.WanT2V`; scoring, encoding, and training reuse the task-aware VideoGPA scripts.

The formal T2V generation default is 480p landscape (`832*480`) with 81 frames.

Multi-GPU generation has two modes:

- `A14B_PARALLEL_MODE=distributed` is the default. `GPU_IDS` is treated as one A14B distributed worker group for one video at a time. The launcher defaults to PyTorch FSDP for DiT and T5 plus Wan sequence parallel (`DIT_FSDP=1 T5_FSDP=1 USE_SP=1 ULYSSES_SIZE=<gpu_count>`), matching the official FSDP + Ulysses inference path more closely than FSDP-only launches.
- `A14B_PARALLEL_MODE=throughput` runs one independent single-GPU A14B process per listed GPU and shards prompts across them. This is usually better for total candidate throughput if a full 480p A14B process fits in one 80GB GPU without CPU offload.

The generation log prints the resolved runtime flags and per-rank CUDA memory after engine load:

```text
[wan22_14b_t2v] distributed A14B generation: GPU_IDS=... ULYSSES_SIZE=...
[rank 0] CUDA after engine load: ... dit_fsdp=True t5_fsdp=True use_sp=True ulysses_size=4 offload_model=False
```

If output is still too slow, first compare these log lines with `nvidia-smi dmon` or `nvidia-smi pmon` on the remote node. FSDP without `use_sp=True` can shard memory without giving much single-video compute speedup.

Throughput micro check:

```bash
export A14B_PARALLEL_MODE=throughput
export GPU_IDS=0,1,2,3
RUN_DIR="$VGM_OUTPUT_ROOT/videogpa/wan22_14b_t2v/formal/<run_id>" MODE=micro \
  bash scripts/videogpa/wan22_14b_t2v/02_generate_candidates.sh
```

If any shard OOMs, retry the micro check with `OFFLOAD_MODEL=1` or use the default distributed mode. CPU offload is a compatibility fallback and can be much slower.
