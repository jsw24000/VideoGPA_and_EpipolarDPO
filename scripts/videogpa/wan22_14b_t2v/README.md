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
For the three seeds of one prompt, T5 prompt embeddings are cached by default (`cache_text_embeddings: true`). The positive embedding is retained only for the current prompt; the shared negative embedding is reused by the worker. Set `CACHE_TEXT_EMBEDDINGS=0` for an A/B check.

Multi-GPU generation has two modes:

- `A14B_PARALLEL_MODE=distributed` is the default. `GPU_IDS` is treated as one A14B distributed worker group for one video at a time. The launcher defaults to PyTorch FSDP for DiT and T5 plus Wan sequence parallel (`DIT_FSDP=1 T5_FSDP=1 USE_SP=1 ULYSSES_SIZE=<gpu_count>`), matching the official FSDP + Ulysses inference path more closely than FSDP-only launches.
- `A14B_PARALLEL_MODE=throughput` runs one independent single-GPU A14B process per listed GPU and shards prompts across them. This is usually better for total candidate throughput on four 80GB GPUs, although native single-GPU A14B still swaps its two experts at the noise boundary as described below.

The generation log prints the resolved runtime flags and per-rank CUDA memory after engine load:

```text
[wan22_14b_t2v] distributed A14B generation: GPU_IDS=... ULYSSES_SIZE=...
[rank 0] CUDA after engine load: ... offload_model=False init_on_cpu=False attention_backend=fa3
```

If output is still too slow, first compare these log lines with `nvidia-smi dmon` or `nvidia-smi pmon` on the remote node. FSDP without `use_sp=True` can shard memory without giving much single-video compute speedup.

`offload_model=False` alone does not guarantee that a single-GPU throughput worker has no host transfers. Native `WanT2V` defaults to `init_on_cpu=True` without FSDP or sequence parallel, so it keeps only the active 14B expert on GPU and swaps experts at the noise boundary. This is intentional for the roughly 27B total dual-expert model and is logged explicitly:

```text
[WanT2V transfer] rank=0 offload_model=False init_on_cpu=True actions=... seconds=...
[WanT2V timing] rank=0 text_encode_s=... denoise_s=... vae_decode_s=...
[timing] group=... seed=... t5_cache_prepare_s=... generate_including_vae_s=... mp4_save_s=... total_s=...
```

Use the timing lines before adding asynchronous video encoding. If `mp4_save_s` is only a few seconds compared with about 870 seconds of denoising, ffmpeg pipelining cannot materially improve throughput and may increase four-worker CPU contention.

The selected attention path is also logged. `attention_backend=fa2` is expected on A100. On H100/H800, the official Wan benchmark uses FA3; test FA3 in a cloned environment if the log still says `fa2`, then compare identical prompt/seed micro runs before changing the formal environment.

Throughput micro check:

```bash
export A14B_PARALLEL_MODE=throughput
export GPU_IDS=0,1,2,3
RUN_DIR="$VGM_OUTPUT_ROOT/videogpa/wan22_14b_t2v/formal/<run_id>" MODE=micro \
  bash scripts/videogpa/wan22_14b_t2v/02_generate_candidates.sh
```

If any shard OOMs, retry the micro check with `OFFLOAD_MODEL=1` or use the default distributed mode. CPU offload is a compatibility fallback and can be much slower.

Inspect all completed timing records without requiring `rg`:

```bash
for f in "${RUN_DIR}"/logs/generation.shard_*.log; do
  echo "===== ${f} ====="
  tr '\r' '\n' < "${f}" | grep -aE '\[WanT2V transfer\]|\[WanT2V timing\]|\[timing\]|attention_backend='
done
```

To measure four-worker contention, compare one fresh single-GPU micro run against a fresh four-GPU throughput micro run with the same resolution, frame count, steps, prompt, and seed. Do not reuse an output directory containing valid MP4s because generation will skip them. During each run capture GPU, CPU, and storage pressure with `nvidia-smi dmon -s pucm -d 5`, `pidstat -rud -h 5`, and `iostat -xz 5`. A single worker near 12 minutes versus four workers near 14.5 minutes indicates shared CPU memory, PCIe, or storage contention of about 20%; the new transfer and save timings show which one.
