# Data Pipeline

Data scripts now live under `scripts/data`. The old `data/scripts` entrypoints
are deprecated wrappers only.

Local dry-run:

```bash
cd <repo>
source scripts/env/activate_profile.sh local
python scripts/data/run_dl3dv_pipeline.py --dry-run --splits 1K --limit 1
```

Cluster sequence after clone:

```bash
cd /data/pbq/system/peibaoqi/project_a/zk/repos/VideoGPA_and_EpipolarDPO
source scripts/env/activate_profile.sh cluster_zk
python scripts/env/check_paths.py --strict
python scripts/data/00_preflight_dl3dv.py --dry-run
python scripts/data/run_dl3dv_pipeline.py --resume
```

The pipeline writes:

- first frames: `VGM_FIRST_FRAMES_ROOT`
- manifests: `VGM_MANIFEST_ROOT`
- reports and logs: `VGM_VALIDATION_ROOT`

Canonical manifest records should use relative fields such as
`video_relpath`, `first_frame_relpath`, and `scene_relpath`, all relative to
`VGM_DL3DV_ROOT`.
