# DL3DV Runtime Data

This directory is the local-profile DL3DV root:

```text
VGM_DL3DV_ROOT=<repo>/data
```

For `cluster_zk`, `VGM_DL3DV_ROOT` points outside the repository. Codex running
locally cannot see that remote data, which is expected.

Tracked content here is limited to lightweight schemas, placeholders, and docs.
Generated manifests, first frames, logs, validation reports, and archives are
runtime data and are ignored by Git.

Use the profile-aware data scripts from `scripts/data`:

```bash
source scripts/env/activate_profile.sh local
python scripts/data/run_dl3dv_pipeline.py --dry-run --splits 1K --limit 1
```

Canonical manifest records should store paths relative to `VGM_DL3DV_ROOT`, for
example `first_frames/train/8K/<scene>/first_frame.png`, and loaders should
resolve them through `vgm_common.paths`.
