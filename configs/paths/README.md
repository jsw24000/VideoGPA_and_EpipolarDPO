# Path Profiles

This directory is the only committed place that stores environment-specific
absolute roots.

Activate one profile before running repo scripts:

```bash
source scripts/env/activate_profile.sh local
source scripts/env/activate_profile.sh cluster_zk
```

The activated environment exports:

- `VGM_PROFILE`
- `VGM_ROOT`
- `VGM_REPO_ROOT`
- `VGM_DL3DV_ROOT`
- `VGM_MODEL_ROOT`
- `VGM_OUTPUT_ROOT`
- `VGM_ARCHIVES_ROOT`
- `VGM_EXTRACTED_ROOT`
- `VGM_MANIFEST_ROOT`
- `VGM_FIRST_FRAMES_ROOT`
- `VGM_VALIDATION_ROOT`

`local` resolves all roots inside the current clone. `cluster_zk` resolves data,
models, and outputs outside the clone.
