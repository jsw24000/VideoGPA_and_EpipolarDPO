# Local Profile

Use local repo folders for data, models, and outputs:

```bash
cd <repo>
source scripts/env/activate_profile.sh local
python scripts/env/check_paths.py
```

Resolved roots:

```text
VGM_REPO_ROOT=<repo>
VGM_ROOT=<repo>
VGM_DL3DV_ROOT=<repo>/data
VGM_MODEL_ROOT=<repo>/models
VGM_OUTPUT_ROOT=<repo>/outputs
VGM_MANIFEST_ROOT=<repo>/data/manifests
VGM_FIRST_FRAMES_ROOT=<repo>/data/first_frames
VGM_VALIDATION_ROOT=<repo>/data/validation
```

Local data and model payloads are not tracked by Git. The directory placeholders
and README files are tracked so a clone has the expected shape.
