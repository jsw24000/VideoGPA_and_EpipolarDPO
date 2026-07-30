# Path Layout

All repo-owned wrappers resolve paths from the active profile:

```text
VGM_PROFILE
VGM_ROOT
VGM_REPO_ROOT
VGM_DL3DV_ROOT
VGM_MODEL_ROOT
VGM_OUTPUT_ROOT
VGM_ARCHIVES_ROOT
VGM_EXTRACTED_ROOT
VGM_MANIFEST_ROOT
VGM_FIRST_FRAMES_ROOT
VGM_VALIDATION_ROOT
```

Committed YAML stores relative subpaths only. For example:

- model: `VGM_MODEL_ROOT / model.model_relpath`
- manifest: `VGM_DL3DV_ROOT / data.manifest_relpath`
- first frames: `VGM_DL3DV_ROOT / data.first_frames_relroot`
- outputs: `VGM_OUTPUT_ROOT / experiment.output_subdir`

Resolved absolute paths may be written to run outputs such as
`config_resolved.yaml`; those files are runtime artifacts and are ignored.
