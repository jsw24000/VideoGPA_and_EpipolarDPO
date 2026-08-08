# Cluster ZK Profile

Use cluster-owned data, model, and output folders outside the clone:

```bash
cd /data/pbq/system/peibaoqi/project_a/zk/repos/VideoGPA_and_EpipolarDPO
source scripts/env/activate_profile.sh cluster_zk
python scripts/env/check_paths.py --strict
```

Expected roots:

```text
VGM_ROOT=/data/pbq/system/peibaoqi/project_a/zk
VGM_REPO_ROOT=/data/pbq/system/peibaoqi/project_a/zk/repos/VideoGPA_and_EpipolarDPO
VGM_DL3DV_ROOT=/data/pbq/system/peibaoqi/project_a/zk/data/DL3DV-ALL-960P
VGM_MODEL_ROOT=/data/pbq/system/peibaoqi/project_a/zk/models
VGM_OUTPUT_ROOT=/data/pbq/system/peibaoqi/project_a/zk/outputs/VideoGPA_and_EpipolarDPO
```

DL3DV subdirectories:

```text
$VGM_DL3DV_ROOT/archives
$VGM_DL3DV_ROOT/extracted
$VGM_DL3DV_ROOT/manifests
$VGM_DL3DV_ROOT/first_frames
$VGM_DL3DV_ROOT/validation
```

Codex running locally cannot verify or create these remote paths. On the
cluster, use `--strict` after activation to require the directories to exist.
