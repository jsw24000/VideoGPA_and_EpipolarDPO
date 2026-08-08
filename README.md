# 3DVGM / VideoGPA and Epipolar-DPO

This repo uses path profiles so the same code can run locally with repo-local
runtime folders and on the `cluster_zk` environment with data, models, and
outputs outside the clone.

Local setup:

```bash
cd <repo>
source scripts/env/activate_profile.sh local
python scripts/env/check_paths.py
```

Cluster setup:

```bash
cd /data/pbq/system/peibaoqi/project_a/zk/repos/VideoGPA_and_EpipolarDPO
source scripts/env/activate_profile.sh cluster_zk
python scripts/env/check_paths.py --strict
```

Codex in this local workspace cannot see the remote cluster data/model/output
directories. That is expected. A fresh clone also does not contain models,
first frames, generated manifests, or outputs.

Before training on the cluster, place models under `VGM_MODEL_ROOT`, run the
DL3DV data pipeline to generate manifests and first frames under
`VGM_DL3DV_ROOT`, then run the smoke or formal wrappers. Outputs resolve through
`VGM_OUTPUT_ROOT`.

Useful docs:

- [Path Layout](docs/path_layout.md)
- [Local Profile](docs/local_profile.md)
- [Cluster ZK Profile](docs/cluster_zk_profile.md)
- [Data Pipeline](docs/data_pipeline.md)
- [Gitignored Runtime Data](docs/gitignored_runtime_data.md)
