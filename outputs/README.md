# Outputs Directory

The `local` profile resolves `VGM_OUTPUT_ROOT` to this directory. Experiment
runs, generated videos, checkpoints, logs, and resolved configs are written here
for local runs.

The `cluster_zk` profile resolves `VGM_OUTPUT_ROOT` outside the repository.
Runtime outputs are intentionally not tracked by Git.
