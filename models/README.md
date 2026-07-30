# Models Directory

The `local` profile resolves `VGM_MODEL_ROOT` to this directory. Put local model
weights here when running on this machine.

The `cluster_zk` profile resolves `VGM_MODEL_ROOT` outside the repository. Model
weights, checkpoints, and safetensors are intentionally not tracked by Git.
