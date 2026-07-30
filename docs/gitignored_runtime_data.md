# Gitignored Runtime Data

The following runtime payloads are intentionally not tracked:

- `data/first_frames/*`
- `data/manifests/*`
- `data/logs/*`
- `data/reports/generated/*`
- `data/validation/*`
- `data/first_frames.tar`
- `models/*`
- `outputs/*`
- `wandb/`
- checkpoint and model weight files such as `*.ckpt`, `*.pt`, `*.pth`, and `*.safetensors`

Tracked placeholder files keep the directory shape. Removing a runtime file
from Git index does not delete the local file.
