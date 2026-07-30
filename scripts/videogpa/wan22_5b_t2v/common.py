from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vgm_common.config import resolve_experiment_config  # noqa: E402
from vgm_common.paths import get_model_root, get_repo_root, resolve_repo_path  # noqa: E402


def find_project_root(start: Path | None = None) -> Path:
    return get_repo_root()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_id(value: str) -> str:
    return value.strip().replace("/", "__").replace("\\", "__").replace(" ", "_")


def parse_size(size: str) -> tuple[int, int]:
    if "*" not in size:
        raise ValueError(f"Expected size as WIDTH*HEIGHT, got {size!r}")
    w, h = size.split("*", 1)
    return int(w), int(h)


def git_short_hash(project_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or "nogit"
    except Exception:
        return "nogit"


def iter_dirs_limited(root: Path, max_depth: int = 5):
    root = root.resolve()
    if not root.exists():
        return
    for parent, dirs, _files in os.walk(root):
        parent_path = Path(parent)
        try:
            depth = len(parent_path.relative_to(root).parts)
        except ValueError:
            continue
        if depth > max_depth:
            dirs[:] = []
            continue
        yield parent_path
        if depth == max_depth:
            dirs[:] = []


def make_run_id(project_root: Path) -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{git_short_hash(project_root)}"


def _find_unique_model_dir(models_root: Path, kind: str) -> Path:
    if kind == "wan":
        candidates = [
            p
            for p in iter_dirs_limited(models_root, max_depth=5)
            if p.is_dir()
            and p.name == "Wan2.2-TI2V-5B"
            and (p / "Wan2.2_VAE.pth").is_file()
            and (p / "models_t5_umt5-xxl-enc-bf16.pth").is_file()
            and (p / "diffusion_pytorch_model.safetensors.index.json").is_file()
        ]
    elif kind == "vggt":
        candidates = [
            p
            for p in iter_dirs_limited(models_root, max_depth=5)
            if p.is_dir()
            and p.name == "VGGT-1B"
            and (p / "config.json").is_file()
            and ((p / "model.safetensors").is_file() or (p / "model.pt").is_file())
        ]
    else:
        raise ValueError(kind)
    candidates = sorted({p.resolve() for p in candidates})
    if not candidates:
        raise FileNotFoundError(f"No {kind} model candidate found under {models_root}")
    if len(candidates) > 1:
        lines = "\n".join(str(p) for p in candidates)
        raise RuntimeError(f"Multiple {kind} model candidates found:\n{lines}")
    return candidates[0]


def resolve_path(project_root: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        return resolve_repo_path(path)
    return path.resolve(strict=False)


def resolve_config(config_path: Path, run_dir: Path | None = None) -> dict[str, Any]:
    return resolve_experiment_config(config_path, run_dir)


def require_files(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))


def deterministic_sample(items: list[Any], size: int, seed: int) -> list[Any]:
    if size >= len(items):
        return list(items)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(items)), size))
    return [items[i] for i in indices]


def relpath(path: Path, base: Path) -> str:
    return str(path.resolve().relative_to(base.resolve()))
