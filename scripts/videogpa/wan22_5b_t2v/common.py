from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def find_project_root(start: Path | None = None) -> Path:
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    cur = (start or Path(__file__)).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / "VideoGPA").is_dir() and (parent / "data" / "manifests").is_dir():
            return parent
    raise RuntimeError(f"Could not locate project root from {cur}")


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
        path = project_root / path
    return path.resolve()


def resolve_config(config_path: Path, run_dir: Path | None = None) -> dict[str, Any]:
    project_root = find_project_root(config_path)
    cfg = read_yaml(config_path)
    cfg.setdefault("project", {})
    cfg.setdefault("paths", {})
    cfg["project"]["project_root"] = str(project_root)

    paths = cfg["paths"]
    if os.environ.get("VIDEOGPA_ROOT"):
        paths["videogpa_root"] = os.environ["VIDEOGPA_ROOT"]
    if os.environ.get("VIDEOGPA_OUTPUT_ROOT"):
        paths["output_root"] = os.environ["VIDEOGPA_OUTPUT_ROOT"]
    if os.environ.get("WAN22_5B_MODEL_PATH"):
        paths["wan_model_path"] = os.environ["WAN22_5B_MODEL_PATH"]
    if os.environ.get("VGGT_MODEL_PATH"):
        paths["vggt_model_path"] = os.environ["VGGT_MODEL_PATH"]

    for key in ["videogpa_root", "train_manifest", "output_root"]:
        paths[key] = str(resolve_path(project_root, paths[key]))

    models_root = project_root / "models"
    if paths.get("wan_model_path", "auto") == "auto":
        paths["wan_model_path"] = str(_find_unique_model_dir(models_root, "wan"))
    else:
        paths["wan_model_path"] = str(resolve_path(project_root, paths["wan_model_path"]))

    if paths.get("vggt_model_path", "auto") == "auto":
        paths["vggt_model_path"] = str(_find_unique_model_dir(models_root, "vggt"))
    else:
        paths["vggt_model_path"] = str(resolve_path(project_root, paths["vggt_model_path"]))

    if run_dir is not None:
        cfg["paths"]["run_dir"] = str(resolve_path(project_root, run_dir))
    return cfg


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
