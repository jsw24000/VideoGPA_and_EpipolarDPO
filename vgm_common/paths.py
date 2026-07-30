from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Mapping


ROOT_ENV_KEYS = (
    "VGM_PROFILE",
    "VGM_ROOT",
    "VGM_REPO_ROOT",
    "VGM_DL3DV_ROOT",
    "VGM_MODEL_ROOT",
    "VGM_OUTPUT_ROOT",
)

DERIVED_ENV_DEFAULTS = {
    "VGM_ARCHIVES_ROOT": "archives",
    "VGM_EXTRACTED_ROOT": "extracted",
    "VGM_MANIFEST_ROOT": "manifests",
    "VGM_FIRST_FRAMES_ROOT": "first_frames",
    "VGM_VALIDATION_ROOT": "validation",
}

ACTIVATE_HELP = (
    "Activate a path profile first:\n"
    "  source scripts/env/activate_profile.sh local\n"
    "  source scripts/env/activate_profile.sh cluster_zk"
)


class PathConfigError(RuntimeError):
    """Raised when VGM path profile configuration is missing or unsafe."""


def package_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def activate_profile(profile: str) -> None:
    script = package_repo_root() / "scripts" / "env" / "activate_profile.sh"
    if not script.exists():
        raise PathConfigError(f"Missing profile activation script: {script}")
    command = f"source {shlex.quote(str(script))} {shlex.quote(profile)} >/dev/null && env -0"
    proc = subprocess.run(
        ["bash", "-lc", command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise PathConfigError(proc.stderr.strip() or f"Failed to activate profile {profile!r}")
    for item in proc.stdout.split("\0"):
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        if key.startswith("VGM_") or key == "PYTHONPATH":
            os.environ[key] = value


def ensure_profile(profile: str | None = None) -> None:
    if profile:
        activate_profile(profile)
    missing = [key for key in ROOT_ENV_KEYS if not os.environ.get(key)]
    if missing:
        raise PathConfigError(f"Missing VGM environment variable(s): {', '.join(missing)}.\n{ACTIVATE_HELP}")
    for key, suffix in DERIVED_ENV_DEFAULTS.items():
        os.environ.setdefault(key, str(get_dl3dv_root() / suffix))


def get_profile() -> str:
    ensure_profile()
    return os.environ["VGM_PROFILE"]


def _path_from_env(key: str) -> Path:
    ensure_profile()
    value = os.environ.get(key)
    if not value:
        raise PathConfigError(f"Missing {key}.\n{ACTIVATE_HELP}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise PathConfigError(f"{key} must be an absolute path, got: {value}")
    return path.resolve(strict=False)


def get_repo_root() -> Path:
    return _path_from_env("VGM_REPO_ROOT")


def get_dl3dv_root() -> Path:
    value = os.environ.get("VGM_DL3DV_ROOT")
    if not value:
        missing = "VGM_DL3DV_ROOT"
        raise PathConfigError(f"Missing {missing}.\n{ACTIVATE_HELP}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise PathConfigError(f"VGM_DL3DV_ROOT must be an absolute path, got: {value}")
    return path.resolve(strict=False)


def get_model_root() -> Path:
    return _path_from_env("VGM_MODEL_ROOT")


def get_output_root() -> Path:
    return _path_from_env("VGM_OUTPUT_ROOT")


def get_archives_root() -> Path:
    return _path_from_env("VGM_ARCHIVES_ROOT") if os.environ.get("VGM_ARCHIVES_ROOT") else get_dl3dv_root() / "archives"


def get_extracted_root() -> Path:
    return _path_from_env("VGM_EXTRACTED_ROOT") if os.environ.get("VGM_EXTRACTED_ROOT") else get_dl3dv_root() / "extracted"


def get_manifest_root() -> Path:
    return _path_from_env("VGM_MANIFEST_ROOT") if os.environ.get("VGM_MANIFEST_ROOT") else get_dl3dv_root() / "manifests"


def get_first_frames_root() -> Path:
    return _path_from_env("VGM_FIRST_FRAMES_ROOT") if os.environ.get("VGM_FIRST_FRAMES_ROOT") else get_dl3dv_root() / "first_frames"


def get_validation_root() -> Path:
    return _path_from_env("VGM_VALIDATION_ROOT") if os.environ.get("VGM_VALIDATION_ROOT") else get_dl3dv_root() / "validation"


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def resolve_under_root(root: Path, relpath: str | os.PathLike[str], label: str, *, allow_absolute: bool = False) -> Path:
    path = Path(relpath).expanduser()
    if path.is_absolute():
        if allow_absolute:
            return path.resolve(strict=False)
        raise PathConfigError(f"{label} must be relative to its configured root, got absolute path: {path}")
    resolved = (root / path).resolve(strict=False)
    root_resolved = root.resolve(strict=False)
    if not is_relative_to(resolved, root_resolved):
        raise PathConfigError(f"{label} escapes {root_resolved}: {relpath}")
    return resolved


def resolve_repo_path(relpath: str | os.PathLike[str]) -> Path:
    return resolve_under_root(get_repo_root(), relpath, "repo path")


def resolve_data_path(relpath: str | os.PathLike[str]) -> Path:
    return resolve_under_root(get_dl3dv_root(), relpath, "DL3DV data path")


def resolve_manifest_path(relpath: str | os.PathLike[str]) -> Path:
    return resolve_under_root(get_manifest_root(), relpath, "manifest path")


def resolve_first_frame_path(relpath: str | os.PathLike[str]) -> Path:
    return resolve_under_root(get_first_frames_root(), relpath, "first-frame path")


def resolve_model_path(relpath: str | os.PathLike[str]) -> Path:
    return resolve_under_root(get_model_root(), relpath, "model path")


def resolve_output_path(relpath: str | os.PathLike[str]) -> Path:
    return resolve_under_root(get_output_root(), relpath, "output path")


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolved_roots() -> dict[str, Path | str]:
    ensure_profile()
    return {
        "profile": os.environ["VGM_PROFILE"],
        "root": _path_from_env("VGM_ROOT"),
        "repo_root": get_repo_root(),
        "dl3dv_root": get_dl3dv_root(),
        "model_root": get_model_root(),
        "output_root": get_output_root(),
        "archives_root": get_archives_root(),
        "extracted_root": get_extracted_root(),
        "manifest_root": get_manifest_root(),
        "first_frames_root": get_first_frames_root(),
        "validation_root": get_validation_root(),
    }


def root_text() -> dict[str, str]:
    ensure_profile()
    return {
        "VGM_PROFILE": os.environ["VGM_PROFILE"],
        "VGM_ROOT": str(_path_from_env("VGM_ROOT")),
        "VGM_REPO_ROOT": str(get_repo_root()),
        "VGM_DL3DV_ROOT": str(get_dl3dv_root()),
        "VGM_MODEL_ROOT": str(get_model_root()),
        "VGM_OUTPUT_ROOT": str(get_output_root()),
        "VGM_ARCHIVES_ROOT": str(get_archives_root()),
        "VGM_EXTRACTED_ROOT": str(get_extracted_root()),
        "VGM_MANIFEST_ROOT": str(get_manifest_root()),
        "VGM_FIRST_FRAMES_ROOT": str(get_first_frames_root()),
        "VGM_VALIDATION_ROOT": str(get_validation_root()),
    }


def iter_dirs_limited(root: Path, max_depth: int = 5):
    root = root.resolve(strict=False)
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


def find_unique_model_dir(kind: str, model_root: Path | None = None) -> Path:
    root = model_root or get_model_root()
    if kind == "wan":
        candidates = [
            p
            for p in iter_dirs_limited(root, max_depth=5)
            if p.is_dir()
            and p.name == "Wan2.2-TI2V-5B"
            and (p / "Wan2.2_VAE.pth").is_file()
            and (p / "models_t5_umt5-xxl-enc-bf16.pth").is_file()
            and (p / "diffusion_pytorch_model.safetensors.index.json").is_file()
        ]
    elif kind == "vggt":
        candidates = [
            p
            for p in iter_dirs_limited(root, max_depth=5)
            if p.is_dir()
            and p.name == "VGGT-1B"
            and (p / "config.json").is_file()
            and ((p / "model.safetensors").is_file() or (p / "model.pt").is_file())
        ]
    else:
        raise ValueError(kind)
    candidates = sorted({p.resolve(strict=False) for p in candidates})
    if not candidates:
        raise FileNotFoundError(f"No {kind} model candidate found under {root}")
    if len(candidates) > 1:
        lines = "\n".join(str(p) for p in candidates)
        raise RuntimeError(f"Multiple {kind} model candidates found:\n{lines}")
    return candidates[0]


def export_env_lines(values: Mapping[str, str] | None = None) -> str:
    values = values or root_text()
    return "\n".join(f"{key}={value}" for key, value in values.items()) + "\n"
