from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from .paths import (
    PathConfigError,
    ensure_profile,
    find_unique_model_dir,
    get_first_frames_root,
    get_manifest_root,
    get_output_root,
    get_repo_root,
    resolve_data_path,
    resolve_model_path,
    resolve_output_path,
    resolve_repo_path,
)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp.replace(path)


def _config_path(path: str | os.PathLike[str]) -> Path:
    raw = Path(path).expanduser()
    return raw.resolve(strict=False) if raw.is_absolute() else resolve_repo_path(raw)


def _legacy_manifest_relpath(value: str | None) -> str:
    if not value:
        return "manifests/videogpa_protocol/train_t2v.json"
    path = Path(value)
    if path.is_absolute():
        raise PathConfigError(f"Committed config manifest paths must be relative: {value}")
    parts = path.parts
    if len(parts) >= 2 and parts[0] == "data":
        return Path(*parts[1:]).as_posix()
    return path.as_posix()


def _legacy_output_subdir(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    path = Path(value)
    if path.is_absolute():
        raise PathConfigError(f"Committed config output paths must be relative: {value}")
    parts = path.parts
    if parts and parts[0] == "outputs":
        return Path(*parts[1:]).as_posix()
    return path.as_posix()


def _require_relative_config_path(value: object, label: str) -> str:
    if value is None or str(value).strip() == "":
        raise PathConfigError(f"Missing required relative {label}")
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        raise PathConfigError(f"Committed config {label} must be relative: {value}")
    if ".." in path.parts:
        raise PathConfigError(f"Committed config {label} must not escape its root: {value}")
    return path.as_posix()


def _resolve_model_value(value: str | None, kind: str) -> Path:
    if not value or value == "auto":
        return find_unique_model_dir(kind)
    return resolve_model_path(value)


def resolve_cli_path(value: str | os.PathLike[str] | None, *, root: Path | None = None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    base = root or get_repo_root()
    return (base / path).resolve(strict=False)


def resolve_experiment_config(
    config_path: str | os.PathLike[str],
    run_dir: str | os.PathLike[str] | None = None,
    *,
    model_path_override: str | None = None,
    vggt_model_path_override: str | None = None,
    output_subdir_override: str | None = None,
) -> dict[str, Any]:
    ensure_profile()
    path = _config_path(config_path)
    cfg = read_yaml(path)
    cfg.setdefault("experiment", {})
    cfg.setdefault("project", {})
    cfg.setdefault("paths", {})
    cfg.setdefault("model", {})
    cfg.setdefault("data", {})

    experiment = cfg["experiment"]
    if "name" not in experiment:
        experiment["name"] = cfg["project"].get("run_type") or path.stem
    default_subdir = f"videogpa/wan22_5b_t2v/{experiment['name']}"
    output_subdir = output_subdir_override or experiment.get("output_subdir")
    output_subdir = _legacy_output_subdir(output_subdir or cfg["paths"].get("output_root"), default_subdir)
    experiment["output_subdir"] = output_subdir

    cfg["project"]["project_root"] = str(get_repo_root())
    cfg["project"].setdefault("task", "t2v")

    model_cfg = cfg["model"]
    model_cfg.setdefault("model_relpath", cfg["paths"].get("wan_model_path", "wan/Wan2.2-TI2V-5B"))
    model_cfg.setdefault("vggt_model_relpath", cfg["paths"].get("vggt_model_path", "vggt/VGGT-1B"))
    model_path = (
        resolve_cli_path(model_path_override, root=resolve_model_path("."))
        if model_path_override
        else _resolve_model_value(model_cfg["model_relpath"], "wan")
    )
    vggt_path = (
        resolve_cli_path(vggt_model_path_override, root=resolve_model_path("."))
        if vggt_model_path_override
        else _resolve_model_value(model_cfg["vggt_model_relpath"], "vggt")
    )

    data_cfg = cfg["data"]
    if "manifest_relpath" not in data_cfg:
        data_cfg["manifest_relpath"] = _legacy_manifest_relpath(cfg["paths"].get("train_manifest"))
    data_cfg.setdefault("first_frames_relroot", "first_frames")
    manifest_path = resolve_data_path(data_cfg["manifest_relpath"])
    first_frames_root = resolve_data_path(data_cfg["first_frames_relroot"])
    output_root = resolve_output_path(output_subdir)

    source_cfg = cfg.get("source")
    resolved_source_paths: dict[str, str] = {}
    if isinstance(source_cfg, dict) and source_cfg.get("run_relpath") is not None:
        source_run_relpath = _require_relative_config_path(source_cfg.get("run_relpath"), "source.run_relpath")
        source_manifest_relpath = _require_relative_config_path(
            source_cfg.get("candidate_manifest_relpath", "manifests/candidate_groups.json"),
            "source.candidate_manifest_relpath",
        )
        source_cfg["run_relpath"] = source_run_relpath
        source_cfg["candidate_manifest_relpath"] = source_manifest_relpath
        source_run = resolve_output_path(source_run_relpath)
        source_candidate_manifest = (source_run / source_manifest_relpath).resolve(strict=False)
        resolved_source_paths = {
            "source_run": str(source_run),
            "source_candidate_manifest": str(source_candidate_manifest),
        }

    cfg["paths"].update(
        {
            "config_path": str(path),
            "videogpa_root": str(resolve_repo_path(cfg["paths"].get("videogpa_relpath", "VideoGPA"))),
            "wan_source_root": str(resolve_repo_path(cfg["paths"].get("wan_source_relpath", "third_party/Wan2.2"))),
            "wan_model_path": str(model_path),
            "vggt_model_path": str(vggt_path),
            "train_manifest": str(manifest_path),
            "manifest_root": str(get_manifest_root()),
            "first_frames_root": str(first_frames_root),
            "output_root": str(output_root),
            "profile_output_root": str(get_output_root()),
        }
    )
    cfg["paths"].update(resolved_source_paths)
    if run_dir is not None:
        run_path = Path(run_dir).expanduser()
        cfg["paths"]["run_dir"] = str(run_path.resolve(strict=False) if run_path.is_absolute() else resolve_output_path(run_path))
    return cfg


def write_resolved_config(run_dir: Path, cfg: dict[str, Any]) -> None:
    write_yaml(run_dir / "config_resolved.yaml", cfg)
    write_yaml(run_dir / "config" / "config_resolved.yaml", cfg)


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve a VGM experiment YAML with the active path profile.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument(
        "--print",
        choices=[
            "json",
            "output_root",
            "model_path",
            "manifest_path",
            "first_frames_root",
            "source_run",
            "source_candidate_manifest",
        ],
        default="json",
    )
    args = parser.parse_args()

    if args.profile:
        from .paths import activate_profile

        activate_profile(args.profile)
    cfg = resolve_experiment_config(args.config, args.run_dir)
    if args.print == "json":
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
    elif args.print == "output_root":
        print(cfg["paths"]["output_root"])
    elif args.print == "model_path":
        print(cfg["paths"]["wan_model_path"])
    elif args.print == "manifest_path":
        print(cfg["paths"]["train_manifest"])
    elif args.print == "first_frames_root":
        print(cfg["paths"]["first_frames_root"])
    elif args.print == "source_run":
        print(cfg["paths"]["source_run"])
    elif args.print == "source_candidate_manifest":
        print(cfg["paths"]["source_candidate_manifest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
