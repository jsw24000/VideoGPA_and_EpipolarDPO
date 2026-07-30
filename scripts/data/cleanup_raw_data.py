#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from dl3dv_conditions.common import (
    PipelineError,
    find_project_root,
    free_bytes,
    human_bytes,
    image_size,
    is_relative_to,
    read_jsonl,
    resolve_asset_relpath,
    sha256_file,
    storage_from_local_config,
    write_json,
)
from vgm_common.paths import activate_profile, get_manifest_root, get_validation_root


def path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file() or child.is_symlink():
            total += child.stat().st_size
    return total


def validate_recorded_first_frames(project_root: Path, asset_root: Path) -> tuple[int, list[str]]:
    records = read_jsonl(get_manifest_root() / "first_frames.jsonl")
    errors: list[str] = []
    for record in records:
        uid = record.get("scene_uid", "<missing>")
        relpath = record.get("first_frame_relpath")
        digest = record.get("first_frame_sha256")
        if not relpath or not digest:
            errors.append(f"{uid}: missing relpath or sha256")
            continue
        path = resolve_asset_relpath(asset_root, relpath)
        if not path.exists():
            errors.append(f"{uid}: missing first frame {path}")
            continue
        try:
            image_size(path)
        except Exception as exc:
            errors.append(f"{uid}: undecodable first frame {path}: {exc}")
            continue
        if sha256_file(path) != digest:
            errors.append(f"{uid}: sha256 mismatch")
    return len(records), errors


def collect_cleanup_targets(layout, first_frame_records: list[dict[str, Any]]) -> list[Path]:
    targets: list[Path] = []
    for root in (layout.dl3dv_raw_960p, layout.staging):
        if root.exists():
            targets.extend(sorted(root.iterdir()))

    completed_repo_files = {record.get("hf_file") for record in first_frame_records if record.get("hf_file")}
    for repo_file in completed_repo_files:
        candidate = layout.download_cache / repo_file
        if candidate.exists():
            targets.append(candidate)

    unique_targets: list[Path] = []
    seen: set[Path] = set()
    for target in targets:
        resolved = target.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_targets.append(target)
    return unique_targets


def assert_safe_target(layout, target: Path) -> None:
    allowed_roots = [layout.dl3dv_raw_960p.resolve(), layout.staging.resolve(), layout.download_cache.resolve()]
    resolved = target.resolve()
    if any(is_relative_to(resolved, root) for root in allowed_roots):
        if is_relative_to(resolved, layout.first_frames.resolve()):
            raise PipelineError(f"Refusing to delete first_frames path: {resolved}")
        if is_relative_to(resolved, (layout.project_root / "data").resolve()):
            raise PipelineError(f"Refusing to delete project data path: {resolved}")
        return
    raise PipelineError(f"Cleanup target is outside allowed roots: {resolved}")


def cleanup_raw_data(project_root: Path, asset_root_arg: str | None, dry_run: bool, confirm_cleanup: bool) -> dict[str, Any]:
    layout = storage_from_local_config(project_root)
    if asset_root_arg:
        asset_root = Path(asset_root_arg).expanduser().resolve()
        layout = type(layout)(
            project_root=layout.project_root,
            project_data=layout.project_data,
            scratch_root=asset_root,
            asset_root=asset_root,
            dl3dv_raw_960p=asset_root / "archives",
            first_frames=asset_root / "first_frames",
            download_cache=asset_root / "archives",
            staging=asset_root / "extracted",
            manifests=asset_root / "manifests",
            validation=asset_root / "validation",
        )
    if not dry_run and not confirm_cleanup:
        raise PipelineError("Real cleanup requires --confirm-cleanup. Dry-run is the default.")

    first_frame_records = read_jsonl(get_manifest_root() / "first_frames.jsonl")
    first_frame_count, first_frame_errors = validate_recorded_first_frames(project_root, layout.asset_root)
    if first_frame_errors:
        raise PipelineError("Refusing cleanup because recorded first frames failed validation: " + "; ".join(first_frame_errors[:20]))

    targets = collect_cleanup_targets(layout, first_frame_records)
    for target in targets:
        assert_safe_target(layout, target)

    before_free = free_bytes(layout.asset_root)
    deleted: list[dict[str, Any]] = []
    for target in targets:
        item = {"path": str(target), "size_bytes": path_size(target), "deleted": False}
        if not dry_run:
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
            item["deleted"] = True
        deleted.append(item)
    after_free = free_bytes(layout.asset_root)
    result = {
        "dry_run": dry_run,
        "asset_root": str(layout.asset_root),
        "first_frame_records_validated": first_frame_count,
        "target_count": len(targets),
        "target_size_bytes": sum(item["size_bytes"] for item in deleted),
        "free_before": human_bytes(before_free),
        "free_after": human_bytes(after_free),
        "deleted": deleted,
    }
    write_json(get_validation_root() / "reports" / "cleanup_raw_data_report.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely clean raw DL3DV staging/cache data without touching first frames.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--asset-root", default=None)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--confirm-cleanup", action="store_true")
    args = parser.parse_args()

    try:
        if args.profile:
            activate_profile(args.profile)
        project_root = find_project_root(Path(args.project_root).resolve() if args.project_root else None)
        dry_run = not args.confirm_cleanup or args.dry_run
        if args.confirm_cleanup:
            dry_run = False
        result = cleanup_raw_data(project_root, args.asset_root, dry_run, args.confirm_cleanup)
        print(json.dumps({key: result[key] for key in ("dry_run", "asset_root", "first_frame_records_validated", "target_count", "target_size_bytes", "free_before", "free_after")}, indent=2))
        return 0
    except PipelineError as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
