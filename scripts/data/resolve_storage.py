#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from dl3dv_conditions.common import (
    PipelineError,
    free_bytes,
    human_bytes,
    resolve_storage_layout,
    write_yaml,
)
from vgm_common.paths import activate_profile


def build_storage_config(layout) -> dict:
    return {
        "project_root": str(layout.project_root),
        "project_data": str(layout.project_data),
        "scratch_root": str(layout.scratch_root),
        "asset_root": str(layout.asset_root),
        "archives": str(layout.dl3dv_raw_960p),
        "extracted": str(layout.staging),
        "manifests": str(layout.manifests),
        "first_frames": str(layout.first_frames),
        "validation": str(layout.validation),
        "canonical_paths": "relative_to_vgm_dl3dv_root",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve and validate DL3DV external storage.")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scratch-root", default=None, help="External scratch root or final 3DVGM_data root.")
    parser.add_argument("--min-free-gb", type=float, default=1.0, help="Minimum required free space.")
    parser.add_argument("--project-root", default=None, help="Project root override.")
    args = parser.parse_args()

    try:
        if args.profile:
            activate_profile(args.profile)
        project_root = Path(args.project_root).resolve() if args.project_root else None
        layout = resolve_storage_layout(project_root, args.scratch_root, min_free_gb=args.min_free_gb, create=not args.dry_run)
        config_path = layout.project_root / "configs" / "data" / "storage.local.yaml"
        if not args.dry_run:
            write_yaml(config_path, build_storage_config(layout))

        print(f"project_root: {layout.project_root}")
        print(f"project_data: {layout.project_data}")
        print(f"project_free: {human_bytes(free_bytes(layout.project_root))}")
        print(f"scratch_root: {layout.scratch_root}")
        print(f"asset_root: {layout.asset_root}")
        print(f"archives: {layout.dl3dv_raw_960p}")
        print(f"extracted: {layout.staging}")
        print(f"manifests: {layout.manifests}")
        print(f"first_frames: {layout.first_frames}")
        print(f"validation: {layout.validation}")
        print(f"external_free: {human_bytes(free_bytes(layout.asset_root))}")
        print(f"wrote: {config_path if not args.dry_run else 'dry-run'}")
        return 0
    except PipelineError as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
