#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dl3dv_conditions.common import find_project_root, storage_from_local_config
from validate_condition_pack import parse_splits, validate_condition_pack
from vgm_common.paths import activate_profile, get_validation_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a DL3DV validation report under VGM_VALIDATION_ROOT.")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.profile:
        activate_profile(args.profile)
    project_root = find_project_root(Path(args.project_root).resolve() if args.project_root else None)
    layout = storage_from_local_config(project_root)
    if args.dry_run:
        print(json.dumps({"reports_dir": str(get_validation_root() / "reports"), "dry_run": True}, indent=2))
        return 0
    result = validate_condition_pack(project_root, layout.asset_root, parse_splits(args.splits), args.limit)
    print(json.dumps({key: result[key] for key in ("status", "error_count", "warning_count")}, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
