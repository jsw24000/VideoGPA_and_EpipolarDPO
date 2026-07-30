#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dl3dv_conditions.common import find_project_root, resolve_storage_layout
from vgm_common.paths import activate_profile, root_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight DL3DV profile paths without touching large data.")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.profile:
        activate_profile(args.profile)
    project_root = find_project_root(Path(args.project_root).resolve() if args.project_root else None)
    layout = resolve_storage_layout(project_root, None, create=False)
    print(
        json.dumps(
            {
                "profile": root_text(),
                "layout": layout.as_dict(),
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
