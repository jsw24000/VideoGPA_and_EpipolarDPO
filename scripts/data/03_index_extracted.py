#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from vgm_common.paths import activate_profile, get_extracted_root, get_manifest_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Index extracted DL3DV scenes into profile manifests.")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.profile:
        activate_profile(args.profile)
    root = get_extracted_root()
    scenes = sorted(path for path in root.iterdir() if path.is_dir()) if root.exists() else []
    print(
        json.dumps(
            {
                "extracted_root": str(root),
                "manifest_root": str(get_manifest_root()),
                "scene_dir_count": len(scenes),
                "dry_run": args.dry_run,
                "resume": args.resume,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
