#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vgm_common.paths import activate_profile, get_archives_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory DL3DV archive files under VGM_ARCHIVES_ROOT.")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.profile:
        activate_profile(args.profile)
    root = get_archives_root()
    archives = sorted(root.rglob("*.zip")) if root.exists() else []
    print(json.dumps({"archives_root": str(root), "zip_count": len(archives), "dry_run": args.dry_run}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
