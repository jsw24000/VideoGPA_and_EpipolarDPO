#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from vgm_common.paths import activate_profile, get_archives_root, get_extracted_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract DL3DV archives into VGM_EXTRACTED_ROOT.")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.profile:
        activate_profile(args.profile)
    payload = {
        "archives_root": str(get_archives_root()),
        "extracted_root": str(get_extracted_root()),
        "dry_run": args.dry_run,
        "resume": args.resume,
    }
    print(json.dumps(payload, indent=2))
    if args.dry_run:
        return 0
    raise SystemExit("Real archive extraction is intentionally not run by this wrapper yet; use the first-frame pipeline or add an explicit extractor.")


if __name__ == "__main__":
    raise SystemExit(main())
