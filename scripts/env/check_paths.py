#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vgm_common.paths import (
    PathConfigError,
    activate_profile,
    get_dl3dv_root,
    get_model_root,
    get_output_root,
    get_profile,
    get_repo_root,
    is_relative_to,
    root_text,
)


def check_path(path: Path, strict: bool, label: str, errors: list[str], warnings: list[str]) -> None:
    if path.exists():
        return
    message = f"{label} does not exist: {path}"
    if strict:
        errors.append(message)
    else:
        warnings.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description="Print and validate resolved VGM path profile roots.")
    parser.add_argument("--profile", default=None, help="Optional profile to activate for this process.")
    parser.add_argument("--strict", action="store_true", help="Require all resolved roots to exist.")
    args = parser.parse_args()

    try:
        if args.profile:
            activate_profile(args.profile)
        profile = get_profile()
        repo_root = get_repo_root()
        dl3dv_root = get_dl3dv_root()
        model_root = get_model_root()
        output_root = get_output_root()
        values = root_text()
    except PathConfigError as exc:
        print(f"ERROR: {exc}")
        return 2

    print("Resolved VGM paths:")
    for key, value in values.items():
        print(f"{key}={value}")

    errors: list[str] = []
    warnings: list[str] = []
    if profile == "local":
        for label, path in (
            ("repo_root", repo_root),
            ("local data root", dl3dv_root),
            ("local model root", model_root),
            ("local output root", output_root),
        ):
            check_path(path, True, label, errors, warnings)
        for key, value in values.items():
            if key != "VGM_PROFILE" and value and not Path(value).is_absolute():
                errors.append(f"local profile path is not absolute: {key}={value}")
        expected_output = repo_root / "outputs"
        if output_root != expected_output.resolve(strict=False):
            errors.append(f"local output root should be {expected_output}, got {output_root}")
    elif profile == "cluster_zk":
        for label, path in (
            ("repo_root", repo_root),
            ("cluster data root", dl3dv_root),
            ("cluster model root", model_root),
            ("cluster output root", output_root),
        ):
            check_path(path, args.strict, label, errors, warnings)
        if is_relative_to(model_root, repo_root):
            errors.append(f"cluster model root must not be inside the repo: {model_root}")
        if is_relative_to(output_root, repo_root):
            errors.append(f"cluster output root must not be inside the repo: {output_root}")
    else:
        errors.append(f"Unsupported VGM_PROFILE={profile}")

    wrong_repo_output = repo_root / "outputs"
    if profile == "cluster_zk" and output_root == wrong_repo_output.resolve(strict=False):
        errors.append(f"cluster output is falling into the repo outputs directory: {output_root}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("\nStatus: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
