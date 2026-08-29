from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    read_json,
    render_source_validation_report,
    resolve_config,
    source_manifest_path,
    source_run_path,
    validate_source_manifest,
    write_json,
    write_run_config,
    write_text,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate read-only VideoGPA source candidates for Epipolar-DPO")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input-json", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--no-file-check", action="store_true", help="Skip video/image existence checks; intended only for synthetic tests")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg = resolve_config(Path(args.config), run_dir)
    source_run = source_run_path(cfg)
    input_json = Path(args.input_json).expanduser().resolve() if args.input_json else source_manifest_path(cfg)
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else run_dir / "manifests" / "source_validation.json"
    report = Path(args.report).expanduser().resolve() if args.report else run_dir / "reports" / "source_validation.md"

    payload = read_json(input_json)
    summary = validate_source_manifest(
        payload,
        cfg,
        source_run,
        require_files=not args.no_file_check,
        max_groups=args.max_groups,
    )
    write_json(output_json, summary)
    write_text(report, render_source_validation_report(summary))
    write_run_config(run_dir, cfg)
    if summary["status"] != "PASS":
        raise SystemExit(f"Source validation failed with {summary['error_count']} error(s); see {output_json}")
    print(f"Source validation PASS: {output_json}")


if __name__ == "__main__":
    main()
