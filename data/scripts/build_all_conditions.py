#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from build_condition_manifests import build_master_records
from cleanup_raw_data import cleanup_raw_data
from download_dl3dv_first_frames import download_first_frames, parse_splits
from export_prompt_jsons import export_prompt_jsons
from inspect_official_captions import inspect_captions
from resolve_storage import build_storage_config

from dl3dv_conditions.common import (
    DEFAULT_SEED,
    HF_DATASET,
    PipelineError,
    find_project_root,
    free_bytes,
    human_bytes,
    resolve_storage_layout,
    write_json,
    write_jsonl,
    write_yaml,
)
from validate_condition_pack import validate_condition_pack


def stage_result(name: str, status: str, details: dict[str, Any] | None = None, error: str | None = None) -> dict[str, Any]:
    return {
        "stage": name,
        "status": status,
        "details": details or {},
        "error": error,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the DL3DV condition data pack end to end.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--scratch-root", default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run-download", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--cleanup-raw", action="store_true")
    parser.add_argument("--confirm-cleanup", action="store_true")
    parser.add_argument("--clear-invalid-proxy-env", action="store_true")
    parser.add_argument("--repo-id", default=HF_DATASET)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--keep-download-cache", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--min-free-gb", type=float, default=1.0)
    args = parser.parse_args()

    project_root = find_project_root(Path(args.project_root).resolve() if args.project_root else None)
    run_report_path = project_root / "data" / "reports" / "build_all_conditions_run.json"
    log_path = project_root / "data" / "logs" / "build_all_conditions.jsonl"
    run: dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "args": vars(args),
        "stages": [],
    }
    final_status = 0
    layout = None

    def record(stage: dict[str, Any]) -> None:
        run["stages"].append(stage)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(stage, ensure_ascii=False))
            handle.write("\n")
        write_json(run_report_path, run)

    try:
        layout = resolve_storage_layout(project_root, args.scratch_root, min_free_gb=args.min_free_gb, create=True)
        write_yaml(project_root / "data" / "configs" / "storage.local.yaml", build_storage_config(layout))
        record(
            stage_result(
                "storage check",
                "ok",
                {
                    "project_root": str(layout.project_root),
                    "scratch_root": str(layout.scratch_root),
                    "asset_root": str(layout.asset_root),
                    "project_free": human_bytes(free_bytes(layout.project_root)),
                    "external_free": human_bytes(free_bytes(layout.asset_root)),
                },
            )
        )
    except Exception as exc:
        record(stage_result("storage check", "failed", error=f"{type(exc).__name__}: {exc}"))
        final_status = 2
        run["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        run["status"] = "failed"
        write_json(run_report_path, run)
        print(json.dumps(run, indent=2, ensure_ascii=False))
        return final_status

    try:
        caption_records, caption_stats, caption_issues = inspect_captions(project_root)
        write_json(project_root / "data" / "reports" / "caption_statistics.json", caption_stats)
        write_jsonl(project_root / "data" / "reports" / "caption_issues.jsonl", caption_issues)
        write_jsonl(project_root / "data" / "manifests" / "caption_index.jsonl", caption_records)
        status = "ok" if caption_stats["error_count"] == 0 else "failed"
        if status != "ok":
            final_status = 1
        record(stage_result("caption index", status, caption_stats))
    except Exception as exc:
        record(stage_result("caption index", "failed", error=f"{type(exc).__name__}: {exc}"))
        print(json.dumps(run, indent=2, ensure_ascii=False))
        return 2

    if args.skip_download:
        record(stage_result("first-frame download/extraction", "skipped", {"reason": "--skip-download"}))
    else:
        try:
            download_stats = download_first_frames(
                project_root=project_root,
                scratch_root=str(layout.asset_root),
                splits=parse_splits(args.splits),
                limit=args.limit,
                resume=args.resume,
                dry_run=args.dry_run_download,
                clear_invalid_proxy_env=args.clear_invalid_proxy_env,
                repo_id=args.repo_id,
                token=args.hf_token,
                keep_download_cache=args.keep_download_cache,
                allow_single_image_dir_fallback=True,
                retries=args.retries,
                min_free_gb=args.min_free_gb,
            )
            status = "ok" if download_stats["failure_count"] == 0 and not args.dry_run_download else "failed"
            if status != "ok":
                final_status = 1
            record(stage_result("first-frame download/extraction", status, download_stats))
        except Exception as exc:
            final_status = 1
            record(stage_result("first-frame download/extraction", "failed", error=f"{type(exc).__name__}: {exc}"))

    try:
        records, manifest_stats = build_master_records(
            project_root,
            args.seed,
            project_root / "data" / "manifests" / "caption_index.jsonl",
            project_root / "data" / "manifests" / "first_frames.jsonl",
            splits=parse_splits(args.splits),
            limit=args.limit,
        )
        train = [record for record in records if record["split"] == "train"]
        test = [record for record in records if record["split"] == "test"]
        write_jsonl(project_root / "data" / "manifests" / "master_train.jsonl", train)
        write_jsonl(project_root / "data" / "manifests" / "master_test.jsonl", test)
        write_jsonl(project_root / "data" / "manifests" / "master_all.jsonl", records)
        write_json(project_root / "data" / "reports" / "manifest_statistics.json", manifest_stats)
        record(stage_result("master manifest", "ok", manifest_stats))
    except Exception as exc:
        final_status = 1
        record(stage_result("master manifest", "failed", error=f"{type(exc).__name__}: {exc}"))

    try:
        export_stats = export_prompt_jsons(
            project_root=project_root,
            asset_root=layout.asset_root,
            master_all_path=project_root / "data" / "manifests" / "master_all.jsonl",
            output_dir=project_root / "data" / "manifests" / "videogpa_protocol",
        )
        record(stage_result("prompt JSON export", "ok", export_stats))
    except Exception as exc:
        final_status = 1
        record(stage_result("prompt JSON export", "failed", error=f"{type(exc).__name__}: {exc}"))

    try:
        validation = validate_condition_pack(project_root, layout.asset_root, splits=parse_splits(args.splits), limit=args.limit)
        status = "ok" if validation["status"] == "pass" else "failed"
        if status != "ok":
            final_status = 1
        record(stage_result("validation", status, validation))
    except Exception as exc:
        final_status = 1
        record(stage_result("validation", "failed", error=f"{type(exc).__name__}: {exc}"))

    if args.cleanup_raw:
        try:
            cleanup = cleanup_raw_data(project_root, str(layout.asset_root), dry_run=not args.confirm_cleanup, confirm_cleanup=args.confirm_cleanup)
            record(stage_result("optional raw cleanup", "ok", cleanup))
        except Exception as exc:
            final_status = 1
            record(stage_result("optional raw cleanup", "failed", error=f"{type(exc).__name__}: {exc}"))
    else:
        record(stage_result("optional raw cleanup", "skipped", {"reason": "not requested"}))

    run["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    run["status"] = "pass" if final_status == 0 else "failed"
    write_json(run_report_path, run)
    print(json.dumps({key: run[key] for key in ("status", "project_root", "stages")}, indent=2, ensure_ascii=False))
    return final_status


if __name__ == "__main__":
    raise SystemExit(main())
