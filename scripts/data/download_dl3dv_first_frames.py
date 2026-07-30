#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dl3dv_conditions.common import (
    DEFAULT_SEED,
    HF_DATASET,
    IMAGE_EXTENSIONS,
    PipelineError,
    copy_stream_to_file,
    filter_records_by_splits,
    find_project_root,
    first_frame_relpath,
    free_bytes,
    human_bytes,
    image_size,
    load_storage_or_resolve,
    natural_sort_key,
    read_jsonl,
    resolve_asset_relpath,
    sha256_file,
    write_json,
    write_jsonl,
)
from vgm_common.paths import activate_profile, get_manifest_root, get_validation_root


def parse_splits(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    splits: list[str] = []
    for value in values:
        splits.extend(part.strip() for part in value.split(",") if part.strip())
    return splits or None


def check_proxy_environment(clear: bool = False) -> list[str]:
    proxy_keys = [key for key in os.environ if key.lower().endswith("_proxy") or key.lower() in {"all_proxy", "no_proxy"}]
    invalid = [key for key in proxy_keys if os.environ.get(key, "").lower().startswith("socks://")]
    if invalid and clear:
        for key in invalid:
            os.environ.pop(key, None)
        return [f"cleared invalid proxy env {key}" for key in invalid]
    if invalid:
        details = ", ".join(f"{key}={os.environ[key]}" for key in invalid)
        raise PipelineError(
            "Detected invalid socks:// proxy environment. Use socks5:// or rerun with "
            f"--clear-invalid-proxy-env to clear it for this process only. Values: {details}"
        )
    return []


def assert_hf_access(repo_id: str, token: str | bool | None):
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        info = api.dataset_info(repo_id, token=token)
        files = api.list_repo_files(repo_id, repo_type="dataset", token=token)
    except Exception as exc:
        raise PipelineError(
            f"Could not access Hugging Face dataset {repo_id}. Confirm HF login and dataset terms. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return info, set(files)


def hf_download_with_retry(
    repo_id: str,
    repo_file: str,
    local_dir: Path,
    token: str | bool | None,
    retries: int,
) -> Path:
    from huggingface_hub import hf_hub_download

    delay = 2.0
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=repo_file,
                repo_type="dataset",
                token=token,
                local_dir=local_dir,
            )
            return Path(path)
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
            time.sleep(delay)
            delay *= 2.0
    raise PipelineError(f"Failed to download {repo_file} after {retries + 1} attempt(s): {last_exc}")


def find_first_image_member(zip_path: Path, image_dir: str, allow_single_dir_fallback: bool = True) -> tuple[str, str, bool]:
    with zipfile.ZipFile(zip_path) as archive:
        candidates = []
        image_dirs = set()
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                continue
            parts = Path(name).parts
            image_dirs.update(part for part in parts if part.startswith("images_"))
            if image_dir not in parts:
                continue
            candidates.append(name)
        resolved_image_dir = image_dir
        used_fallback = False
        if not candidates and allow_single_dir_fallback and len(image_dirs) == 1:
            resolved_image_dir = next(iter(image_dirs))
            for name in archive.namelist():
                if name.endswith("/") or Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                if resolved_image_dir in Path(name).parts:
                    candidates.append(name)
            used_fallback = True
        if not candidates:
            available = ", ".join(sorted(image_dirs)) or "none"
            raise PipelineError(f"No decodable image candidate found under {image_dir} in {zip_path.name}; available image dirs: {available}")
        candidates.sort(key=lambda item: (natural_sort_key(Path(item).name), natural_sort_key(item)))
        return candidates[0], resolved_image_dir, used_fallback


def existing_record_is_valid(record: dict[str, Any], asset_root: Path) -> bool:
    relpath = record.get("first_frame_relpath")
    expected_sha = record.get("first_frame_sha256")
    if not relpath or not expected_sha:
        return False
    path = resolve_asset_relpath(asset_root, relpath)
    if not path.exists():
        return False
    try:
        image_size(path)
    except Exception:
        return False
    return sha256_file(path) == expected_sha


def extract_first_frame_from_zip(
    zip_path: Path,
    zip_member: str,
    output_path: Path,
) -> tuple[str, int, int, int]:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(zip_member) as stream:
            copy_stream_to_file(stream, output_path)
    width, height = image_size(output_path)
    digest = sha256_file(output_path)
    return digest, width, height, output_path.stat().st_size


def build_first_frame_record(
    caption_record: dict[str, Any],
    relpath: str,
    digest: str,
    width: int,
    height: int,
    size_bytes: int,
    repo_id: str,
    repo_file: str,
    zip_member: str,
    requested_image_dir: str,
    resolved_image_dir: str,
    image_dir_fallback_used: bool,
) -> dict[str, Any]:
    return {
        "record_version": 1,
        "scene_uid": caption_record["scene_uid"],
        "split_group": caption_record["split_group"],
        "source_subset": caption_record["source_subset"],
        "scene_id": caption_record["scene_id"],
        "caption_source_key": caption_record["caption_source_key"],
        "first_frame_relpath": relpath,
        "first_frame_sha256": digest,
        "first_frame_width": width,
        "first_frame_height": height,
        "first_frame_size_bytes": size_bytes,
        "hf_dataset": repo_id,
        "hf_file": repo_file,
        "zip_member": zip_member,
        "requested_image_dir": requested_image_dir,
        "resolved_image_dir": resolved_image_dir,
        "image_dir_fallback_used": image_dir_fallback_used,
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def download_first_frames(
    project_root: Path,
    scratch_root: str | None,
    splits: list[str] | None,
    limit: int | None,
    resume: bool,
    dry_run: bool,
    clear_invalid_proxy_env: bool,
    repo_id: str = HF_DATASET,
    token: str | bool | None = None,
    keep_download_cache: bool = False,
    allow_single_image_dir_fallback: bool = True,
    retries: int = 3,
    min_free_gb: float = 1.0,
) -> dict[str, Any]:
    proxy_notes = check_proxy_environment(clear_invalid_proxy_env)
    layout = load_storage_or_resolve(project_root, scratch_root, min_free_gb=min_free_gb, create=not dry_run)
    caption_index_path = get_manifest_root() / "caption_index.jsonl"
    caption_records = read_jsonl(caption_index_path)
    if not caption_records:
        raise PipelineError(f"Caption index is empty or missing: {caption_index_path}")
    selected = filter_records_by_splits(caption_records, splits, limit)

    print(f"project_root: {layout.project_root}")
    print(f"project_free: {human_bytes(free_bytes(layout.project_root))}")
    print(f"asset_root: {layout.asset_root}")
    print(f"external_free: {human_bytes(free_bytes(layout.asset_root))}")
    for note in proxy_notes:
        print(note)

    if dry_run:
        return {
            "hf_dataset": repo_id,
            "hf_dataset_sha": None,
            "asset_root": str(layout.asset_root),
            "selected_records": len(selected),
            "first_frame_records_total": 0,
            "dry_run": True,
            "resume": resume,
            "keep_download_cache": keep_download_cache,
            "allow_single_image_dir_fallback": allow_single_image_dir_fallback,
            "counts": {"dry_run_would_process": len(selected)},
            "selected_by_subset": dict(Counter(record["source_subset"] for record in selected)),
            "failure_count": 0,
            "seed_note": f"First-frame extraction is independent of seed {DEFAULT_SEED}; prompt seed is handled later.",
        }

    info, repo_files = assert_hf_access(repo_id, token)
    existing_path = get_manifest_root() / "first_frames.jsonl"
    existing = {record["scene_uid"]: record for record in read_jsonl(existing_path) if "scene_uid" in record}
    output_records = dict(existing)
    failures: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    by_subset: Counter[str] = Counter()

    for caption_record in selected:
        uid = caption_record["scene_uid"]
        subset = caption_record["source_subset"]
        scene_id = caption_record["scene_id"]
        split_group = caption_record["split_group"]
        image_dir = caption_record.get("caption_image_dir", "images_8")
        repo_file = f"{subset}/{scene_id}.zip"
        by_subset[subset] += 1

        if resume and uid in output_records and existing_record_is_valid(output_records[uid], layout.asset_root):
            counters["skipped_existing"] += 1
            continue
        if repo_file not in repo_files:
            failures.append(
                {
                    "scene_uid": uid,
                    "source_subset": subset,
                    "scene_id": scene_id,
                    "stage": "repo_file_lookup",
                    "message": f"Missing Hugging Face repo file: {repo_file}",
                }
            )
            counters["failed"] += 1
            continue
        if dry_run:
            counters["dry_run_would_download"] += 1
            continue

        status_dir = layout.staging / "first_frame_status" / subset
        status_dir.mkdir(parents=True, exist_ok=True)
        partial_status = status_dir / f"{scene_id}.partial.json"
        done_status = status_dir / f"{scene_id}.done.json"
        partial_status.write_text(json.dumps({"scene_uid": uid, "repo_file": repo_file}, ensure_ascii=False), encoding="utf-8")

        downloaded_zip: Path | None = None
        try:
            downloaded_zip = hf_download_with_retry(repo_id, repo_file, layout.download_cache, token, retries)
            zip_member, resolved_image_dir, used_fallback = find_first_image_member(
                downloaded_zip,
                image_dir,
                allow_single_dir_fallback=allow_single_image_dir_fallback,
            )
            if used_fallback:
                counters["image_dir_fallback_used"] += 1
            extension = Path(zip_member).suffix.lower()
            relpath = first_frame_relpath(split_group, subset, scene_id, extension)
            output_path = resolve_asset_relpath(layout.asset_root, relpath)
            digest, width, height, size_bytes = extract_first_frame_from_zip(downloaded_zip, zip_member, output_path)
            output_records[uid] = build_first_frame_record(
                caption_record,
                relpath,
                digest,
                width,
                height,
                size_bytes,
                repo_id,
                repo_file,
                zip_member,
                image_dir,
                resolved_image_dir,
                used_fallback,
            )
            done_status.write_text(json.dumps(output_records[uid], ensure_ascii=False), encoding="utf-8")
            partial_status.unlink(missing_ok=True)
            counters["extracted"] += 1
        except Exception as exc:
            failures.append(
                {
                    "scene_uid": uid,
                    "source_subset": subset,
                    "scene_id": scene_id,
                    "stage": "download_or_extract",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
            counters["failed"] += 1
        finally:
            if downloaded_zip and downloaded_zip.exists() and not keep_download_cache:
                try:
                    downloaded_zip.unlink()
                except OSError:
                    pass

    subset_order = {subset: i for i, subset in enumerate(("1K", "8K", "9K", "10K", "11K"))}
    ordered_records = sorted(output_records.values(), key=lambda item: (subset_order.get(item["source_subset"], 99), item["scene_id"]))
    write_jsonl(existing_path, ordered_records)
    write_jsonl(get_validation_root() / "reports" / "download_failures.jsonl", failures)
    stats = {
        "hf_dataset": repo_id,
        "hf_dataset_sha": getattr(info, "sha", None),
        "asset_root": str(layout.asset_root),
        "selected_records": len(selected),
        "first_frame_records_total": len(ordered_records),
        "dry_run": dry_run,
        "resume": resume,
        "keep_download_cache": keep_download_cache,
        "allow_single_image_dir_fallback": allow_single_image_dir_fallback,
        "counts": dict(counters),
        "selected_by_subset": dict(by_subset),
        "failure_count": len(failures),
        "seed_note": f"First-frame extraction is independent of seed {DEFAULT_SEED}; prompt seed is handled later.",
    }
    write_json(get_validation_root() / "reports" / "first_frame_statistics.json", stats)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Download DL3DV scene archives one at a time and extract first frames.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--scratch-root", default=None)
    parser.add_argument("--splits", nargs="*", default=None, help="Subsets/splits to process, e.g. 1K or 8K 9K.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clear-invalid-proxy-env", action="store_true")
    parser.add_argument("--repo-id", default=HF_DATASET)
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token. Defaults to local HF login.")
    parser.add_argument("--keep-download-cache", action="store_true")
    parser.add_argument("--no-image-dir-fallback", action="store_true", help="Fail if caption image_dir is missing instead of using a unique images_* dir.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--min-free-gb", type=float, default=1.0)
    args = parser.parse_args()

    try:
        if args.profile:
            activate_profile(args.profile)
        project_root = find_project_root(Path(args.project_root).resolve() if args.project_root else None)
        stats = download_first_frames(
            project_root=project_root,
            scratch_root=args.scratch_root,
            splits=parse_splits(args.splits),
            limit=args.limit,
            resume=args.resume,
            dry_run=args.dry_run,
            clear_invalid_proxy_env=args.clear_invalid_proxy_env,
            repo_id=args.repo_id,
            token=args.hf_token,
            keep_download_cache=args.keep_download_cache,
            allow_single_image_dir_fallback=not args.no_image_dir_fallback,
            retries=args.retries,
            min_free_gb=args.min_free_gb,
        )
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return 1 if stats["failure_count"] else 0
    except PipelineError as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
