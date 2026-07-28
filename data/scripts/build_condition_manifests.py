#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from dl3dv_conditions.common import (
    DEFAULT_SEED,
    PipelineError,
    filter_records_by_splits,
    find_project_root,
    generate_official_i2v_prompt,
    read_jsonl,
    require_relative_path,
    write_json,
    write_jsonl,
)


def build_master_record(caption_record: dict[str, Any], first_frame_record: dict[str, Any] | None, seed: int, project_root: Path) -> dict[str, Any]:
    split = caption_record["split_group"]
    uid = caption_record["scene_uid"]
    scripted = generate_official_i2v_prompt(uid, seed, project_root)
    relpath = first_frame_record.get("first_frame_relpath") if first_frame_record else None
    relpath = require_relative_path(relpath, "first_frame_relpath") if relpath else None

    record = {
        "record_version": 1,
        "scene_uid": uid,
        "scene_id": caption_record["scene_id"],
        "split": split,
        "source_subset": caption_record["source_subset"],
        "caption_source": caption_record.get("caption_source", "VideoGPA CogVLM caption"),
        "caption_source_file": caption_record["caption_source_file"],
        "caption_source_key": caption_record["caption_source_key"],
        "vlm_caption_raw": caption_record["vlm_caption_raw"],
        "vlm_caption": caption_record["vlm_caption"],
        "first_frame_relpath": relpath,
        "first_frame_sha256": first_frame_record.get("first_frame_sha256") if first_frame_record else None,
        "first_frame_width": first_frame_record.get("first_frame_width") if first_frame_record else None,
        "first_frame_height": first_frame_record.get("first_frame_height") if first_frame_record else None,
        "first_frame_size_bytes": first_frame_record.get("first_frame_size_bytes") if first_frame_record else None,
        "caption_image_dir": caption_record.get("caption_image_dir", "images_8"),
        "resolved_image_dir": first_frame_record.get("resolved_image_dir") if first_frame_record else None,
        "image_dir_fallback_used": first_frame_record.get("image_dir_fallback_used", False) if first_frame_record else False,
        "scripted_camera_seed": scripted["scripted_camera_seed"],
        "scripted_camera_motion": scripted["scripted_camera_motion"],
        "scripted_i2v_text_prompt": scripted["i2v_train_text_prompt"],
        "i2v_train_text_prompt": scripted["i2v_train_text_prompt"] if split == "train" else None,
        "t2v_train_text_prompt": caption_record["vlm_caption"] if split == "train" else None,
        "i2v_test_text_prompt": caption_record["vlm_caption"] if split == "test" else None,
        "t2v_test_text_prompt": caption_record["vlm_caption"] if split == "test" else None,
        "first_frame_missing": first_frame_record is None,
    }
    return record


def parse_splits(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    splits: list[str] = []
    for value in values:
        splits.extend(part.strip() for part in value.split(",") if part.strip())
    return splits or None


def build_master_records(
    project_root: Path,
    seed: int,
    caption_index: Path,
    first_frames_index: Path,
    splits: list[str] | None = None,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    captions = read_jsonl(caption_index)
    if not captions:
        raise PipelineError(f"Caption index is empty or missing: {caption_index}")
    captions = filter_records_by_splits(captions, splits, limit)
    first_frames = {record["scene_uid"]: record for record in read_jsonl(first_frames_index) if "scene_uid" in record}
    records = [build_master_record(caption, first_frames.get(caption["scene_uid"]), seed, project_root) for caption in captions]
    subset_order = {subset: i for i, subset in enumerate(("1K", "8K", "9K", "10K", "11K"))}
    records.sort(key=lambda item: (item["split"] != "train", subset_order.get(item["source_subset"], 99), item["scene_id"]))

    counts = Counter(record["split"] for record in records)
    by_subset = Counter(record["source_subset"] for record in records)
    missing = [record["scene_uid"] for record in records if record["first_frame_missing"]]
    stats = {
        "seed": seed,
        "splits": splits,
        "limit": limit,
        "total_records": len(records),
        "train_records": counts.get("train", 0),
        "test_records": counts.get("test", 0),
        "by_subset": dict(by_subset),
        "first_frame_records": len(first_frames),
        "missing_first_frames": len(missing),
        "missing_first_frame_scene_uids": missing[:100],
        "missing_first_frame_scene_uid_preview_count": min(len(missing), 100),
    }
    return records, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Build portable DL3DV condition master manifests.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--caption-index", default="data/manifests/caption_index.jsonl")
    parser.add_argument("--first-frames-index", default="data/manifests/first_frames.jsonl")
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    try:
        project_root = find_project_root(Path(args.project_root).resolve() if args.project_root else None)
        caption_index = (project_root / args.caption_index).resolve()
        first_frames_index = (project_root / args.first_frames_index).resolve()
        records, stats = build_master_records(
            project_root,
            args.seed,
            caption_index,
            first_frames_index,
            splits=parse_splits(args.splits),
            limit=args.limit,
        )
        train = [record for record in records if record["split"] == "train"]
        test = [record for record in records if record["split"] == "test"]
        manifest_dir = project_root / "data" / "manifests"
        write_jsonl(manifest_dir / "master_train.jsonl", train)
        write_jsonl(manifest_dir / "master_test.jsonl", test)
        write_jsonl(manifest_dir / "master_all.jsonl", records)
        write_json(project_root / "data" / "reports" / "manifest_statistics.json", stats)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return 0
    except PipelineError as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
