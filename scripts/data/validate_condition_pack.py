#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from dl3dv_conditions.common import (
    TEST_SUBSETS,
    TRAIN_SUBSETS,
    PipelineError,
    filter_records_by_splits,
    find_project_root,
    generate_official_motion_from_seed,
    image_size,
    read_jsonl,
    resolve_asset_relpath,
    sha256_file,
    storage_from_local_config,
    validate_official_motion_structure,
    write_json,
    write_jsonl,
)


def _err(errors: list[dict[str, Any]], issue_type: str, message: str, scene_uid: str | None = None, field: str | None = None) -> None:
    errors.append({"severity": "error", "issue_type": issue_type, "scene_uid": scene_uid, "field": field, "message": message})


def _warn(warnings: list[dict[str, Any]], issue_type: str, message: str, scene_uid: str | None = None, field: str | None = None) -> None:
    warnings.append({"severity": "warning", "issue_type": issue_type, "scene_uid": scene_uid, "field": field, "message": message})


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_json_reload(project_root: Path, errors: list[dict[str, Any]]) -> None:
    json_files = [
        project_root / "data" / "reports" / "caption_statistics.json",
        project_root / "data" / "reports" / "first_frame_statistics.json",
        project_root / "data" / "reports" / "manifest_statistics.json",
        project_root / "data" / "reports" / "export_statistics.json",
        project_root / "data" / "manifests" / "videogpa_protocol" / "train_i2v.json",
        project_root / "data" / "manifests" / "videogpa_protocol" / "train_t2v.json",
        project_root / "data" / "manifests" / "videogpa_protocol" / "test_i2v.json",
        project_root / "data" / "manifests" / "videogpa_protocol" / "test_t2v.json",
    ]
    jsonl_files = [
        project_root / "data" / "manifests" / "caption_index.jsonl",
        project_root / "data" / "manifests" / "first_frames.jsonl",
        project_root / "data" / "manifests" / "master_train.jsonl",
        project_root / "data" / "manifests" / "master_test.jsonl",
        project_root / "data" / "manifests" / "master_all.jsonl",
        project_root / "data" / "manifests" / "shared_protocol" / "train_i2v.jsonl",
        project_root / "data" / "manifests" / "shared_protocol" / "train_t2v.jsonl",
        project_root / "data" / "manifests" / "shared_protocol" / "test_i2v.jsonl",
        project_root / "data" / "manifests" / "shared_protocol" / "test_t2v.jsonl",
    ]
    for path in json_files:
        if not path.exists():
            _err(errors, "missing_json_file", f"Missing JSON output: {path.relative_to(project_root)}")
            continue
        try:
            _load_json(path)
        except Exception as exc:
            _err(errors, "invalid_json_file", f"{path.relative_to(project_root)} cannot be reloaded: {exc}")
    for path in jsonl_files:
        if not path.exists():
            _err(errors, "missing_jsonl_file", f"Missing JSONL output: {path.relative_to(project_root)}")
            continue
        try:
            read_jsonl(path)
        except Exception as exc:
            _err(errors, "invalid_jsonl_file", f"{path.relative_to(project_root)} cannot be reloaded: {exc}")


def validate_exports(project_root: Path, asset_root: Path, errors: list[dict[str, Any]]) -> dict[str, int]:
    protocol_dir = project_root / "data" / "manifests" / "videogpa_protocol"
    counts: dict[str, int] = {}
    exports: dict[str, dict[str, Any]] = {}
    for name in ("train_i2v", "train_t2v", "test_i2v", "test_t2v"):
        path = protocol_dir / f"{name}.json"
        if not path.exists():
            _err(errors, "missing_export", f"Missing VideoGPA export: {path.relative_to(project_root)}")
            counts[name] = 0
            exports[name] = {}
            continue
        data = _load_json(path)
        if not isinstance(data, dict):
            _err(errors, "invalid_export_root", f"Export root must be object: {path.relative_to(project_root)}")
            data = {}
        exports[name] = data
        counts[name] = len(data)

    for name in ("train_i2v", "test_i2v"):
        for uid, record in exports[name].items():
            image_prompt = record.get("image_prompt")
            if not image_prompt:
                _err(errors, "missing_export_image_prompt", f"{name} missing image_prompt", uid)
                continue
            path = Path(image_prompt)
            if not path.is_absolute():
                _err(errors, "export_image_prompt_not_absolute", f"{name} image_prompt is not absolute: {image_prompt}", uid)
                continue
            if not path.exists():
                _err(errors, "export_image_missing", f"{name} image_prompt does not exist: {path}", uid)
                continue
            try:
                image_size(path)
            except Exception as exc:
                _err(errors, "export_image_not_decodable", f"{name} image_prompt is not decodable: {path}: {exc}", uid)

    if set(exports["test_i2v"]) != set(exports["test_t2v"]):
        _err(errors, "test_export_scene_mismatch", "test_i2v and test_t2v scene IDs differ")
    for uid in set(exports["test_i2v"]) & set(exports["test_t2v"]):
        if exports["test_i2v"][uid].get("text_prompt") != exports["test_t2v"][uid].get("text_prompt"):
            _err(errors, "test_export_caption_mismatch", "test_i2v and test_t2v text_prompt differ", uid)

    return counts


def parse_splits(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    splits: list[str] = []
    for value in values:
        splits.extend(part.strip() for part in value.split(",") if part.strip())
    return splits or None


def validate_condition_pack(project_root: Path, asset_root: Path, splits: list[str] | None = None, limit: int | None = None) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    missing_records: list[dict[str, Any]] = []
    manifest_dir = project_root / "data" / "manifests"
    caption_records_all = read_jsonl(manifest_dir / "caption_index.jsonl")
    caption_records = filter_records_by_splits(caption_records_all, splits, limit)
    first_frame_records = read_jsonl(manifest_dir / "first_frames.jsonl")
    master_records = read_jsonl(manifest_dir / "master_all.jsonl")
    master_train = read_jsonl(manifest_dir / "master_train.jsonl")
    master_test = read_jsonl(manifest_dir / "master_test.jsonl")

    caption_uids = {record["scene_uid"] for record in caption_records}
    master_uids = [record.get("scene_uid") for record in master_records]
    first_frame_by_uid = {record["scene_uid"]: record for record in first_frame_records if "scene_uid" in record}
    duplicate_uids = [uid for uid, count in Counter(master_uids).items() if uid and count > 1]
    for uid in duplicate_uids:
        _err(errors, "duplicate_scene_uid", "scene_uid appears more than once in master_all", uid)

    missing_from_master = sorted(caption_uids - set(master_uids))
    extra_in_master = sorted(set(master_uids) - caption_uids)
    for uid in missing_from_master:
        _err(errors, "caption_dropped_from_master", "Caption record is missing from master_all", uid)
    for uid in extra_in_master:
        _err(errors, "extra_master_record", "Master record has no matching caption_index record", uid)

    train_scene_ids = {record["scene_id"] for record in master_records if record.get("split") == "train"}
    test_scene_ids = {record["scene_id"] for record in master_records if record.get("split") == "test"}
    for scene_id in sorted(train_scene_ids & test_scene_ids):
        _err(errors, "train_test_scene_overlap", f"Scene ID appears in train and test: {scene_id}")

    if len(master_train) != sum(1 for record in master_records if record.get("split") == "train"):
        _err(errors, "master_train_count_mismatch", "master_train.jsonl count differs from master_all train records")
    if len(master_test) != sum(1 for record in master_records if record.get("split") == "test"):
        _err(errors, "master_test_count_mismatch", "master_test.jsonl count differs from master_all test records")

    for record in master_records:
        uid = record.get("scene_uid", "<missing>")
        subset = record.get("source_subset")
        split = record.get("split")
        if split == "train" and subset not in TRAIN_SUBSETS:
            _err(errors, "train_subset_violation", f"Train record has invalid source_subset: {subset}", uid, "source_subset")
        if split == "test" and subset not in TEST_SUBSETS:
            _err(errors, "test_subset_violation", f"Test record has invalid source_subset: {subset}", uid, "source_subset")
        if split not in {"train", "test"}:
            _err(errors, "invalid_split", f"Invalid split: {split}", uid, "split")

        caption = record.get("vlm_caption")
        if not isinstance(caption, str) or not caption:
            _err(errors, "empty_caption", "vlm_caption is empty", uid, "vlm_caption")
        if record.get("vlm_caption_raw") is None:
            _err(errors, "missing_raw_caption", "vlm_caption_raw is missing", uid, "vlm_caption_raw")

        relpath = record.get("first_frame_relpath")
        if relpath and Path(relpath).is_absolute():
            _err(errors, "canonical_path_absolute", f"first_frame_relpath must be relative: {relpath}", uid, "first_frame_relpath")
        if not relpath:
            missing = {"scene_uid": uid, "source_subset": subset, "scene_id": record.get("scene_id"), "reason": "missing_first_frame_relpath"}
            missing_records.append(missing)
            _err(errors, "missing_first_frame", "Missing first_frame_relpath", uid, "first_frame_relpath")
        else:
            image_path = resolve_asset_relpath(asset_root, relpath)
            if not image_path.exists():
                missing = {"scene_uid": uid, "source_subset": subset, "scene_id": record.get("scene_id"), "reason": "first_frame_file_missing", "path": str(image_path)}
                missing_records.append(missing)
                _err(errors, "missing_first_frame", f"First frame file is missing: {image_path}", uid, "first_frame_relpath")
            else:
                try:
                    width, height = image_size(image_path)
                    if record.get("first_frame_width") not in (width, None):
                        _err(errors, "first_frame_width_mismatch", f"Expected {record.get('first_frame_width')}, got {width}", uid)
                    if record.get("first_frame_height") not in (height, None):
                        _err(errors, "first_frame_height_mismatch", f"Expected {record.get('first_frame_height')}, got {height}", uid)
                    expected_sha = record.get("first_frame_sha256")
                    if expected_sha and sha256_file(image_path) != expected_sha:
                        _err(errors, "first_frame_sha256_mismatch", "First frame SHA256 does not match manifest", uid)
                except Exception as exc:
                    _err(errors, "first_frame_not_decodable", f"First frame is not decodable: {image_path}: {exc}", uid)

        motion = record.get("scripted_camera_motion")
        valid_motion, reason = validate_official_motion_structure(motion or "", project_root)
        if not valid_motion:
            _err(errors, "invalid_scripted_camera_motion", reason, uid, "scripted_camera_motion")
        if record.get("scripted_camera_seed") is None:
            _err(errors, "missing_scripted_camera_seed", "Missing scripted_camera_seed", uid)
        else:
            regenerated = generate_official_motion_from_seed(int(record["scripted_camera_seed"]), project_root)
            regenerated_again = generate_official_motion_from_seed(int(record["scripted_camera_seed"]), project_root)
            if regenerated != regenerated_again or regenerated != motion:
                _err(errors, "non_reproducible_scripted_motion", "Scripted camera motion cannot be reproduced from recorded seed", uid)

        if split == "train":
            t2v_prompt = record.get("t2v_train_text_prompt")
            if t2v_prompt != caption:
                _err(errors, "t2v_train_caption_changed", "T2V train prompt must equal natural VLM caption", uid)
            if motion and isinstance(t2v_prompt, str) and (motion in t2v_prompt or "Camera motion:" in t2v_prompt):
                _err(errors, "t2v_train_has_scripted_motion", "T2V train prompt contains scripted camera motion", uid)
            expected_i2v = record.get("scripted_i2v_text_prompt")
            if record.get("i2v_train_text_prompt") != expected_i2v:
                _err(errors, "i2v_train_prompt_mismatch", "I2V train prompt must equal official scripted prompt", uid)
        if split == "test":
            if record.get("i2v_test_text_prompt") != caption:
                _err(errors, "i2v_test_not_natural_caption", "I2V test prompt must equal natural VLM caption", uid)
            if record.get("t2v_test_text_prompt") != caption:
                _err(errors, "t2v_test_not_natural_caption", "T2V test prompt must equal natural VLM caption", uid)
            if record.get("i2v_test_text_prompt") != record.get("t2v_test_text_prompt"):
                _err(errors, "test_prompt_mismatch", "I2V and T2V test prompts differ", uid)

        if uid not in first_frame_by_uid:
            _warn(warnings, "no_first_frame_index_record", "No matching first_frames.jsonl record", uid)

    export_counts = validate_exports(project_root, asset_root, errors)
    validate_json_reload(project_root, errors)
    counts = Counter(record.get("split") for record in master_records)
    by_subset = Counter(record.get("source_subset") for record in master_records)
    result = {
        "status": "pass" if not errors else "fail",
        "project_root": str(project_root),
        "asset_root": str(asset_root),
        "splits": splits,
        "limit": limit,
        "caption_records_all": len(caption_records_all),
        "caption_records": len(caption_records),
        "first_frame_records": len(first_frame_records),
        "master_records": len(master_records),
        "train_records": counts.get("train", 0),
        "test_records": counts.get("test", 0),
        "by_subset": dict(by_subset),
        "export_counts": export_counts,
        "missing_first_frames": len(missing_records),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors[:200],
        "warnings": warnings[:200],
        "truncated_issue_lists": len(errors) > 200 or len(warnings) > 200,
    }
    write_json(project_root / "data" / "reports" / "final_validation.json", result)
    write_jsonl(project_root / "data" / "reports" / "missing_records.jsonl", missing_records)
    write_dataset_summary(project_root / "data" / "reports" / "dataset_summary.md", result)
    return result


def write_dataset_summary(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# DL3DV Condition Dataset Summary",
        "",
        f"- Status: {result['status']}",
        f"- Project root: `{result['project_root']}`",
        f"- Asset root: `{result['asset_root']}`",
        f"- Caption records: {result['caption_records']}",
        f"- First-frame records: {result['first_frame_records']}",
        f"- Master train records: {result['train_records']}",
        f"- Master test records: {result['test_records']}",
        f"- Missing first frames: {result['missing_first_frames']}",
        f"- Errors: {result['error_count']}",
        f"- Warnings: {result['warning_count']}",
        "",
        "## By Subset",
        "",
    ]
    for subset, count in sorted(result["by_subset"].items()):
        lines.append(f"- {subset}: {count}")
    lines.extend(["", "## Export Counts", ""])
    for name, count in sorted(result["export_counts"].items()):
        lines.append(f"- {name}: {count}")
    if result["errors"]:
        lines.extend(["", "## First Errors", ""])
        for issue in result["errors"][:20]:
            uid = issue.get("scene_uid") or "global"
            lines.append(f"- `{uid}` {issue['issue_type']}: {issue['message']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Strictly validate the DL3DV condition pack.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--asset-root", default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    try:
        project_root = find_project_root(Path(args.project_root).resolve() if args.project_root else None)
        asset_root = Path(args.asset_root).expanduser().resolve() if args.asset_root else storage_from_local_config(project_root).asset_root
        result = validate_condition_pack(project_root, asset_root, parse_splits(args.splits), args.limit)
        print(json.dumps({key: result[key] for key in ("status", "caption_records", "first_frame_records", "train_records", "test_records", "missing_first_frames", "error_count", "warning_count")}, indent=2))
        return 0 if result["status"] == "pass" else 1
    except PipelineError as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
