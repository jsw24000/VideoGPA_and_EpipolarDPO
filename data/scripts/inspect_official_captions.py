#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dl3dv_conditions.common import (
    ALL_SUBSETS,
    PipelineError,
    caption_source_file,
    find_project_root,
    parse_caption_key,
    read_json,
    scene_uid,
    split_group_for_subset,
    write_json,
    write_jsonl,
)


def _issue(issue_type: str, subset: str, key: str | None, message: str, severity: str = "warning") -> dict[str, Any]:
    return {
        "severity": severity,
        "issue_type": issue_type,
        "source_subset": subset,
        "caption_source_key": key,
        "message": message,
    }


def inspect_captions(project_root: Path, caption_dir: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    caption_dir = caption_dir or (project_root / "VideoGPA" / "dl3dv_video_captions")
    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "caption_dir": str(caption_dir.relative_to(project_root) if caption_dir.is_relative_to(project_root) else caption_dir),
        "subsets": {},
    }
    scene_to_uids: dict[str, list[str]] = defaultdict(list)
    key_counter: Counter[str] = Counter()

    for subset in ALL_SUBSETS:
        default_rel_file = caption_source_file(subset)
        path = caption_dir / f"captions_{subset}.json"
        rel_file = default_rel_file if path.resolve() == (project_root / default_rel_file).resolve() else str(path)
        subset_stats = {
            "source_file": rel_file,
            "json_items": 0,
            "indexed_records": 0,
            "empty_captions": 0,
            "invalid_keys": 0,
            "non_string_captions": 0,
            "suspicious_captions": 0,
        }
        if not path.exists():
            issues.append(_issue("missing_caption_file", subset, None, f"Missing caption file: {path}", "error"))
            stats["subsets"][subset] = subset_stats
            continue
        data = read_json(path)
        if not isinstance(data, dict):
            issues.append(_issue("invalid_json_root", subset, None, "Caption JSON root must be an object", "error"))
            stats["subsets"][subset] = subset_stats
            continue
        subset_stats["json_items"] = len(data)
        for key, raw_caption in data.items():
            key_counter[key] += 1
            try:
                parsed = parse_caption_key(key)
            except ValueError as exc:
                subset_stats["invalid_keys"] += 1
                issues.append(_issue("invalid_caption_key", subset, key, str(exc), "error"))
                continue
            if parsed["subset"] != subset:
                issues.append(
                    _issue(
                        "subset_mismatch",
                        subset,
                        key,
                        f"File captions_{subset}.json contains key for {parsed['subset']}",
                        "error",
                    )
                )
            if not isinstance(raw_caption, str):
                subset_stats["non_string_captions"] += 1
                issues.append(_issue("non_string_caption", subset, key, f"Caption type is {type(raw_caption).__name__}", "error"))
                continue
            stripped = raw_caption.strip()
            if not stripped:
                subset_stats["empty_captions"] += 1
                issues.append(_issue("empty_caption", subset, key, "Caption is empty after stripping outer whitespace", "error"))
            if "\x00" in raw_caption or "\ufffd" in raw_caption:
                subset_stats["suspicious_captions"] += 1
                issues.append(_issue("damaged_caption", subset, key, "Caption contains NUL or replacement characters"))
            if len(stripped) < 8:
                subset_stats["suspicious_captions"] += 1
                issues.append(_issue("suspicious_short_caption", subset, key, "Caption is very short"))

            uid = scene_uid(parsed["subset"], parsed["scene_id"])
            scene_to_uids[parsed["scene_id"]].append(uid)
            records.append(
                {
                    "split_group": split_group_for_subset(parsed["subset"]),
                    "source_subset": parsed["subset"],
                    "scene_uid": uid,
                    "scene_id": parsed["scene_id"],
                    "caption_source": "VideoGPA CogVLM caption",
                    "caption_source_file": rel_file,
                    "caption_source_key": key,
                    "caption_image_dir": parsed["image_dir"],
                    "vlm_caption_raw": raw_caption,
                    "vlm_caption": stripped,
                }
            )
            subset_stats["indexed_records"] += 1
        stats["subsets"][subset] = subset_stats

    for key, count in key_counter.items():
        if count > 1:
            subset = key.split("/", 1)[0]
            issues.append(_issue("duplicate_caption_key", subset, key, f"Caption key appears {count} times", "error"))

    train_scene_ids = {record["scene_id"] for record in records if record["split_group"] == "train"}
    test_scene_ids = {record["scene_id"] for record in records if record["split_group"] == "test"}
    overlap = sorted(train_scene_ids & test_scene_ids)
    for scene_id in overlap:
        issues.append(_issue("train_test_scene_overlap", "all", None, f"Scene ID appears in train and test: {scene_id}", "error"))

    duplicate_scene_ids = {scene_id: uids for scene_id, uids in scene_to_uids.items() if len(uids) > 1}
    for scene_id, uids in sorted(duplicate_scene_ids.items()):
        issues.append(_issue("duplicate_scene_id", "all", None, f"Scene ID appears multiple times: {scene_id} -> {uids}", "error"))

    subset_order = {subset: i for i, subset in enumerate(ALL_SUBSETS)}
    records.sort(key=lambda item: (subset_order[item["source_subset"]], item["scene_id"]))
    stats.update(
        {
            "total_records": len(records),
            "train_records": sum(1 for item in records if item["split_group"] == "train"),
            "test_records": sum(1 for item in records if item["split_group"] == "test"),
            "duplicate_scene_id_count": len(duplicate_scene_ids),
            "train_test_overlap_count": len(overlap),
            "issue_count": len(issues),
            "error_count": sum(1 for issue in issues if issue["severity"] == "error"),
        }
    )
    return records, stats, issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect official VideoGPA DL3DV captions and build caption_index.jsonl.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--caption-dir", default=None)
    args = parser.parse_args()

    try:
        project_root = find_project_root(Path(args.project_root).resolve() if args.project_root else None)
        caption_dir = Path(args.caption_dir).resolve() if args.caption_dir else None
        records, stats, issues = inspect_captions(project_root, caption_dir)
        write_json(project_root / "data" / "reports" / "caption_statistics.json", stats)
        write_jsonl(project_root / "data" / "reports" / "caption_issues.jsonl", issues)
        write_jsonl(project_root / "data" / "manifests" / "caption_index.jsonl", records)
        print(f"indexed_records: {stats['total_records']}")
        print(f"train_records: {stats['train_records']}")
        print(f"test_records: {stats['test_records']}")
        print(f"issues: {stats['issue_count']} errors: {stats['error_count']}")
        return 1 if stats["error_count"] else 0
    except PipelineError as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
