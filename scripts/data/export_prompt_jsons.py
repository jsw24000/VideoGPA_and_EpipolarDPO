#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dl3dv_conditions.common import (
    PipelineError,
    find_project_root,
    image_size,
    read_jsonl,
    resolve_asset_relpath,
    storage_from_local_config,
    write_json,
    write_jsonl,
)


def resolve_asset_root(project_root: Path, asset_root_arg: str | None) -> Path:
    if asset_root_arg:
        return Path(asset_root_arg).expanduser().resolve()
    return storage_from_local_config(project_root).asset_root


def require_image_path(record: dict[str, Any], asset_root: Path) -> str:
    relpath = record.get("first_frame_relpath")
    if not relpath:
        raise PipelineError(f"Missing first_frame_relpath for {record['scene_uid']}")
    path = resolve_asset_relpath(asset_root, relpath)
    if not path.exists():
        raise PipelineError(f"Missing first frame for {record['scene_uid']}: {path}")
    try:
        image_size(path)
    except Exception as exc:
        raise PipelineError(f"First frame is not decodable for {record['scene_uid']}: {path}: {exc}") from exc
    return str(path)


def shared_record(record: dict[str, Any], mode: str, text_prompt: str, image_prompt: str | None = None) -> dict[str, Any]:
    item = {
        "record_version": 1,
        "protocol": "videogpa_dl3dv_conditions_v1",
        "mode": mode,
        "split": record["split"],
        "scene_uid": record["scene_uid"],
        "scene_id": record["scene_id"],
        "source_subset": record["source_subset"],
        "text_prompt": text_prompt,
        "caption_source": record["caption_source"],
        "caption_source_file": record["caption_source_file"],
        "caption_source_key": record["caption_source_key"],
        "vlm_caption": record["vlm_caption"],
        "first_frame_relpath": record.get("first_frame_relpath"),
        "first_frame_sha256": record.get("first_frame_sha256"),
    }
    if image_prompt is not None:
        item["image_prompt"] = image_prompt
    if mode == "i2v_train":
        item["camera_motion"] = record["scripted_camera_motion"]
    return item


def export_prompt_jsons(project_root: Path, asset_root: Path, master_all_path: Path, output_dir: Path) -> dict[str, Any]:
    records = read_jsonl(master_all_path)
    if not records:
        raise PipelineError(f"Master manifest is empty or missing: {master_all_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = project_root / "data" / "manifests" / "shared_protocol"
    shared_dir.mkdir(parents=True, exist_ok=True)

    train_i2v: dict[str, dict[str, Any]] = {}
    train_t2v: dict[str, dict[str, Any]] = {}
    test_i2v: dict[str, dict[str, Any]] = {}
    test_t2v: dict[str, dict[str, Any]] = {}
    shared = {
        "train_i2v": [],
        "train_t2v": [],
        "test_i2v": [],
        "test_t2v": [],
    }

    for record in records:
        uid = record["scene_uid"]
        if record["split"] == "train":
            train_t2v[uid] = {"text_prompt": record["t2v_train_text_prompt"]}
            shared["train_t2v"].append(shared_record(record, "t2v_train", record["t2v_train_text_prompt"]))
            image_prompt = require_image_path(record, asset_root)
            train_i2v[uid] = {
                "text_prompt": record["i2v_train_text_prompt"],
                "image_prompt": image_prompt,
                "camera_motion": record["scripted_camera_motion"],
            }
            shared["train_i2v"].append(shared_record(record, "i2v_train", record["i2v_train_text_prompt"], image_prompt))
        elif record["split"] == "test":
            test_t2v[uid] = {"text_prompt": record["t2v_test_text_prompt"]}
            shared["test_t2v"].append(shared_record(record, "t2v_test", record["t2v_test_text_prompt"]))
            image_prompt = require_image_path(record, asset_root)
            test_i2v[uid] = {
                "text_prompt": record["i2v_test_text_prompt"],
                "image_prompt": image_prompt,
            }
            shared["test_i2v"].append(shared_record(record, "i2v_test", record["i2v_test_text_prompt"], image_prompt))
        else:
            raise PipelineError(f"Unknown split in master manifest for {uid}: {record['split']}")

    write_json(output_dir / "train_i2v.json", train_i2v)
    write_json(output_dir / "train_t2v.json", train_t2v)
    write_json(output_dir / "test_i2v.json", test_i2v)
    write_json(output_dir / "test_t2v.json", test_t2v)
    for name, items in shared.items():
        write_jsonl(shared_dir / f"{name}.jsonl", items)

    summary = {
        "asset_root": str(asset_root),
        "videogpa_protocol_dir": str(output_dir),
        "shared_protocol_dir": str(shared_dir),
        "counts": {
            "train_i2v": len(train_i2v),
            "train_t2v": len(train_t2v),
            "test_i2v": len(test_i2v),
            "test_t2v": len(test_t2v),
        },
    }
    write_json(project_root / "data" / "reports" / "export_statistics.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Export VideoGPA-compatible prompt JSONs and shared JSONL protocol files.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--asset-root", default=None)
    parser.add_argument("--master-all", default="data/manifests/master_all.jsonl")
    parser.add_argument("--output-dir", default="data/manifests/videogpa_protocol")
    args = parser.parse_args()

    try:
        project_root = find_project_root(Path(args.project_root).resolve() if args.project_root else None)
        asset_root = resolve_asset_root(project_root, args.asset_root)
        summary = export_prompt_jsons(
            project_root=project_root,
            asset_root=asset_root,
            master_all_path=(project_root / args.master_all).resolve(),
            output_dir=(project_root / args.output_dir).resolve(),
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    except PipelineError as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
