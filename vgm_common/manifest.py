from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .paths import resolve_data_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"JSONL item is not an object at {path}:{line_no}")
            rows.append(item)
    return rows


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def resolve_manifest_record(record: dict[str, Any], *, allow_legacy_absolute: bool = True) -> dict[str, Any]:
    resolved = dict(record)
    for rel_key, path_key in (
        ("video_relpath", "video_path"),
        ("first_frame_relpath", "first_frame_path"),
        ("scene_relpath", "scene_path"),
    ):
        value = record.get(rel_key)
        if value:
            resolved[path_key] = str(resolve_data_path(value))

    for legacy_key in ("video_path", "image_prompt", "first_frame_path"):
        value = record.get(legacy_key)
        if not value or legacy_key in resolved:
            continue
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            if allow_legacy_absolute:
                resolved[legacy_key] = str(path.resolve(strict=False))
            else:
                raise ValueError(f"Legacy absolute manifest path is disabled: {legacy_key}={value}")
        else:
            resolved[legacy_key] = str(resolve_data_path(path))
    return resolved
