from __future__ import annotations

from pathlib import Path

from inspect_official_captions import inspect_captions


def test_official_caption_split_integrity() -> None:
    project_root = Path(__file__).resolve().parents[2]
    records, stats, issues = inspect_captions(project_root)
    train_ids = {record["scene_id"] for record in records if record["split_group"] == "train"}
    test_ids = {record["scene_id"] for record in records if record["split_group"] == "test"}
    assert not (train_ids & test_ids)
    assert stats["subsets"]["1K"]["indexed_records"] == 1000
    assert stats["train_records"] == 840 + 900 + 909 + 498
    assert not [issue for issue in issues if issue["severity"] == "error"]
