from __future__ import annotations

import pytest

from dl3dv_conditions.common import parse_caption_key, split_group_for_subset


def test_caption_key_parsing() -> None:
    parsed = parse_caption_key("8K/abc123/images_8")
    assert parsed == {"subset": "8K", "scene_id": "abc123", "image_dir": "images_8"}
    assert split_group_for_subset(parsed["subset"]) == "train"


def test_caption_key_rejects_wrong_image_dir() -> None:
    with pytest.raises(ValueError):
        parse_caption_key("1K/abc123/images_4")
