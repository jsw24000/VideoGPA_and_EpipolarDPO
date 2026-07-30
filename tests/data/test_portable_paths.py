from __future__ import annotations

from pathlib import Path

import pytest

from dl3dv_conditions.common import PipelineError, resolve_asset_relpath


def test_relative_first_frame_path_resolves_under_asset_root() -> None:
    resolved = resolve_asset_relpath(Path("/tmp/new_asset_root"), "first_frames/train/8K/scene/first_frame.png")
    assert str(resolved) == "/tmp/new_asset_root/first_frames/train/8K/scene/first_frame.png"


def test_absolute_canonical_path_is_rejected() -> None:
    with pytest.raises(PipelineError):
        resolve_asset_relpath(Path("/tmp/new_asset_root"), "/data1/3DVGM_data/first_frames/x.png")
