from __future__ import annotations

import json
from pathlib import Path

import pytest

from vgm_common.manifest import read_jsonl, resolve_manifest_record


def set_profile(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setenv("VGM_PROFILE", "local")
    monkeypatch.setenv("VGM_ROOT", str(root))
    monkeypatch.setenv("VGM_REPO_ROOT", str(root))
    monkeypatch.setenv("VGM_DL3DV_ROOT", str(root / "data"))
    monkeypatch.setenv("VGM_MODEL_ROOT", str(root / "models"))
    monkeypatch.setenv("VGM_OUTPUT_ROOT", str(root / "outputs"))
    monkeypatch.setenv("VGM_ARCHIVES_ROOT", str(root / "data" / "archives"))
    monkeypatch.setenv("VGM_EXTRACTED_ROOT", str(root / "data" / "extracted"))
    monkeypatch.setenv("VGM_MANIFEST_ROOT", str(root / "data" / "manifests"))
    monkeypatch.setenv("VGM_FIRST_FRAMES_ROOT", str(root / "data" / "first_frames"))
    monkeypatch.setenv("VGM_VALIDATION_ROOT", str(root / "data" / "validation"))


def test_manifest_record_resolves_relative_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    set_profile(monkeypatch, tmp_path)
    record = {
        "scene_uid": "8K/example",
        "video_relpath": "extracted/train/8K/example/video.mp4",
        "first_frame_relpath": "first_frames/train/8K/example/first_frame.png",
    }
    resolved = resolve_manifest_record(record)
    assert resolved["video_path"] == str(tmp_path / "data" / "extracted" / "train" / "8K" / "example" / "video.mp4")
    assert resolved["first_frame_path"] == str(tmp_path / "data" / "first_frames" / "train" / "8K" / "example" / "first_frame.png")


def test_minimal_dataloader_reads_relative_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    set_profile(monkeypatch, tmp_path)
    manifest = tmp_path / "data" / "manifests" / "tiny.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"scene_uid": "8K/a", "first_frame_relpath": "first_frames/train/8K/a/first_frame.png"}) + "\n",
        encoding="utf-8",
    )

    class TinyDataset(torch.utils.data.Dataset):
        def __init__(self, path: Path) -> None:
            self.rows = [resolve_manifest_record(row) for row in read_jsonl(path)]

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> str:
            return self.rows[idx]["first_frame_path"]

    loader = torch.utils.data.DataLoader(TinyDataset(manifest), batch_size=1)
    assert next(iter(loader))[0] == str(tmp_path / "data" / "first_frames" / "train" / "8K" / "a" / "first_frame.png")
