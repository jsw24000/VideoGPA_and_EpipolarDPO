from __future__ import annotations

from pathlib import Path

from build_condition_manifests import build_master_record


def _caption(split: str = "train") -> dict:
    subset = "8K" if split == "train" else "1K"
    return {
        "split_group": split,
        "source_subset": subset,
        "scene_uid": f"{subset}/scene",
        "scene_id": "scene",
        "caption_source": "VideoGPA CogVLM caption",
        "caption_source_file": f"VideoGPA/dl3dv_video_captions/captions_{subset}.json",
        "caption_source_key": f"{subset}/scene/images_8",
        "vlm_caption_raw": " A room with a chair.\n",
        "vlm_caption": "A room with a chair.",
    }


def _first_frame(split: str = "train") -> dict:
    subset = "8K" if split == "train" else "1K"
    return {
        "scene_uid": f"{subset}/scene",
        "first_frame_relpath": f"first_frames/{split}/{subset}/scene/first_frame.png",
        "first_frame_sha256": "0" * 64,
        "first_frame_width": 960,
        "first_frame_height": 540,
        "first_frame_size_bytes": 123,
    }


def test_train_master_prompt_fields() -> None:
    project_root = Path(__file__).resolve().parents[2]
    record = build_master_record(_caption("train"), _first_frame("train"), 2026, project_root)
    assert record["record_version"] == 1
    assert record["split"] == "train"
    assert record["t2v_train_text_prompt"] == record["vlm_caption"]
    assert record["scripted_camera_motion"] not in record["t2v_train_text_prompt"]
    assert record["i2v_train_text_prompt"] == record["scripted_i2v_text_prompt"]
    assert not Path(record["first_frame_relpath"]).is_absolute()


def test_test_master_uses_natural_caption_for_i2v_and_t2v() -> None:
    project_root = Path(__file__).resolve().parents[2]
    record = build_master_record(_caption("test"), _first_frame("test"), 2026, project_root)
    assert record["i2v_test_text_prompt"] == record["vlm_caption"]
    assert record["t2v_test_text_prompt"] == record["vlm_caption"]
    assert record["i2v_train_text_prompt"] is None
