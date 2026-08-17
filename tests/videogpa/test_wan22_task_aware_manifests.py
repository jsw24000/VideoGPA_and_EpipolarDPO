from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
T2V_SCRIPT_DIR = REPO_ROOT / "scripts" / "videogpa" / "wan22_5b_t2v"
if str(T2V_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(T2V_SCRIPT_DIR))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_pair_from_group_preserves_i2v_task_and_image_fields() -> None:
    scorer = load_module("wan22_score_preferences_test", T2V_SCRIPT_DIR / "score_preferences.py")
    cfg = {
        "project": {"task": "i2v"},
        "scoring": {
            "metric_name": "consistency_score",
            "metric_mode": "min",
            "min_score_gap": 0.05,
            "winner_score_threshold": 0.8,
            "motion_threshold": 0.001,
        },
    }
    group = {
        "group_id": "scene_a",
        "task": "i2v",
        "text_prompt": "A prompt.",
        "image_path": "first_frames/train/8K/scene_a/first_frame.png",
        "camera_motion": "orbit left",
        "videos": [
            {"seed": 1, "video_path": "candidates/scene_a/seed_1.mp4", "consistency_score": 0.1, "motion_norm": 0.2},
            {"seed": 2, "video_path": "candidates/scene_a/seed_2.mp4", "consistency_score": 0.4, "motion_norm": 0.2},
        ],
    }
    pair, reason = scorer.pair_from_group(group, cfg, fallback=False)
    assert reason is None
    assert pair["task"] == "i2v"
    assert pair["image_path"] == group["image_path"]
    assert pair["camera_motion"] == "orbit left"


def test_merge_scored_pairs_preserves_task_and_candidate_base_path(tmp_path: Path) -> None:
    merger = load_module("wan22_merge_shards_test", T2V_SCRIPT_DIR / "merge_shards.py")
    scored = tmp_path / "scored.json"
    pairs = tmp_path / "pairs.json"
    scored_out = tmp_path / "merged_scored.json"
    pairs_out = tmp_path / "merged_pairs.json"
    candidate_base = tmp_path / "external_candidates"
    write_json(
        scored,
        {
            "task": "i2v",
            "base_path": str(candidate_base),
            "metric_name": "consistency_score",
            "metric_mode": "min",
            "groups": [{"group_id": "scene_a", "videos": []}],
        },
    )
    write_json(
        pairs,
        {
            "task": "i2v",
            "base_path": str(candidate_base),
            "pairs": [{"pair_id": "p0"}, {"pair_id": "p1"}],
            "filtered": [],
        },
    )
    merger.merge_scored_pairs([scored], [pairs], scored_out, pairs_out)
    merged_pairs = json.loads(pairs_out.read_text(encoding="utf-8"))
    assert merged_pairs["task"] == "i2v"
    assert merged_pairs["base_path"] == str(candidate_base)
    assert len(merged_pairs["pairs"]) == 2
