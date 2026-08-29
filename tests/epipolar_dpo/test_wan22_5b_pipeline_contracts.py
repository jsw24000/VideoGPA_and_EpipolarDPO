from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vgm_common.config import PathConfigError, resolve_experiment_config  # noqa: E402

COMMON_PATH = REPO_ROOT / "scripts" / "epipolar_dpo" / "wan22_5b" / "common.py"
MERGE_PATH = REPO_ROOT / "scripts" / "epipolar_dpo" / "wan22_5b" / "merge_shards.py"
LOSS_PATH = REPO_ROOT / "VideoGPA" / "train" / "loss.py"


def load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def profile_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "profile"
    paths = {
        "root": root,
        "repo": REPO_ROOT,
        "data": root / "data",
        "models": root / "models",
        "outputs": root / "outputs",
        "manifests": root / "data" / "manifests",
        "first_frames": root / "data" / "first_frames",
        "validation": root / "data" / "validation",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("VGM_PROFILE", "test")
    monkeypatch.setenv("VGM_ROOT", str(paths["root"]))
    monkeypatch.setenv("VGM_REPO_ROOT", str(paths["repo"]))
    monkeypatch.setenv("VGM_DL3DV_ROOT", str(paths["data"]))
    monkeypatch.setenv("VGM_MODEL_ROOT", str(paths["models"]))
    monkeypatch.setenv("VGM_OUTPUT_ROOT", str(paths["outputs"]))
    monkeypatch.setenv("VGM_MANIFEST_ROOT", str(paths["manifests"]))
    monkeypatch.setenv("VGM_FIRST_FRAMES_ROOT", str(paths["first_frames"]))
    monkeypatch.setenv("VGM_VALIDATION_ROOT", str(paths["validation"]))
    return paths


def config_text(source_run_relpath: str, *, task: str = "t2v") -> str:
    manifest = "manifests/videogpa_protocol/train_i2v.json" if task == "i2v" else "manifests/videogpa_protocol/train_t2v.json"
    condition_schema = "[encoder_hidden_states, image_latent]" if task == "i2v" else "[encoder_hidden_states]"
    return f"""
experiment:
  name: epipolar_test
  output_subdir: epipolar_dpo/wan22_5b_{task}/formal
project:
  method: epipolar_dpo
  task: {task}
source:
  run_relpath: {source_run_relpath}
  candidate_manifest_relpath: manifests/candidate_groups.json
  expected_groups: 2
  expected_candidates: 3
  expected_candidate_videos: 6
  candidate_seeds: [1001, 1002, 1003]
model:
  model_relpath: wan/Wan2.2-TI2V-5B
  vggt_model_relpath: vggt/VGGT-1B
  wan_task_key: ti2v-5B
  architecture: single_ti2v_5b
  vae_version: wan2_2
data:
  manifest_relpath: {manifest}
  first_frames_relroot: first_frames
generation:
  frame_num: 81
  size: "1280*704"
  fps: 24
scoring:
  metric_name: epipolar_consistency
  metric_mode: min
  min_score_gap: 0.5
  winner_score_threshold: 8.0
motion_filter:
  enabled: true
  metric_name: motion_dynamics
  max_motion_dynamics: 0.9
encoding:
  latent_provenance: posthoc_mp4_vae
  condition_schema: {condition_schema}
training:
  loss_strategy: epipolar_dpo
  dpo_beta: 500.0
"""


def make_video_entry(group_id: str, seed: int, *, width: int = 1280, height: int = 704) -> dict:
    return {
        "generation_id": f"seed_{seed}",
        "seed": seed,
        "video_path": f"candidates/{group_id}/seed_{seed}.mp4",
        "ffprobe": {"ok": True, "frames": 81, "width": width, "height": height, "r_frame_rate": "24/1"},
    }


def touch_source_videos(source_run: Path, group_ids: list[str]) -> None:
    for group_id in group_ids:
        for seed in (1001, 1002, 1003):
            path = source_run / "candidates" / group_id / f"seed_{seed}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic")


def test_source_run_resolves_under_output_root_and_rejects_absolute(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = profile_env(monkeypatch, tmp_path)
    cfg_path = tmp_path / "epipolar.yaml"
    cfg_path.write_text(config_text("videogpa/wan22_5b_t2v/formal/source"), encoding="utf-8")
    cfg = resolve_experiment_config(cfg_path, paths["outputs"] / "epipolar_dpo" / "target")
    assert cfg["paths"]["source_run"] == str(paths["outputs"] / "videogpa" / "wan22_5b_t2v" / "formal" / "source")
    assert cfg["paths"]["source_candidate_manifest"].endswith("source/manifests/candidate_groups.json")

    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(config_text(str(tmp_path / "absolute_source")), encoding="utf-8")
    with pytest.raises(PathConfigError):
        resolve_experiment_config(bad_path, paths["outputs"] / "target")


def test_source_video_paths_resolve_against_source_run_not_epipolar_run(tmp_path: Path) -> None:
    common = load_module("epipolar_common_source_path_test", COMMON_PATH)
    source_run = tmp_path / "outputs" / "videogpa" / "source"
    epipolar_run = tmp_path / "outputs" / "epipolar_dpo" / "target"
    epipolar_run.mkdir(parents=True)
    touch_source_videos(source_run, ["g0", "g1"])
    cfg = {
        "project": {"task": "t2v", "project_root": str(REPO_ROOT)},
        "paths": {
            "source_run": str(source_run),
            "source_candidate_manifest": str(source_run / "manifests" / "candidate_groups.json"),
            "first_frames_root": str(tmp_path / "first_frames"),
        },
        "source": {
            "run_relpath": "videogpa/source",
            "candidate_manifest_relpath": "manifests/candidate_groups.json",
            "expected_groups": 2,
            "expected_candidates": 3,
            "expected_candidate_videos": 6,
            "candidate_seeds": [1001, 1002, 1003],
        },
        "generation": {"frame_num": 81, "size": "1280*704", "fps": 24},
        "encoding": {"latent_provenance": "posthoc_mp4_vae"},
    }
    payload = {
        "task": "t2v",
        "base_path": str(source_run),
        "groups": [
            {"group_id": "g0", "text_prompt": "Prompt zero.", "task": "t2v", "videos": [make_video_entry("g0", seed) for seed in (1001, 1002, 1003)]},
            {"group_id": "g1", "text_prompt": "Prompt one.", "task": "t2v", "videos": [make_video_entry("g1", seed) for seed in (1001, 1002, 1003)]},
        ],
    }
    summary = common.validate_source_manifest(payload, cfg, source_run)
    assert summary["status"] == "PASS"
    assert summary["sampled_video_checks"][0]["resolved_video_path"] == str(source_run / "candidates" / "g0" / "seed_1001.mp4")
    assert not (epipolar_run / "candidates" / "g0" / "seed_1001.mp4").exists()


def test_i2v_source_validation_allows_actual_1248_width(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    common = load_module("epipolar_common_i2v_validation_test", COMMON_PATH)
    paths = profile_env(monkeypatch, tmp_path)
    source_run = paths["outputs"] / "videogpa" / "i2v_source"
    first_frame = paths["first_frames"] / "train" / "8K" / "g0" / "first_frame.png"
    first_frame.parent.mkdir(parents=True)
    first_frame.write_bytes(b"png")
    touch_source_videos(source_run, ["g0", "g1"])
    cfg = {
        "project": {"task": "i2v", "project_root": str(REPO_ROOT)},
        "paths": {
            "source_run": str(source_run),
            "source_candidate_manifest": str(source_run / "manifests" / "candidate_groups.json"),
            "first_frames_root": str(paths["first_frames"]),
        },
        "source": {
            "run_relpath": "videogpa/i2v_source",
            "candidate_manifest_relpath": "manifests/candidate_groups.json",
            "expected_groups": 2,
            "expected_candidates": 3,
            "expected_candidate_videos": 6,
            "candidate_seeds": [1001, 1002, 1003],
        },
        "generation": {"frame_num": 81, "size": "1280*704", "fps": 24},
        "encoding": {"latent_provenance": "posthoc_mp4_vae"},
    }
    payload = {
        "task": "i2v",
        "image_conditioned": True,
        "base_path": str(source_run),
        "groups": [
            {
                "group_id": "g0",
                "scene_id": "g0",
                "source_bucket": "8k",
                "source_split": "train",
                "text_prompt": "Prompt with camera motion.",
                "task": "i2v",
                "image_conditioned": True,
                "image_path": str(first_frame),
                "camera_motion": "orbit left",
                "videos": [make_video_entry("g0", seed, width=1248, height=704) for seed in (1001, 1002, 1003)],
            },
            {
                "group_id": "g1",
                "scene_id": "g0",
                "source_bucket": "8k",
                "source_split": "train",
                "text_prompt": "Prompt two.",
                "task": "i2v",
                "image_conditioned": True,
                "image_path": str(first_frame),
                "videos": [make_video_entry("g1", seed, width=1248, height=704) for seed in (1001, 1002, 1003)],
            },
        ],
    }
    summary = common.validate_source_manifest(payload, cfg, source_run)
    assert summary["status"] == "PASS"
    assert summary["condition_schema"] == ["encoder_hidden_states", "image_latent"]


def test_pair_selection_metric_mode_min_filters_and_propagates_t2v_metadata(tmp_path: Path) -> None:
    common = load_module("epipolar_common_pair_test", COMMON_PATH)
    source_run = tmp_path / "source"
    cfg = {
        "project": {"task": "t2v"},
        "paths": {"source_candidate_manifest": str(source_run / "manifests" / "candidate_groups.json")},
        "source": {"run_relpath": "videogpa/source", "candidate_manifest_relpath": "manifests/candidate_groups.json"},
        "scoring": {"metric_name": "epipolar_consistency", "metric_mode": "min", "min_score_gap": 0.5, "winner_score_threshold": 8.0},
        "motion_filter": {"enabled": True, "metric_name": "motion_dynamics", "max_motion_dynamics": 0.9},
        "encoding": {"latent_provenance": "posthoc_mp4_vae"},
    }

    def scored_video(group_id: str, seed: int, score: float, motion: float, *, valid: bool = True) -> dict:
        video = make_video_entry(group_id, seed)
        video["epipolar_consistency"] = score
        video["epipolar_valid"] = valid
        video["motion_dynamics"] = motion
        return video

    payload = {
        "task": "t2v",
        "groups": [
            {
                "group_id": "g0",
                "scene_uid": "8K/g0",
                "scene_id": "g0",
                "source_split": "train",
                "source_bucket": "8k",
                "text_prompt": "A room with a moving camera.",
                "task": "t2v",
                "videos": [
                    scored_video("g0", 1001, 1.0, 0.2),
                    scored_video("g0", 1002, 3.0, 0.2),
                    scored_video("g0", 1003, 2.0, 0.2),
                ],
            },
            {
                "group_id": "g1",
                "text_prompt": "Invalid metric group.",
                "task": "t2v",
                "videos": [scored_video("g1", 1001, -1.0, 0.2, valid=False), scored_video("g1", 1002, 2.0, 0.2)],
            },
            {
                "group_id": "g2",
                "text_prompt": "Motion filter group.",
                "task": "t2v",
                "videos": [scored_video("g2", 1001, 1.0, 0.95), scored_video("g2", 1002, 4.0, 0.96)],
            },
            {
                "group_id": "g3",
                "text_prompt": "Small gap group.",
                "task": "t2v",
                "videos": [scored_video("g3", 1001, 1.0, 0.2), scored_video("g3", 1002, 1.1, 0.2)],
            },
            {
                "group_id": "g4",
                "text_prompt": "Threshold group.",
                "task": "t2v",
                "videos": [scored_video("g4", 1001, 9.0, 0.2), scored_video("g4", 1002, 10.0, 0.2)],
            },
        ],
    }
    pair_payload, summary = common.select_preference_pairs(payload, cfg, source_run)
    assert pair_payload["metric_mode"] == "min"
    assert pair_payload["base_path"] == str(source_run)
    assert pair_payload["latent_provenance"] == "posthoc_mp4_vae"
    assert len(pair_payload["pairs"]) == 1
    pair = pair_payload["pairs"][0]
    assert pair["pair_id"] == "g0__seed_1001__vs__seed_1002"
    assert pair["winner"]["seed"] == 1001
    assert pair["loser"]["seed"] == 1002
    assert pair["text_prompt"] == "A room with a moving camera."
    assert summary["scored_invalid"] == 1
    assert summary["groups_removed_motion_filter"] == 1
    assert summary["groups_removed_small_gap"] == 1
    assert summary["groups_removed_winner_threshold"] == 1
    assert summary["pairs_final"] == 1


def test_pair_selection_propagates_i2v_image_metadata(tmp_path: Path) -> None:
    common = load_module("epipolar_common_i2v_pair_test", COMMON_PATH)
    source_run = tmp_path / "source"
    cfg = {
        "project": {"task": "i2v"},
        "paths": {"source_candidate_manifest": str(source_run / "manifests" / "candidate_groups.json")},
        "source": {"run_relpath": "videogpa/i2v_source", "candidate_manifest_relpath": "manifests/candidate_groups.json"},
        "scoring": {"metric_name": "epipolar_consistency", "metric_mode": "min", "min_score_gap": 0.0, "winner_score_threshold": 8.0},
        "motion_filter": {"enabled": True, "metric_name": "motion_dynamics", "max_motion_dynamics": 0.9},
        "encoding": {"latent_provenance": "posthoc_mp4_vae"},
    }
    videos = []
    for seed, score in [(1001, 0.3), (1002, 2.0)]:
        video = make_video_entry("g0", seed, width=1248)
        video["epipolar_consistency"] = score
        video["epipolar_valid"] = True
        video["motion_dynamics"] = 0.1
        videos.append(video)
    payload = {
        "task": "i2v",
        "groups": [
            {
                "group_id": "g0",
                "text_prompt": "I2V prompt.",
                "task": "i2v",
                "image_conditioned": True,
                "image_path": "first_frames/train/8K/g0/first_frame.png",
                "image_prompt": "first_frames/train/8K/g0/first_frame.png",
                "first_frame_relpath": "first_frames/train/8K/g0/first_frame.png",
                "camera_motion": "push in",
                "videos": videos,
            }
        ],
    }
    pair_payload, _summary = common.select_preference_pairs(payload, cfg, source_run)
    pair = pair_payload["pairs"][0]
    assert pair_payload["condition_schema"] == ["encoder_hidden_states", "image_latent"]
    assert pair["image_path"] == "first_frames/train/8K/g0/first_frame.png"
    assert pair["image_prompt"] == "first_frames/train/8K/g0/first_frame.png"
    assert pair["first_frame_relpath"] == "first_frames/train/8K/g0/first_frame.png"
    assert pair["camera_motion"] == "push in"
    assert pair["image_conditioned"] is True


def test_scored_shard_merge_is_deterministic(tmp_path: Path) -> None:
    merger = load_module("epipolar_merge_shards_test", MERGE_PATH)
    shard0 = tmp_path / "scored0.json"
    shard1 = tmp_path / "scored1.json"
    write_json(shard0, {"task": "t2v", "groups": [{"group_id": "g2"}, {"group_id": "g0"}], "score_summary": {"scored_now": 2, "reused": 0, "invalid_or_failed": 0}})
    write_json(shard1, {"task": "t2v", "groups": [{"group_id": "g1"}], "score_summary": {"scored_now": 1, "reused": 1, "invalid_or_failed": 1}})
    order_payload = {"groups": [{"group_id": "g0"}, {"group_id": "g1"}, {"group_id": "g2"}]}
    merged = merger.merge_scored_payloads([shard0, shard1], order_payload)
    assert [group["group_id"] for group in merged["groups"]] == ["g0", "g1", "g2"]
    assert merged["score_summary"] == {"groups": 3, "candidates": 0, "scored_now": 3, "reused": 1, "invalid_or_failed": 1}


def test_epipolar_dpo_loss_matches_upstream_formula() -> None:
    torch = pytest.importorskip("torch")
    loss_mod = load_module("videogpa_loss_epipolar_test", LOSS_PATH)
    loss_fn = loss_mod.create_loss_strategy("epipolar_dpo", beta=2.0)
    v_win = torch.tensor([[[[[0.0]]]], [[[[1.0]]]]])
    v_lose = torch.tensor([[[[[2.0]]]], [[[[3.0]]]]])
    v_win_ref = torch.zeros_like(v_win)
    v_lose_ref = torch.ones_like(v_lose)
    v_win_target = torch.ones_like(v_win)
    v_lose_target = torch.zeros_like(v_lose)

    out = loss_fn(v_win, v_lose, v_win_ref, v_lose_ref, v_win_target, v_lose_target)
    model_win_err = (v_win - v_win_target).pow(2).mean(dim=[1, 2, 3, 4])
    model_lose_err = (v_lose - v_lose_target).pow(2).mean(dim=[1, 2, 3, 4])
    ref_win_err = (v_win_ref - v_win_target).pow(2).mean(dim=[1, 2, 3, 4])
    ref_lose_err = (v_lose_ref - v_lose_target).pow(2).mean(dim=[1, 2, 3, 4])
    win_diff = model_win_err - ref_win_err
    lose_diff = model_lose_err - ref_lose_err
    inside_term = -0.5 * 2.0 * (win_diff - lose_diff)
    expected = -torch.nn.functional.logsigmoid(inside_term).mean()

    assert torch.allclose(out.loss, expected)
    assert out.metrics["half_factor"].item() == pytest.approx(0.5)
    assert out.metrics["sign"].item() == pytest.approx(-1.0)
    assert out.metrics["beta"].item() == pytest.approx(2.0)
