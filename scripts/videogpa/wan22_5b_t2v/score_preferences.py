from __future__ import annotations

"""
Thin VideoGPA scorer wrapper for the WAN2.2 T2V smoke chain.

The core VGGT/video-processing implementation stays in VideoGPA
(`pipelines.process_video.VideoProcessor` and `metrics.consistency_score`).
This wrapper only adapts CLI paths, local model discovery, thresholding, and
smoke-only fallback pair selection.
"""

import argparse
import math
import os
import sys
from pathlib import Path

from common import read_json, resolve_config, safe_id, write_json
from vgm_common.paths import get_model_root


def add_videogpa_paths(project_root: Path) -> None:
    videogpa_root = project_root / "VideoGPA"
    for path in [videogpa_root, videogpa_root / "train"]:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def pair_from_group(group: dict, cfg: dict, fallback: bool) -> tuple[dict | None, str | None]:
    scoring = cfg["scoring"]
    task = str(group.get("task") or cfg.get("project", {}).get("task", "t2v")).lower()
    metric = scoring["metric_name"]
    mode = scoring["metric_mode"]
    min_gap = float(scoring["min_score_gap"])
    threshold = float(scoring["winner_score_threshold"])
    motion_threshold = float(scoring["motion_threshold"])
    videos = []
    reject_reasons = []
    for video in group.get("videos", []):
        score = finite_float(video.get(metric))
        motion = finite_float(video.get("motion_norm"))
        if score is None:
            reject_reasons.append("non_finite_score")
            continue
        if motion is None:
            reject_reasons.append("non_finite_motion")
            continue
        if motion < motion_threshold and not fallback:
            reject_reasons.append("motion_below_threshold")
            continue
        videos.append(video)
    if len(videos) < 2:
        return None, ",".join(sorted(set(reject_reasons))) or "less_than_two_valid_videos"

    reverse = mode == "max"
    ordered = sorted(videos, key=lambda item: float(item[metric]), reverse=reverse)
    winner = ordered[0]
    loser = ordered[-1]
    winner_score = float(winner[metric])
    loser_score = float(loser[metric])
    gap = abs(winner_score - loser_score)

    if not fallback:
        if mode == "min" and winner_score >= threshold:
            return None, "winner_threshold"
        if mode == "max" and winner_score <= threshold:
            return None, "winner_threshold"
        if gap < min_gap:
            return None, "score_gap"

    if winner.get("seed") == loser.get("seed"):
        return None, "same_seed"
    if winner.get("video_path") == loser.get("video_path"):
        return None, "same_video_path"

    pair_id = safe_id(f"{group['group_id']}__{winner.get('seed')}__vs__{loser.get('seed')}")
    pair = {
        "pair_id": pair_id,
        "group_id": group["group_id"],
        "scene_uid": group.get("scene_uid"),
        "scene_id": group.get("scene_id"),
        "prompt": group.get("text_prompt", group.get("prompt", "")),
        "text_prompt": group.get("text_prompt", group.get("prompt", "")),
        "task": task,
        "source_split": group.get("source_split", "train"),
        "source_bucket": group.get("source_bucket", "8k"),
        "winner": winner,
        "loser": loser,
        "winner_score": winner_score,
        "loser_score": loser_score,
        "score_gap": gap,
        "winner_motion_norm": float(winner.get("motion_norm", 0.0)),
        "loser_motion_norm": float(loser.get("motion_norm", 0.0)),
        "debug_fallback": fallback,
    }
    for key in ("image_path", "image_prompt", "input_image_path", "first_frame_path", "first_frame_relpath", "camera_motion"):
        if group.get(key) is not None:
            pair[key] = group[key]
    return pair, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Score WAN T2V candidates with VideoGPA VGGT components")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input-json", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--pairs-json", default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--allow-insufficient-pairs", action="store_true")
    parser.add_argument("--disable-debug-fallback", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg = resolve_config(Path(args.config), run_dir)
    task = str(cfg.get("project", {}).get("task", "t2v")).lower()
    project_root = Path(cfg["project"]["project_root"])
    add_videogpa_paths(project_root)
    vggt_path = Path(cfg["paths"]["vggt_model_path"]).resolve()
    if not vggt_path.exists():
        raise FileNotFoundError(f"VGGT model path does not exist: {vggt_path}")

    from metrics.consistency_score import Consistency_Score
    from pipelines.process_video import VideoProcessor

    input_json = Path(args.input_json).resolve() if args.input_json else run_dir / "manifests/candidate_groups.json"
    output_json = Path(args.output_json).resolve() if args.output_json else run_dir / "manifests/scored_candidates.json"
    pairs_json = Path(args.pairs_json).resolve() if args.pairs_json else run_dir / "manifests/preference_pairs.json"
    data = read_json(input_json)
    groups = data.get("groups", data if isinstance(data, list) else [])
    task = str(data.get("task") or task).lower() if isinstance(data, dict) else task
    candidate_base_path = Path(data.get("base_path", run_dir)).expanduser().resolve() if isinstance(data, dict) else run_dir
    if args.max_prompts:
        groups = groups[: args.max_prompts]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.num_shards > 1:
        groups = [group for idx, group in enumerate(groups) if idx % args.num_shards == args.shard_index]
    if not groups:
        raise RuntimeError(f"No groups found in {input_json}")

    import lpips
    import torch

    gpu_id = args.gpu_id
    if gpu_id is None:
        gpu_id = int(os.environ.get("GPU_ID", cfg["training"].get("device", 0)))
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    os.environ.setdefault("HF_HOME", str(get_model_root() / ".hf_cache"))
    lpips_net = lpips.LPIPS(net="vgg").to(device).eval()
    metrics = {"Consistency_Score": Consistency_Score(lpips_net, device=device)}
    processor = VideoProcessor(metrics=metrics, model_name=str(vggt_path), device=device)

    scored_groups = []
    for idx, group in enumerate(groups):
        scored_videos = []
        for video in group.get("videos", []):
            entry = dict(video)
            video_path = Path(entry.get("video_path", ""))
            if not video_path.is_absolute():
                video_path = candidate_base_path / video_path
            if not video_path.exists():
                entry["score_error"] = f"missing video: {video_path}"
                scored_videos.append(entry)
                continue
            result = processor.process(
                video_path=str(video_path),
                thresholds=[cfg["scoring"]["conf_threshold"]],
                num_frames=int(cfg["scoring"]["num_sampled_frames"]),
                save_visuals=False,
                out_dir=None,
            )
            metric_result = result.get(cfg["scoring"]["conf_threshold"], {})
            score = finite_float(metric_result.get("Consistency_Score"))
            motion = finite_float(metric_result.get("motion_norm"))
            if score is None or motion is None:
                entry["score_error"] = "non-finite scorer output"
            else:
                entry["consistency_score"] = score
                entry["motion_norm"] = motion
            scored_videos.append(entry)
        new_group = dict(group)
        new_group["videos"] = scored_videos
        scored_groups.append(new_group)
        print(f"Scored group {idx + 1}/{len(groups)}: {group.get('group_id')}")

    scored_payload = {
        "task": task,
        "base_path": str(candidate_base_path),
        "metric_name": cfg["scoring"]["metric_name"],
        "metric_mode": cfg["scoring"]["metric_mode"],
        "motion_threshold": cfg["scoring"]["motion_threshold"],
        "min_score_gap": cfg["scoring"]["min_score_gap"],
        "winner_score_threshold": cfg["scoring"]["winner_score_threshold"],
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "groups": scored_groups,
    }
    write_json(output_json, scored_payload)

    pairs = []
    filtered = []
    for group in scored_groups:
        pair, reason = pair_from_group(group, cfg, fallback=False)
        if pair:
            pairs.append(pair)
        else:
            filtered.append({"group_id": group.get("group_id"), "reason": reason})

    fallback_used = False
    fallback_pairs = []
    fallback_allowed = cfg["scoring"].get("smoke_fallback_if_no_pairs", False) and not args.disable_debug_fallback
    if len(pairs) < 2 and fallback_allowed:
        fallback_used = True
        for group in scored_groups:
            pair, _ = pair_from_group(group, cfg, fallback=True)
            if pair:
                fallback_pairs.append(pair)
        write_json(
            run_dir / "manifests/preference_pairs_smoke_unfiltered.json",
            {
                "debug_only": "DEBUG_ONLY_NOT_COMPARABLE",
                "task": task,
                "base_path": str(candidate_base_path),
                "pairs": fallback_pairs,
            },
        )
        pairs = fallback_pairs

    if len(pairs) < 2:
        write_json(
            pairs_json,
            {
                "task": task,
                "base_path": str(candidate_base_path),
                "pairs": pairs,
                "filtered": filtered,
                "debug_fallback_used": fallback_used,
                "status": "INSUFFICIENT_PAIRS",
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            },
        )
        if args.allow_insufficient_pairs:
            print(f"Shard produced fewer than 2 preference pairs: {len(pairs)}")
            return
        raise SystemExit("Fewer than 2 preference pairs; stopping before training")

    pair_payload = {
        "task": task,
        "base_path": str(candidate_base_path),
        "metric_name": cfg["scoring"]["metric_name"],
        "metric_mode": cfg["scoring"]["metric_mode"],
        "motion_threshold": cfg["scoring"]["motion_threshold"],
        "min_score_gap": cfg["scoring"]["min_score_gap"],
        "winner_score_threshold": cfg["scoring"]["winner_score_threshold"],
        "debug_fallback_used": fallback_used,
        "debug_only": "DEBUG_ONLY_NOT_COMPARABLE" if fallback_used else False,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "pairs": pairs,
        "filtered": filtered,
    }
    write_json(pairs_json, pair_payload)
    print(f"Wrote {len(pairs)} preference pairs to {pairs_json}")


if __name__ == "__main__":
    main()
