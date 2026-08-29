from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any
import types

from common import (
    finite_float,
    index_scored_videos,
    load_groups,
    read_json,
    resolve_config,
    resolve_video_path,
    reusable_score,
    score_identity,
    shard_groups,
    source_manifest_path,
    source_run_path,
    write_json,
    write_run_config,
)


def add_upstream_metric_path(project_root: Path) -> None:
    upstream = project_root / "Epipolar-DPO"
    text = str(upstream)
    if text not in sys.path:
        sys.path.insert(0, text)


def _ensure_package_stub(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = module
        return
    paths = list(getattr(module, "__path__", []))
    if str(path) not in paths:
        paths.insert(0, str(path))
        module.__path__ = paths  # type: ignore[attr-defined]


def _load_metric_module(module_name: str, module_path: Path):
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_evaluators(project_root: Path, cfg: dict[str, Any]):
    add_upstream_metric_path(project_root)

    metrics_root = project_root / "Epipolar-DPO" / "metrics"
    video_eval_root = metrics_root / "video_evaluation"
    _ensure_package_stub("metrics", metrics_root)
    _ensure_package_stub("metrics.video_evaluation", video_eval_root)
    _load_metric_module("metrics.video_evaluation.base", video_eval_root / "base.py")
    dynamics_module = _load_metric_module("metrics.video_evaluation.dynamics", video_eval_root / "dynamics.py")
    epipolar_module = _load_metric_module("metrics.video_evaluation.epipolar", video_eval_root / "epipolar.py")

    DynamicsEvaluator = dynamics_module.DynamicsEvaluator
    EpipolarEvaluator = epipolar_module.EpipolarEvaluator

    scoring = cfg.get("scoring", {})
    epipolar_cfg = scoring.get("epipolar", {})
    motion_cfg = scoring.get("motion", {})
    epipolar = EpipolarEvaluator.from_config(epipolar_cfg)
    motion = DynamicsEvaluator.from_config(motion_cfg)
    return epipolar, motion


def score_one_video(video: dict[str, Any], source_run: Path, epipolar: Any, motion: Any, metric_name: str) -> dict[str, Any]:
    entry = dict(video)
    try:
        video_path = resolve_video_path(entry.get("video_path", ""), source_run)
    except Exception as exc:
        entry["epipolar_valid"] = False
        entry["score_error"] = str(exc)
        return entry
    if not video_path.is_file():
        entry["epipolar_valid"] = False
        entry["score_error"] = f"missing video: {video_path}"
        return entry

    diagnostics: dict[str, Any] = {}
    try:
        epipolar_score, epipolar_details = epipolar.evaluate_video(str(video_path))
    except Exception as exc:
        epipolar_score, epipolar_details = -1.0, {"error": f"{type(exc).__name__}: {exc}"}
    diagnostics["epipolar"] = epipolar_details
    score = finite_float(epipolar_score)
    if score is None or score < 0:
        entry["epipolar_valid"] = False
        entry["score_error"] = epipolar_details.get("error", "invalid epipolar score") if isinstance(epipolar_details, dict) else "invalid epipolar score"
    else:
        entry[metric_name] = score
        entry["epipolar_valid"] = True

    try:
        if hasattr(motion, "first_frame"):
            motion.first_frame = None
        motion_score, motion_details = motion.evaluate_video(str(video_path))
    except Exception as exc:
        motion_score, motion_details = -1.0, {"error": f"{type(exc).__name__}: {exc}"}
    diagnostics["motion_dynamics"] = motion_details
    motion_value = finite_float(motion_score)
    if motion_value is not None:
        entry["motion_dynamics"] = motion_value
    else:
        entry["motion_dynamics"] = -1.0
    entry["diagnostics"] = diagnostics
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Score WAN2.2 candidates with upstream Epipolar-DPO epipolar and motion metrics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input-json", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--check-runtime-imports", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg = resolve_config(Path(args.config), run_dir)
    project_root = Path(cfg["project"]["project_root"])
    source_run = source_run_path(cfg)
    input_json = Path(args.input_json).expanduser().resolve() if args.input_json else source_manifest_path(cfg)
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else run_dir / "manifests" / "scored_candidates.json"
    metric_name = str(cfg.get("scoring", {}).get("metric_name", "epipolar_consistency"))

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.check_runtime_imports:
        epipolar, motion = build_evaluators(project_root, cfg)
        print(
            "Epipolar scoring runtime imports OK: "
            f"epipolar={epipolar.__class__.__name__}, motion={motion.__class__.__name__}"
        )
        return

    source_payload = read_json(input_json)
    groups = shard_groups(load_groups(source_payload), args.shard_index, args.num_shards, args.max_groups)
    if not groups:
        raise RuntimeError(f"No groups assigned to shard {args.shard_index}/{args.num_shards}")
    existing = index_scored_videos(read_json(output_json)) if output_json.exists() and not args.force else {}
    epipolar, motion = build_evaluators(project_root, cfg)

    scored_groups: list[dict[str, Any]] = []
    reused = 0
    scored = 0
    failed = 0
    for group_index, group in enumerate(groups):
        group_id = str(group.get("group_id"))
        new_group = dict(group)
        scored_videos = []
        for video in group.get("videos", []):
            if not isinstance(video, dict):
                continue
            cached = existing.get(score_identity(group_id, video))
            if cached is not None and reusable_score(cached, metric_name):
                entry = dict(cached)
                reused += 1
            else:
                entry = score_one_video(video, source_run, epipolar, motion, metric_name)
                scored += 1
            if entry.get("epipolar_valid") is False:
                failed += 1
            scored_videos.append(entry)
        new_group["videos"] = scored_videos
        scored_groups.append(new_group)
        print(f"Epipolar scored group {group_index + 1}/{len(groups)}: {group_id}")

    scoring = cfg.get("scoring", {})
    payload = {
        "method": "epipolar_dpo",
        "task": str(source_payload.get("task") if isinstance(source_payload, dict) else cfg.get("project", {}).get("task", "t2v")).lower(),
        "base_path": str(source_run),
        "candidate_base_path": str(source_run),
        "source_run": str(source_run),
        "source_run_relpath": cfg.get("source", {}).get("run_relpath"),
        "source_candidate_manifest": str(input_json),
        "source_candidate_manifest_relpath": cfg.get("source", {}).get("candidate_manifest_relpath"),
        "metric_name": metric_name,
        "metric_mode": scoring.get("metric_mode", "min"),
        "motion_metric_name": cfg.get("motion_filter", {}).get("metric_name", "motion_dynamics"),
        "metric_provenance": {
            "metric_impl": "Epipolar-DPO.metrics.video_evaluation.epipolar.EpipolarEvaluator",
            "motion_impl": "Epipolar-DPO.metrics.video_evaluation.dynamics.DynamicsEvaluator",
            "epipolar_config": scoring.get("epipolar", {}),
            "motion_config": scoring.get("motion", {}),
            "aggregation": "mean Sampson distance over consecutive sampled frame pairs; lower is better",
            "validity": "score is valid when epipolar_consistency is finite and non-negative",
        },
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "max_groups": args.max_groups,
        "groups": scored_groups,
        "score_summary": {
            "groups": len(scored_groups),
            "candidates": sum(len(group.get("videos", [])) for group in scored_groups),
            "scored_now": scored,
            "reused": reused,
            "invalid_or_failed": failed,
        },
    }
    write_json(output_json, payload)
    write_run_config(run_dir, cfg)
    print(f"Wrote scored candidates: {output_json}")


if __name__ == "__main__":
    main()
