from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vgm_common.config import resolve_experiment_config, write_resolved_config  # noqa: E402
from vgm_common.paths import get_dl3dv_root  # noqa: E402

FIRST_FRAME_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def resolve_config(config_path: Path, run_dir: Path | None = None) -> dict[str, Any]:
    return resolve_experiment_config(config_path, run_dir)


def safe_id(value: object) -> str:
    return str(value).strip().replace("/", "__").replace("\\", "__").replace(" ", "_")


def finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def parse_size(size: object) -> tuple[int, int]:
    text = str(size)
    if "*" not in text:
        raise ValueError(f"Expected size as WIDTH*HEIGHT, got {text!r}")
    width, height = text.split("*", 1)
    return int(width), int(height)


def parse_rate(value: object) -> float | None:
    if value is None:
        return None
    text = str(value)
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            denominator = float(den)
            return float(num) / denominator if denominator else None
        except Exception:
            return None
    return finite_float(text)


def load_groups(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        groups = payload.get("groups", [])
    else:
        groups = payload
    if not isinstance(groups, list):
        raise ValueError("Expected candidate manifest groups to be a list")
    clean = []
    for idx, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError(f"Group {idx} is not an object")
        clean.append(group)
    return clean


def source_run_path(cfg: dict[str, Any]) -> Path:
    value = cfg.get("paths", {}).get("source_run")
    if not value:
        raise ValueError("Config is missing paths.source_run")
    return Path(value).expanduser().resolve()


def source_manifest_path(cfg: dict[str, Any]) -> Path:
    value = cfg.get("paths", {}).get("source_candidate_manifest")
    if not value:
        raise ValueError("Config is missing paths.source_candidate_manifest")
    return Path(value).expanduser().resolve()


def expected_candidates_per_group(cfg: dict[str, Any]) -> int | None:
    source = cfg.get("source", {})
    data = cfg.get("data", {})
    for key in ("expected_candidates_per_group", "expected_candidates", "candidates_per_prompt"):
        value = source.get(key) if isinstance(source, dict) else None
        if value is not None:
            return int(value)
    value = data.get("candidates_per_prompt") if isinstance(data, dict) else None
    return int(value) if value is not None else None


def expected_candidate_seeds(cfg: dict[str, Any]) -> set[int]:
    source = cfg.get("source", {})
    data = cfg.get("data", {})
    seeds = source.get("candidate_seeds") if isinstance(source, dict) else None
    if seeds is None and isinstance(data, dict):
        seeds = data.get("candidate_seeds")
    return {int(seed) for seed in (seeds or [])}


def expected_group_count(cfg: dict[str, Any]) -> int | None:
    source = cfg.get("source", {})
    formal = cfg.get("formal_requirements", {})
    value = source.get("expected_groups") if isinstance(source, dict) else None
    if value is None and isinstance(formal, dict):
        value = formal.get("expected_candidate_groups") or formal.get("expected_train_prompts")
    return int(value) if value is not None else None


def expected_candidate_video_count(cfg: dict[str, Any]) -> int | None:
    source = cfg.get("source", {})
    formal = cfg.get("formal_requirements", {})
    value = source.get("expected_candidate_videos") if isinstance(source, dict) else None
    if value is None and isinstance(formal, dict):
        value = formal.get("expected_candidate_videos")
    return int(value) if value is not None else None


def is_safe_relative_path(value: object) -> bool:
    path = Path(str(value)).expanduser()
    return not path.is_absolute() and ".." not in path.parts and str(value).strip() != ""


def resolve_video_path(video_value: object, source_run: Path) -> Path:
    if not is_safe_relative_path(video_value):
        raise ValueError(f"candidate video_path must be relative to source run: {video_value}")
    return (source_run / Path(str(video_value))).resolve(strict=False)


def find_first_frame(first_frames_root: Path, split: str, bucket: str, scene_id: str) -> Path | None:
    scene_dir = first_frames_root / split / bucket.upper() / scene_id
    for ext in FIRST_FRAME_EXTENSIONS:
        candidate = scene_dir / f"first_frame{ext}"
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_image_path(group: dict[str, Any], cfg: dict[str, Any], source_run: Path) -> tuple[Path | None, list[str]]:
    project_root = Path(cfg["project"]["project_root"])
    data_root = get_dl3dv_root()
    first_frames_root = Path(cfg["paths"]["first_frames_root"])
    values = [
        group.get("image_path"),
        group.get("image_prompt"),
        group.get("input_image_path"),
        group.get("first_frame_path"),
        group.get("first_frame_relpath"),
    ]
    candidates: list[Path] = []

    def add_candidate(path: Path) -> None:
        expanded = path.expanduser()
        if expanded not in candidates:
            candidates.append(expanded)

    for value in values:
        if not value:
            continue
        raw = Path(str(value)).expanduser()
        if raw.is_absolute():
            add_candidate(raw)
        else:
            add_candidate(source_run / raw)
            add_candidate(project_root / raw)
            add_candidate(data_root / raw)
            add_candidate(first_frames_root / raw)
        parts = raw.parts
        if "first_frames" in parts:
            idx = parts.index("first_frames")
            tail = Path(*parts[idx + 1 :])
            add_candidate(first_frames_root / tail)
            add_candidate(data_root / "first_frames" / tail)
        elif parts and parts[0] in {"train", "test"}:
            add_candidate(first_frames_root / raw)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve(), [str(path) for path in candidates]

    scene_id = group.get("scene_id")
    bucket = group.get("source_bucket")
    split = group.get("source_split", "train")
    if scene_id and bucket:
        found = find_first_frame(first_frames_root, str(split), str(bucket), str(scene_id))
        if found is not None:
            return found, [str(path) for path in candidates]
    return None, [str(path) for path in candidates]


def prompt_from_group(group: dict[str, Any]) -> str:
    return str(group.get("text_prompt", group.get("prompt", ""))).strip()


def ffprobe_field(meta: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in meta:
            return meta[key]
    return None


def validate_video_entry(
    *,
    group: dict[str, Any],
    video: dict[str, Any],
    cfg: dict[str, Any],
    source_run: Path,
    require_files: bool,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    resolved_video: Path | None = None
    try:
        resolved_video = resolve_video_path(video.get("video_path", ""), source_run)
    except ValueError as exc:
        errors.append(str(exc))
    if resolved_video is not None and require_files and not resolved_video.is_file():
        errors.append(f"missing video file: {resolved_video}")

    meta = video.get("ffprobe")
    if not isinstance(meta, dict):
        errors.append("missing ffprobe metadata")
        meta = {}
    if meta.get("ok") is not True:
        errors.append(f"ffprobe not ok: {meta.get('error', 'unknown')}")

    expected_frames = int(cfg.get("generation", {}).get("frame_num", 81))
    frames = ffprobe_field(meta, "frames", "nb_read_frames", "frame_count")
    if int(frames or 0) != expected_frames:
        errors.append(f"frame count {frames} != expected {expected_frames}")

    width = int(ffprobe_field(meta, "width") or 0)
    height = int(ffprobe_field(meta, "height") or 0)
    task = str(cfg.get("project", {}).get("task", "t2v")).lower()
    if task == "t2v":
        expected_width, expected_height = parse_size(cfg.get("generation", {}).get("size", "1280*704"))
        if width != expected_width or height != expected_height:
            errors.append(f"resolution {width}x{height} != expected {expected_width}x{expected_height}")
    else:
        multiple = int(cfg.get("source_validation", {}).get("vae_spatial_multiple", 16))
        if width <= 0 or height <= 0:
            errors.append(f"invalid resolution {width}x{height}")
        elif width % multiple != 0 or height % multiple != 0:
            errors.append(f"resolution {width}x{height} is not divisible by {multiple}")

    expected_fps = finite_float(cfg.get("generation", {}).get("fps", 24))
    fps = parse_rate(ffprobe_field(meta, "r_frame_rate", "avg_frame_rate")) or finite_float(video.get("fps"))
    if expected_fps is not None and (fps is None or abs(float(fps) - expected_fps) > 0.01):
        errors.append(f"fps {fps} != expected {expected_fps}")

    result = {
        "group_id": group.get("group_id"),
        "generation_id": video.get("generation_id"),
        "seed": video.get("seed"),
        "video_path": video.get("video_path"),
        "resolved_video_path": str(resolved_video) if resolved_video is not None else None,
        "ffprobe": meta,
    }
    return result, errors


def validate_source_manifest(
    payload: Any,
    cfg: dict[str, Any],
    source_run: Path,
    *,
    require_files: bool = True,
    max_groups: int | None = None,
    max_errors: int = 50,
) -> dict[str, Any]:
    groups = load_groups(payload)
    task = str(cfg.get("project", {}).get("task", "t2v")).lower()
    expected_groups = expected_group_count(cfg)
    expected_total = expected_candidate_video_count(cfg)
    expected_per_group = expected_candidates_per_group(cfg)
    expected_seeds = expected_candidate_seeds(cfg)
    total_candidates = sum(len(group.get("videos", [])) for group in groups)
    selected_groups = groups[:max_groups] if max_groups is not None else groups

    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    group_ids = [str(group.get("group_id", "")).strip() for group in groups]
    empty_group_ids = [idx for idx, value in enumerate(group_ids) if not value]
    duplicate_ids = [group_id for group_id, count in Counter(group_ids).items() if group_id and count > 1]
    if empty_group_ids:
        errors.append({"scope": "manifest", "reason": f"empty group_id at indices {empty_group_ids[:8]}"})
    if duplicate_ids:
        errors.append({"scope": "manifest", "reason": f"duplicate group_id values {duplicate_ids[:8]}"})
    if expected_groups is not None and len(groups) != expected_groups:
        errors.append({"scope": "manifest", "reason": f"group count {len(groups)} != expected {expected_groups}"})
    if expected_total is not None and total_candidates != expected_total:
        errors.append({"scope": "manifest", "reason": f"candidate count {total_candidates} != expected {expected_total}"})

    declared_base = payload.get("base_path") if isinstance(payload, dict) else None
    if declared_base and Path(str(declared_base)).expanduser().resolve(strict=False) != source_run.resolve(strict=False):
        warnings.append("candidate manifest base_path differs from configured source_run; configured source_run is used")

    inspected: list[dict[str, Any]] = []
    for group_index, group in enumerate(selected_groups):
        group_id = str(group.get("group_id", "")).strip()
        group_task = str(group.get("task") or (payload.get("task") if isinstance(payload, dict) else "") or task).lower()
        if group_task != task:
            errors.append({"group_id": group_id, "reason": f"group task {group_task!r} != expected {task!r}"})
        if not prompt_from_group(group):
            errors.append({"group_id": group_id, "reason": "empty prompt/text_prompt"})
        videos = group.get("videos", [])
        if not isinstance(videos, list):
            errors.append({"group_id": group_id, "reason": "videos is not a list"})
            continue
        if expected_per_group is not None and len(videos) != expected_per_group:
            errors.append({"group_id": group_id, "reason": f"video count {len(videos)} != expected {expected_per_group}"})
        seeds = [video.get("seed") for video in videos if isinstance(video, dict)]
        if len(seeds) != len(set(seeds)):
            errors.append({"group_id": group_id, "reason": "duplicate seeds in group"})
        if expected_seeds:
            bad = sorted({int(seed) for seed in seeds if seed is not None} - expected_seeds)
            if bad:
                errors.append({"group_id": group_id, "reason": f"unexpected seeds {bad}"})

        if task == "i2v":
            image_conditioned = group.get("image_conditioned")
            payload_conditioned = payload.get("image_conditioned") if isinstance(payload, dict) else None
            if image_conditioned is not True and payload_conditioned is not True:
                errors.append({"group_id": group_id, "reason": "I2V group is not marked image_conditioned=true"})
            image_path, tried = resolve_image_path(group, cfg, source_run)
            if image_path is None:
                errors.append({"group_id": group_id, "reason": "unresolved image/first-frame path", "tried": tried[:8]})

        for video in videos:
            if not isinstance(video, dict):
                errors.append({"group_id": group_id, "reason": "video entry is not an object"})
                continue
            entry, entry_errors = validate_video_entry(
                group=group,
                video=video,
                cfg=cfg,
                source_run=source_run,
                require_files=require_files,
            )
            inspected.append(entry)
            for reason in entry_errors:
                errors.append({"group_id": group_id, "video_path": video.get("video_path"), "reason": reason})
            if len(errors) >= max_errors:
                break
        if len(errors) >= max_errors:
            errors.append({"scope": "manifest", "reason": f"stopped after {max_errors} errors"})
            break

    condition_schema = ["encoder_hidden_states"] if task == "t2v" else ["encoder_hidden_states", "image_latent"]
    return {
        "status": "PASS" if not errors else "FAIL",
        "method": "epipolar_dpo",
        "task": task,
        "source_run": str(source_run),
        "source_run_relpath": cfg.get("source", {}).get("run_relpath"),
        "source_candidate_manifest": str(source_manifest_path(cfg)),
        "source_candidate_manifest_relpath": cfg.get("source", {}).get("candidate_manifest_relpath"),
        "source_manifest_base_path_declared": declared_base,
        "base_path_semantics": "candidate video_path is resolved relative to source_run, not epipolar run",
        "groups_total": len(groups),
        "groups_inspected": len(selected_groups),
        "detail_scope": "subset" if max_groups is not None else "full",
        "candidates_total": total_candidates,
        "expected_groups": expected_groups,
        "expected_candidates_per_group": expected_per_group,
        "expected_candidate_videos": expected_total,
        "candidate_seeds": sorted(expected_seeds),
        "condition_schema": condition_schema,
        "latent_provenance": cfg.get("encoding", {}).get("latent_provenance", "posthoc_mp4_vae"),
        "sampled_video_checks": inspected[:10],
        "warning_count": len(warnings),
        "warnings": warnings,
        "error_count": len(errors),
        "errors": errors,
    }


def render_source_validation_report(summary: dict[str, Any]) -> str:
    lines = [
        f"# Epipolar-DPO Source Validation ({summary['task'].upper()})",
        "",
        f"Status: {summary['status']}",
        "",
        f"- source_run: `{summary['source_run']}`",
        f"- source_candidate_manifest: `{summary['source_candidate_manifest']}`",
        f"- base_path_semantics: `{summary['base_path_semantics']}`",
        f"- groups_total: `{summary['groups_total']}`",
        f"- groups_inspected: `{summary['groups_inspected']}`",
        f"- candidates_total: `{summary['candidates_total']}`",
        f"- latent_provenance: `{summary['latent_provenance']}`",
        f"- condition_schema: `{summary['condition_schema']}`",
        f"- warning_count: `{summary['warning_count']}`",
        f"- error_count: `{summary['error_count']}`",
        "",
    ]
    if summary["warnings"]:
        lines.append("## Warnings")
        lines.extend(f"- {warning}" for warning in summary["warnings"])
        lines.append("")
    if summary["errors"]:
        lines.append("## Errors")
        for error in summary["errors"][:25]:
            lines.append(f"- `{error}`")
        lines.append("")
    return "\n".join(lines)


def score_identity(group_id: str, video: dict[str, Any]) -> str:
    return "::".join(
        [
            group_id,
            str(video.get("generation_id", "")),
            str(video.get("seed", "")),
            str(video.get("video_path", "")),
        ]
    )


def index_scored_videos(payload: Any) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    if not payload:
        return index
    try:
        groups = load_groups(payload)
    except Exception:
        return index
    for group in groups:
        group_id = str(group.get("group_id", ""))
        for video in group.get("videos", []):
            if isinstance(video, dict):
                index[score_identity(group_id, video)] = video
    return index


def reusable_score(entry: dict[str, Any], metric_name: str) -> bool:
    if finite_float(entry.get(metric_name)) is not None:
        return True
    return "score_error" in entry or entry.get("epipolar_valid") is False


def shard_groups(groups: list[dict[str, Any]], shard_index: int, num_shards: int, max_groups: int | None = None) -> list[dict[str, Any]]:
    selected = groups[:max_groups] if max_groups is not None else groups
    if num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    return [group for idx, group in enumerate(selected) if idx % num_shards == shard_index]


def group_order(payload: Any) -> dict[str, int]:
    try:
        groups = load_groups(payload)
    except Exception:
        return {}
    return {str(group.get("group_id")): idx for idx, group in enumerate(groups) if group.get("group_id") is not None}


def sort_groups(groups: list[dict[str, Any]], order: dict[str, int]) -> list[dict[str, Any]]:
    if not order:
        return groups
    return sorted(groups, key=lambda item: order.get(str(item.get("group_id")), 10**12))


def motion_filter_pass(video: dict[str, Any], cfg: dict[str, Any]) -> tuple[bool, str | None]:
    motion_cfg = cfg.get("motion_filter", {})
    if not motion_cfg.get("enabled", True):
        return True, None
    metric_name = str(motion_cfg.get("metric_name", "motion_dynamics"))
    value = finite_float(video.get(metric_name))
    if value is None or value < 0:
        return False, f"missing_or_non_finite_{metric_name}"
    min_value = motion_cfg.get("min_motion_dynamics")
    max_value = motion_cfg.get("max_motion_dynamics")
    if min_value is not None and value < float(min_value):
        return False, f"{metric_name}_below_min"
    if max_value is not None and value > float(max_value):
        return False, f"{metric_name}_above_max"
    return True, None


def candidate_metric_valid(video: dict[str, Any], metric_name: str) -> tuple[bool, str | None]:
    if video.get("epipolar_valid") is False:
        return False, "epipolar_invalid"
    value = finite_float(video.get(metric_name))
    if value is None or value < 0:
        return False, f"invalid_{metric_name}"
    return True, None


def build_pair_from_group(group: dict[str, Any], cfg: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    scoring = cfg.get("scoring", {})
    metric_name = str(scoring.get("metric_name", "epipolar_consistency"))
    metric_mode = str(scoring.get("metric_mode", "min"))
    min_gap = float(scoring.get("min_score_gap", 0.0))
    threshold = scoring.get("winner_score_threshold")
    task = str(group.get("task") or cfg.get("project", {}).get("task", "t2v")).lower()
    valid: list[dict[str, Any]] = []
    reject_reasons: list[str] = []
    for video in group.get("videos", []):
        if not isinstance(video, dict):
            reject_reasons.append("non_object_video")
            continue
        ok, reason = candidate_metric_valid(video, metric_name)
        if not ok:
            reject_reasons.append(reason or "metric_invalid")
            continue
        ok, reason = motion_filter_pass(video, cfg)
        if not ok:
            reject_reasons.append(reason or "motion_filter")
            continue
        valid.append(video)
    if len(valid) < 2:
        return None, ",".join(sorted(set(reject_reasons))) or "less_than_two_valid_candidates"

    reverse = metric_mode == "max"
    ordered = sorted(valid, key=lambda item: float(item[metric_name]), reverse=reverse)
    winner = ordered[0]
    loser = ordered[-1]
    winner_score = float(winner[metric_name])
    loser_score = float(loser[metric_name])
    score_gap = abs(winner_score - loser_score)
    if threshold is not None:
        threshold_value = float(threshold)
        if metric_mode == "min" and winner_score >= threshold_value:
            return None, "winner_threshold"
        if metric_mode == "max" and winner_score <= threshold_value:
            return None, "winner_threshold"
    if score_gap < min_gap:
        return None, "score_gap"
    if winner.get("video_path") == loser.get("video_path"):
        return None, "same_video_path"
    if winner.get("seed") == loser.get("seed") and winner.get("seed") is not None:
        return None, "same_seed"

    winner_id = winner.get("generation_id", f"seed_{winner.get('seed')}")
    loser_id = loser.get("generation_id", f"seed_{loser.get('seed')}")
    pair_id = safe_id(f"{group.get('group_id')}__{winner_id}__vs__{loser_id}")
    prompt = prompt_from_group(group)
    pair = {
        "pair_id": pair_id,
        "group_id": group.get("group_id"),
        "scene_uid": group.get("scene_uid"),
        "scene_id": group.get("scene_id"),
        "prompt": prompt,
        "text_prompt": prompt,
        "task": task,
        "source_split": group.get("source_split", "train"),
        "source_bucket": group.get("source_bucket", "8k"),
        "winner": dict(winner),
        "loser": dict(loser),
        "winner_score": winner_score,
        "loser_score": loser_score,
        "score_gap": score_gap,
        "metric_name": metric_name,
        "metric_mode": metric_mode,
        "latent_provenance": cfg.get("encoding", {}).get("latent_provenance", "posthoc_mp4_vae"),
    }
    for key in ("image_path", "image_prompt", "input_image_path", "first_frame_path", "first_frame_relpath", "camera_motion", "image_conditioned"):
        if group.get(key) is not None:
            pair[key] = group[key]
    return pair, None


def select_preference_pairs(scored_payload: Any, cfg: dict[str, Any], source_run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    groups = load_groups(scored_payload)
    scoring = cfg.get("scoring", {})
    metric_name = str(scoring.get("metric_name", "epipolar_consistency"))
    metric_mode = str(scoring.get("metric_mode", "min"))
    pairs: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    summary_counts: Counter[str] = Counter()
    candidate_total = 0
    scored_valid = 0
    scored_invalid = 0
    motion_metric = str(cfg.get("motion_filter", {}).get("metric_name", "motion_dynamics"))

    for group in groups:
        group_id = group.get("group_id")
        videos = group.get("videos", [])
        candidate_total += len(videos)
        group_valid_after_metric = 0
        group_valid_after_motion = 0
        metric_reasons: Counter[str] = Counter()
        motion_reasons: Counter[str] = Counter()
        for video in videos:
            ok, reason = candidate_metric_valid(video, metric_name)
            if ok:
                scored_valid += 1
                group_valid_after_metric += 1
            else:
                scored_invalid += 1
                metric_reasons[reason or "metric_invalid"] += 1
                continue
            ok, reason = motion_filter_pass(video, cfg)
            if ok:
                group_valid_after_motion += 1
            else:
                motion_reasons[reason or "motion_filter"] += 1
        pair, reason = build_pair_from_group(group, cfg)
        if pair is not None:
            pairs.append(pair)
            continue
        reason = reason or "filtered"
        filtered.append(
            {
                "group_id": group_id,
                "reason": reason,
                "metric_valid_candidates": group_valid_after_metric,
                "motion_valid_candidates": group_valid_after_motion,
                "metric_reasons": dict(metric_reasons),
                "motion_reasons": dict(motion_reasons),
            }
        )
        if "score_gap" in reason:
            summary_counts["groups_removed_small_gap"] += 1
        elif reason == "winner_threshold":
            summary_counts["groups_removed_winner_threshold"] += 1
        elif group_valid_after_motion < 2:
            summary_counts["groups_removed_insufficient_valid_candidates"] += 1
            if any("below_min" in key for key in motion_reasons):
                summary_counts["groups_removed_static"] += 1
            if motion_reasons:
                summary_counts["groups_removed_motion_filter"] += 1
        else:
            summary_counts["groups_removed_insufficient_valid_candidates"] += 1

    task = str(scored_payload.get("task") if isinstance(scored_payload, dict) else cfg.get("project", {}).get("task", "t2v")).lower()
    condition_schema = ["encoder_hidden_states"] if task == "t2v" else ["encoder_hidden_states", "image_latent"]
    pair_payload = {
        "method": "epipolar_dpo",
        "task": task,
        "base_path": str(source_run),
        "candidate_base_path": str(source_run),
        "source_run": str(source_run),
        "source_run_relpath": cfg.get("source", {}).get("run_relpath"),
        "source_candidate_manifest": str(source_manifest_path(cfg)),
        "source_candidate_manifest_relpath": cfg.get("source", {}).get("candidate_manifest_relpath"),
        "metric_name": metric_name,
        "metric_mode": metric_mode,
        "motion_metric_name": motion_metric,
        "latent_provenance": cfg.get("encoding", {}).get("latent_provenance", "posthoc_mp4_vae"),
        "condition_schema": condition_schema,
        "pairs": pairs,
        "filtered": filtered,
    }
    summary = {
        "method": "epipolar_dpo",
        "task": task,
        "groups_total": len(groups),
        "candidates_total": candidate_total,
        "scored_valid": scored_valid,
        "scored_invalid": scored_invalid,
        "groups_removed_static": int(summary_counts["groups_removed_static"]),
        "groups_removed_motion_filter": int(summary_counts["groups_removed_motion_filter"]),
        "groups_removed_insufficient_valid_candidates": int(summary_counts["groups_removed_insufficient_valid_candidates"]),
        "groups_removed_small_gap": int(summary_counts["groups_removed_small_gap"]),
        "groups_removed_winner_threshold": int(summary_counts["groups_removed_winner_threshold"]),
        "pairs_final": len(pairs),
        "metric_name": metric_name,
        "metric_mode": metric_mode,
        "motion_metric_name": motion_metric,
        "motion_filter": cfg.get("motion_filter", {}),
        "min_score_gap": scoring.get("min_score_gap", 0.0),
        "winner_score_threshold": scoring.get("winner_score_threshold"),
        "latent_provenance": cfg.get("encoding", {}).get("latent_provenance", "posthoc_mp4_vae"),
    }
    return pair_payload, summary


def write_run_config(run_dir: Path, cfg: dict[str, Any]) -> None:
    write_resolved_config(run_dir, cfg)
